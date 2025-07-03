#!/usr/bin/env python3
"""Train TFT model for multiple gauge hydrological forecasting."""

import argparse
import logging
from pathlib import Path
import time
from typing import List, Optional

import pandas as pd

from src.config.settings import Settings
from src.data.loaders import prepare_model_data, split_time_series
from src.data.preprocessors import validate_data_quality, fill_missing_values
from src.models.tft_model import TFTModelWrapper
from src.utils.helpers import (
    setup_logging, 
    set_random_seed, 
    ensure_directory,
    format_duration,
    print_model_summary,
    load_yaml_to_dict
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TFT model for multiple gauge hydrological forecasting"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multi_gauge.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--gauge-list",
        type=Path,
        help="Path to file containing list of gauge IDs (one per line)"
    )
    
    parser.add_argument(
        "--gauge-ids",
        nargs="+",
        help="List of gauge IDs to train model for"
    )
    
    parser.add_argument(
        "--max-gauges",
        type=int,
        help="Maximum number of gauges to use for training"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for models and results"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data quality validation"
    )
    
    parser.add_argument(
        "--parallel-training",
        action="store_true",
        help="Use parallel training for multiple gauges"
    )
    
    return parser.parse_args()


def load_gauge_list(args: argparse.Namespace) -> List[str]:
    """Load list of gauge IDs from arguments."""
    gauge_ids = []
    
    if args.gauge_list:
        # Load from file
        with open(args.gauge_list, 'r') as f:
            gauge_ids = [line.strip() for line in f if line.strip()]
    elif args.gauge_ids:
        # Use command line arguments
        gauge_ids = args.gauge_ids
    else:
        raise ValueError("Must specify either --gauge-list or --gauge-ids")
    
    # Limit number of gauges if specified
    if args.max_gauges and len(gauge_ids) > args.max_gauges:
        gauge_ids = gauge_ids[:args.max_gauges]
    
    return gauge_ids


def main() -> None:
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.output_dir / "logs" / "train_multi_gauge.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting multi-gauge TFT training")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # Load gauge list
        gauge_ids = load_gauge_list(args)
        logger.info(f"Training on {len(gauge_ids)} gauges: {gauge_ids[:5]}{'...' if len(gauge_ids) > 5 else ''}")
        
        # Load configuration
        if args.config.exists():
            settings = Settings.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            settings = Settings()
            logger.info("Using default configuration")
        
        # Update static parameters based on target
        settings.update_static_parameters()
        
        # Set random seed for reproducibility
        set_random_seed(settings.model.random_state)
        
        # Create output directories
        model_dir = ensure_directory(args.output_dir / "models")
        results_dir = ensure_directory(args.output_dir / "results")
        
        # Prepare data for all gauges
        logger.info("Loading and preparing data...")
        target_series, covariate_series = prepare_model_data(
            settings, 
            area_filter=gauge_ids
        )
        
        # Validate data quality for all series
        if not args.no_validate:
            logger.info("Validating data quality...")
            quality_report = validate_data_quality(
                target_series,
                min_length=settings.model.input_chunk_length + settings.model.output_chunk_length,
                max_missing_ratio=0.15  # Slightly more lenient for multi-gauge
            )
            
            if not quality_report["is_valid"]:
                logger.warning("Some data quality issues found:")
                for issue in quality_report["issues"]:
                    logger.warning(f"  - {issue}")
                
                # Filter out problematic series
                valid_indices = []
                for i, report in enumerate(quality_report["series_reports"]):
                    if not report["issues"]:
                        valid_indices.append(i)
                
                if len(valid_indices) == 0:
                    raise ValueError("No valid series after quality filtering")
                
                # Keep only valid series
                if isinstance(target_series, list):
                    target_series = [target_series[i] for i in valid_indices]
                    if covariate_series is not None:
                        covariate_series = [covariate_series[i] for i in valid_indices]
                
                logger.info(f"Kept {len(valid_indices)} valid series out of {len(gauge_ids)}")
            else:
                logger.info("Data quality validation passed for all series")
        
        # Fill missing values
        logger.info("Preprocessing data...")
        target_series = fill_missing_values(target_series, method="linear", limit=7)
        if covariate_series is not None:
            covariate_series = fill_missing_values(covariate_series, method="linear", limit=7)
        
        # Split data into train/validation/test sets
        logger.info("Splitting data...")
        train_target, val_target, test_target = split_time_series(
            target_series,
            train_split=settings.training.train_split,
            val_split=settings.training.val_split
        )
        
        train_covariates, val_covariates, test_covariates = None, None, None
        if covariate_series is not None:
            train_covariates, val_covariates, test_covariates = split_time_series(
                covariate_series,
                train_split=settings.training.train_split,
                val_split=settings.training.val_split
            )
        
        # Log data split info
        if isinstance(train_target, list):
            logger.info(f"Number of series: {len(train_target)}")
            logger.info(f"Average train length: {sum(len(ts) for ts in train_target) / len(train_target):.0f}")
            logger.info(f"Average validation length: {sum(len(ts) for ts in val_target) / len(val_target):.0f}")
            logger.info(f"Average test length: {sum(len(ts) for ts in test_target) / len(test_target):.0f}")
        else:
            logger.info(f"Train length: {len(train_target)}")
            logger.info(f"Validation length: {len(val_target)}")
            logger.info(f"Test length: {len(test_target)}")
        
        # Initialize and train model
        logger.info("Initializing TFT model...")
        model_wrapper = TFTModelWrapper(settings)
        
        logger.info("Starting model training...")
        model_wrapper.train(
            target_series=train_target,
            val_series=val_target,
            covariates=train_covariates,
            val_covariates=val_covariates
        )
        
        # Save model
        model_path = model_dir / "tft_multi_gauge.pkl"
        model_wrapper.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Make predictions on test set
        logger.info("Making predictions on test set...")
        
        # For multi-gauge, we need to predict for each series separately
        all_predictions = []
        all_targets = []
        
        if isinstance(test_target, list):
            for i, test_ts in enumerate(test_target):
                # Use corresponding train series as context
                train_context = train_target[i] if isinstance(train_target, list) else train_target
                cov_context = None
                if train_covariates is not None:
                    cov_context = train_covariates[i] if isinstance(train_covariates, list) else train_covariates
                
                pred = model_wrapper.predict(
                    series=train_context,
                    covariates=cov_context,
                    n=len(test_ts)
                )
                all_predictions.append(pred)
                all_targets.append(test_ts)
        else:
            pred = model_wrapper.predict(
                series=train_target,
                covariates=train_covariates,
                n=len(test_target)
            )
            all_predictions = [pred]
            all_targets = [test_target]
        
        # Evaluate model performance
        logger.info("Evaluating model performance...")
        metrics_df = model_wrapper.evaluate(all_predictions, all_targets)
        
        # Save results
        results_path = results_dir / "metrics_multi_gauge.csv"
        metrics_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
        
        # Calculate summary statistics
        summary_stats = {}
        for metric in ["NSE", "KGE", "RMSE"]:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    summary_stats[f"{metric}_mean"] = float(values.mean())
                    summary_stats[f"{metric}_median"] = float(values.median())
                    summary_stats[f"{metric}_std"] = float(values.std())
        
        # Save summary statistics
        summary_path = results_dir / "summary_multi_gauge.csv"
        pd.Series(summary_stats).to_csv(summary_path, header=["value"])
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {format_duration(elapsed_time)}")
        
        print_model_summary(model_path, summary_stats)
        
        # Print performance summary
        print("\nPERFORMANCE SUMMARY")
        print("-" * 20)
        for metric in ["NSE", "KGE", "RMSE"]:
            if f"{metric}_mean" in summary_stats:
                mean_val = summary_stats[f"{metric}_mean"]
                std_val = summary_stats.get(f"{metric}_std", 0)
                print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
        
        n_good_nse = len(metrics_df[metrics_df["NSE"] > 0.5]) if "NSE" in metrics_df.columns else 0
        n_total = len(metrics_df)
        print(f"Gauges with NSE > 0.5: {n_good_nse}/{n_total} ({100*n_good_nse/n_total:.1f}%)")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

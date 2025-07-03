#!/usr/bin/env python3
"""Train TFT model for single gauge hydrological forecasting."""

import argparse
import logging
from pathlib import Path
import time

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
    print_model_summary
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TFT model for single gauge hydrological forecasting"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/single_gauge.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--gauge-id",
        type=str,
        required=True,
        help="Gauge ID to train model for"
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
    
    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.output_dir / "logs" / f"train_single_{args.gauge_id}.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting single gauge TFT training")
    logger.info(f"Gauge ID: {args.gauge_id}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    try:
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
        
        # Prepare data for single gauge
        logger.info("Loading and preparing data...")
        target_series, covariate_series = prepare_model_data(
            settings, 
            area_filter=[args.gauge_id]
        )
        
        # Validate data quality
        if not args.no_validate:
            logger.info("Validating data quality...")
            quality_report = validate_data_quality(
                target_series,
                min_length=settings.model.input_chunk_length + settings.model.output_chunk_length,
                max_missing_ratio=0.1
            )
            
            if not quality_report["is_valid"]:
                logger.error("Data quality validation failed:")
                for issue in quality_report["issues"]:
                    logger.error(f"  - {issue}")
                raise ValueError("Data quality validation failed")
            
            logger.info("Data quality validation passed")
        
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
        model_path = model_dir / f"tft_single_{args.gauge_id}.pkl"
        model_wrapper.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Make predictions on test set
        logger.info("Making predictions on test set...")
        predictions = model_wrapper.predict(
            series=train_target,  # Use train data as context
            covariates=train_covariates,
            n=len(test_target)
        )
        
        # Evaluate model performance
        logger.info("Evaluating model performance...")
        metrics_df = model_wrapper.evaluate(predictions, test_target)
        
        # Save results
        results_path = results_dir / f"metrics_single_{args.gauge_id}.csv"
        metrics_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {format_duration(elapsed_time)}")
        
        # Get main metrics for summary
        main_metrics = {}
        if not metrics_df.empty:
            row = metrics_df.iloc[0]
            for metric in ["NSE", "KGE", "RMSE"]:
                if metric in row and not pd.isna(row[metric]):
                    main_metrics[metric] = float(row[metric])
        
        print_model_summary(model_path, main_metrics)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

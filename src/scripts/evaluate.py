#!/usr/bin/env python3
"""Evaluate trained TFT model performance."""

import argparse
import logging
from pathlib import Path
import time
from typing import List, Optional

import pandas as pd
import numpy as np

from src.config.settings import Settings
from src.data.loaders import prepare_model_data, split_time_series
from src.data.preprocessors import fill_missing_values
from src.models.tft_model import TFTModelWrapper
from src.evaluation.metrics import create_metrics_dataframe
from src.utils.helpers import (
    setup_logging, 
    ensure_directory,
    format_duration,
    validate_file_exists
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained TFT model performance"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model file"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--gauge-ids",
        nargs="+",
        help="List of gauge IDs to evaluate"
    )
    
    parser.add_argument(
        "--gauge-list",
        type=Path,
        help="Path to file containing list of gauge IDs (one per line)"
    )
    
    parser.add_argument(
        "--test-split",
        choices=["test", "validation", "all"],
        default="test",
        help="Which data split to evaluate on"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create evaluation plots"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions along with evaluation metrics"
    )
    
    return parser.parse_args()


def load_gauge_list(args: argparse.Namespace) -> Optional[List[str]]:
    """Load list of gauge IDs from arguments."""
    gauge_ids = None
    
    if args.gauge_list:
        # Load from file
        validate_file_exists(args.gauge_list, "Gauge list file")
        with open(args.gauge_list, 'r') as f:
            gauge_ids = [line.strip() for line in f if line.strip()]
    elif args.gauge_ids:
        # Use command line arguments
        gauge_ids = args.gauge_ids
    
    return gauge_ids


def create_evaluation_plots(
    predictions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create evaluation plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        
        # Metrics distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # NSE distribution
        if "NSE" in metrics_df.columns:
            metrics_df["NSE"].hist(bins=20, ax=axes[0,0], alpha=0.7)
            axes[0,0].axvline(metrics_df["NSE"].median(), color='red', linestyle='--', 
                            label=f'Median: {metrics_df["NSE"].median():.3f}')
            axes[0,0].set_title("NSE Distribution")
            axes[0,0].set_xlabel("NSE")
            axes[0,0].legend()
        
        # KGE distribution  
        if "KGE" in metrics_df.columns:
            metrics_df["KGE"].hist(bins=20, ax=axes[0,1], alpha=0.7)
            axes[0,1].axvline(metrics_df["KGE"].median(), color='red', linestyle='--',
                            label=f'Median: {metrics_df["KGE"].median():.3f}')
            axes[0,1].set_title("KGE Distribution")
            axes[0,1].set_xlabel("KGE")
            axes[0,1].legend()
        
        # RMSE distribution
        if "RMSE" in metrics_df.columns:
            metrics_df["RMSE"].hist(bins=20, ax=axes[1,0], alpha=0.7)
            axes[1,0].axvline(metrics_df["RMSE"].median(), color='red', linestyle='--',
                            label=f'Median: {metrics_df["RMSE"].median():.3f}')
            axes[1,0].set_title("RMSE Distribution")
            axes[1,0].set_xlabel("RMSE")
            axes[1,0].legend()
        
        # NSE vs KGE scatter
        if "NSE" in metrics_df.columns and "KGE" in metrics_df.columns:
            axes[1,1].scatter(metrics_df["NSE"], metrics_df["KGE"], alpha=0.6)
            axes[1,1].set_xlabel("NSE")
            axes[1,1].set_ylabel("KGE")
            axes[1,1].set_title("NSE vs KGE")
            
            # Add diagonal line
            lims = [
                np.min([axes[1,1].get_xlim(), axes[1,1].get_ylim()]),
                np.max([axes[1,1].get_xlim(), axes[1,1].get_ylim()])
            ]
            axes[1,1].plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {output_dir / 'evaluation_metrics.png'}")
        
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main() -> None:
    """Main evaluation function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.output_dir / "logs" / "evaluation.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting TFT model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test split: {args.test_split}")
    logger.info(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # Validate model file exists
        validate_file_exists(args.model_path, "Model file")
        
        # Load gauge list (optional)
        gauge_ids = load_gauge_list(args)
        if gauge_ids:
            logger.info(f"Evaluating {len(gauge_ids)} specific gauges")
        else:
            logger.info("Evaluating all available gauges")
        
        # Load configuration
        if args.config and args.config.exists():
            settings = Settings.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            settings = Settings()
            logger.info("Using default configuration")
        
        # Update static parameters based on target
        settings.update_static_parameters()
        
        # Create output directory
        ensure_directory(args.output_dir)
        
        # Initialize model wrapper and load trained model
        logger.info("Loading trained model...")
        model_wrapper = TFTModelWrapper(settings)
        model_wrapper.load_model(args.model_path)
        
        # Prepare data
        logger.info("Loading and preparing data...")
        target_series, covariate_series = prepare_model_data(
            settings, 
            area_filter=gauge_ids
        )
        
        # Fill missing values
        logger.info("Preprocessing data...")
        target_series = fill_missing_values(target_series, method="linear", limit=7)
        if covariate_series is not None:
            covariate_series = fill_missing_values(covariate_series, method="linear", limit=7)
        
        # Split data
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
        
        # Select evaluation dataset
        if args.test_split == "test":
            eval_target = test_target
            eval_covariates = test_covariates
            context_target = train_target
            context_covariates = train_covariates
        elif args.test_split == "validation":
            eval_target = val_target
            eval_covariates = val_covariates
            # Use train data as context
            context_target = train_target
            context_covariates = train_covariates
        else:  # "all"
            # Evaluate on entire dataset
            eval_target = target_series
            eval_covariates = covariate_series
            context_target = target_series
            context_covariates = covariate_series
        
        logger.info(f"Evaluating on {args.test_split} split")
        
        # Make predictions
        logger.info("Making predictions...")
        all_predictions = []
        all_targets = []
        
        if isinstance(eval_target, list):
            # Multiple series
            for i, target_ts in enumerate(eval_target):
                gauge_id = "unknown"
                if target_ts.static_covariates is not None:
                    gauge_id = str(target_ts.static_covariates.index[0])
                
                logger.info(f"Predicting for gauge {gauge_id} ({i+1}/{len(eval_target)})")
                
                # Get context data
                context_ts = context_target[i] if isinstance(context_target, list) else context_target
                context_cov = None
                if context_covariates is not None:
                    context_cov = context_covariates[i] if isinstance(context_covariates, list) else context_covariates
                
                # Make prediction
                pred = model_wrapper.predict(
                    series=context_ts,
                    covariates=context_cov,
                    n=len(target_ts)
                )
                
                all_predictions.append(pred)
                all_targets.append(target_ts)
        else:
            # Single series
            pred = model_wrapper.predict(
                series=context_target,
                covariates=context_covariates,
                n=len(eval_target)
            )
            all_predictions = [pred]
            all_targets = [eval_target]
        
        # Evaluate model performance
        logger.info("Calculating evaluation metrics...")
        metrics_df = model_wrapper.evaluate(all_predictions, all_targets)
        
        # Save detailed results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_path = args.output_dir / f"evaluation_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_data = []
            targets_data = []
            
            for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
                gauge_id = f"gauge_{i}"
                if target.static_covariates is not None:
                    gauge_id = str(target.static_covariates.index[0])
                
                # Convert to DataFrame
                pred_df = pred.pd_dataframe()
                pred_df["gauge_id"] = gauge_id
                pred_df["type"] = "prediction"
                
                target_df = target.pd_dataframe()
                target_df["gauge_id"] = gauge_id
                target_df["type"] = "target"
                
                predictions_data.append(pred_df)
                targets_data.append(target_df)
            
            # Combine and save
            all_data = pd.concat(predictions_data + targets_data, ignore_index=True)
            predictions_path = args.output_dir / f"predictions_vs_targets_{timestamp}.csv"
            all_data.to_csv(predictions_path)
            logger.info(f"Predictions saved to {predictions_path}")
        
        # Calculate summary statistics
        summary_stats = {}
        for metric in ["NSE", "KGE", "RMSE", "correlation", "alpha", "beta"]:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    summary_stats[f"{metric}_mean"] = float(values.mean())
                    summary_stats[f"{metric}_median"] = float(values.median())
                    summary_stats[f"{metric}_std"] = float(values.std())
                    summary_stats[f"{metric}_min"] = float(values.min())
                    summary_stats[f"{metric}_max"] = float(values.max())
        
        # Add performance categories
        if "NSE" in metrics_df.columns:
            nse_values = metrics_df["NSE"].dropna()
            summary_stats["n_excellent"] = int((nse_values > 0.75).sum())  # Excellent
            summary_stats["n_good"] = int(((nse_values > 0.5) & (nse_values <= 0.75)).sum())  # Good
            summary_stats["n_satisfactory"] = int(((nse_values > 0.25) & (nse_values <= 0.5)).sum())  # Satisfactory
            summary_stats["n_poor"] = int((nse_values <= 0.25).sum())  # Poor
            summary_stats["n_total"] = len(nse_values)
        
        # Save summary
        summary_path = args.output_dir / f"evaluation_summary_{timestamp}.csv"
        pd.Series(summary_stats).to_csv(summary_path, header=["value"])
        
        # Create plots if requested
        if args.create_plots and args.save_predictions:
            create_evaluation_plots(
                predictions_data[0] if predictions_data else pd.DataFrame(),
                targets_data[0] if targets_data else pd.DataFrame(),
                metrics_df,
                args.output_dir
            )
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation completed in {format_duration(elapsed_time)}")
        
        print("\nEVALUATION SUMMARY")
        print("=" * 50)
        print(f"Evaluated gauges: {len(metrics_df)}")
        print(f"Test split: {args.test_split}")
        
        for metric in ["NSE", "KGE", "RMSE"]:
            if f"{metric}_mean" in summary_stats:
                mean_val = summary_stats[f"{metric}_mean"]
                std_val = summary_stats.get(f"{metric}_std", 0)
                median_val = summary_stats[f"{metric}_median"]
                print(f"{metric}: {mean_val:.3f} ± {std_val:.3f} (median: {median_val:.3f})")
        
        if "n_total" in summary_stats:
            n_total = summary_stats["n_total"]
            n_good = summary_stats.get("n_excellent", 0) + summary_stats.get("n_good", 0)
            print(f"\nPerformance Categories (NSE):")
            print(f"  Excellent (>0.75): {summary_stats.get('n_excellent', 0)}/{n_total}")
            print(f"  Good (0.5-0.75): {summary_stats.get('n_good', 0)}/{n_total}")
            print(f"  Satisfactory (0.25-0.5): {summary_stats.get('n_satisfactory', 0)}/{n_total}")
            print(f"  Poor (≤0.25): {summary_stats.get('n_poor', 0)}/{n_total}")
            print(f"  Good or better: {n_good}/{n_total} ({100*n_good/n_total:.1f}%)")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

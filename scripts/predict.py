#!/usr/bin/env python3
"""Make predictions using trained TFT model."""

import argparse
import logging
from pathlib import Path
import time
from typing import List, Optional

import pandas as pd

from src.config.settings import Settings
from src.data.loaders import prepare_model_data, split_time_series
from src.data.preprocessors import fill_missing_values
from src.models.tft_model import TFTModelWrapper
from src.utils.helpers import (
    setup_logging, 
    ensure_directory,
    format_duration,
    validate_file_exists
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained TFT model"
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
        help="Path to configuration file (optional, inferred from model if not provided)"
    )
    
    parser.add_argument(
        "--gauge-ids",
        nargs="+",
        help="List of gauge IDs to make predictions for"
    )
    
    parser.add_argument(
        "--gauge-list",
        type=Path,
        help="Path to file containing list of gauge IDs (one per line)"
    )
    
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        help="Number of time steps to forecast (default: model output_chunk_length)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Output directory for predictions"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet", "both"],
        default="csv",
        help="Output format for predictions"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include confidence intervals in predictions (for quantile models)"
    )
    
    return parser.parse_args()


def load_gauge_list(args: argparse.Namespace) -> List[str]:
    """Load list of gauge IDs from arguments."""
    gauge_ids = []
    
    if args.gauge_list:
        # Load from file
        validate_file_exists(args.gauge_list, "Gauge list file")
        with open(args.gauge_list, 'r') as f:
            gauge_ids = [line.strip() for line in f if line.strip()]
    elif args.gauge_ids:
        # Use command line arguments
        gauge_ids = args.gauge_ids
    else:
        raise ValueError("Must specify either --gauge-list or --gauge-ids")
    
    return gauge_ids


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    output_format: str,
    filename_prefix: str = "predictions"
) -> None:
    """Save predictions to file."""
    ensure_directory(output_dir)
    
    if output_format in ["csv", "both"]:
        csv_path = output_dir / f"{filename_prefix}.csv"
        predictions.to_csv(csv_path, index=True)
        print(f"Predictions saved to {csv_path}")
    
    if output_format in ["parquet", "both"]:
        parquet_path = output_dir / f"{filename_prefix}.parquet"
        predictions.to_parquet(parquet_path, index=True)
        print(f"Predictions saved to {parquet_path}")


def main() -> None:
    """Main prediction function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.output_dir / "logs" / "predictions.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting TFT model predictions")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    start_time = time.time()
    
    try:
        # Validate model file exists
        validate_file_exists(args.model_path, "Model file")
        
        # Load gauge list
        gauge_ids = load_gauge_list(args)
        logger.info(f"Making predictions for {len(gauge_ids)} gauges")
        
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
        
        # Determine forecast horizon
        forecast_horizon = (args.forecast_horizon if args.forecast_horizon 
                          else settings.model.output_chunk_length)
        logger.info(f"Forecast horizon: {forecast_horizon} time steps")
        
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
        
        # Make predictions
        logger.info("Making predictions...")
        all_predictions = []
        
        if isinstance(target_series, list):
            # Multiple series
            for i, ts in enumerate(target_series):
                gauge_id = "unknown"
                if ts.static_covariates is not None:
                    gauge_id = str(ts.static_covariates.index[0])
                
                logger.info(f"Predicting for gauge {gauge_id} ({i+1}/{len(target_series)})")
                
                # Get corresponding covariates
                cov_ts = None
                if covariate_series is not None:
                    cov_ts = covariate_series[i] if isinstance(covariate_series, list) else covariate_series
                
                # Make prediction
                pred = model_wrapper.predict(
                    series=ts,
                    covariates=cov_ts,
                    n=forecast_horizon
                )
                
                # Convert to DataFrame for easier handling
                pred_df = pred.pd_dataframe()
                pred_df["gauge_id"] = gauge_id
                pred_df["prediction_step"] = range(1, len(pred_df) + 1)
                
                all_predictions.append(pred_df)
        else:
            # Single series
            gauge_id = "unknown"
            if target_series.static_covariates is not None:
                gauge_id = str(target_series.static_covariates.index[0])
            
            logger.info(f"Predicting for gauge {gauge_id}")
            
            pred = model_wrapper.predict(
                series=target_series,
                covariates=covariate_series,
                n=forecast_horizon
            )
            
            pred_df = pred.pd_dataframe()
            pred_df["gauge_id"] = gauge_id
            pred_df["prediction_step"] = range(1, len(pred_df) + 1)
            
            all_predictions.append(pred_df)
        
        # Combine all predictions
        logger.info("Combining predictions...")
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Reorder columns for clarity
        cols = ["gauge_id", "prediction_step"] + [c for c in combined_predictions.columns 
                                                if c not in ["gauge_id", "prediction_step"]]
        combined_predictions = combined_predictions[cols]
        
        # Save predictions
        logger.info("Saving predictions...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"predictions_{timestamp}"
        
        save_predictions(
            combined_predictions,
            args.output_dir,
            args.output_format,
            filename_prefix
        )
        
        # Save summary statistics
        summary_stats = {
            "n_gauges": len(gauge_ids),
            "forecast_horizon": forecast_horizon,
            "total_predictions": len(combined_predictions),
            "prediction_columns": list(combined_predictions.select_dtypes(include=[float, int]).columns)
        }
        
        summary_path = args.output_dir / f"prediction_summary_{timestamp}.yaml"
        from src.utils.helpers import save_dict_to_yaml
        save_dict_to_yaml(summary_stats, summary_path)
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info(f"Predictions completed in {format_duration(elapsed_time)}")
        
        print("\nPREDICTION SUMMARY")
        print("=" * 20)
        print(f"Gauges processed: {len(gauge_ids)}")
        print(f"Forecast horizon: {forecast_horizon} time steps")
        print(f"Total predictions: {len(combined_predictions)}")
        print(f"Output directory: {args.output_dir}")
        
        if args.include_confidence and len(combined_predictions.columns) > 3:
            print(f"Prediction columns: {list(combined_predictions.columns[2:])}")
        
        print("=" * 20)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

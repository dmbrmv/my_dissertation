#!/usr/bin/env python3
"""Complete workflow example for TFT hydrological forecasting."""

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
    print_model_summary,
    create_experiment_name
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete TFT workflow: train, predict, and evaluate"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/single_gauge.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--gauge-ids",
        nargs="+",
        required=True,
        help="List of gauge IDs to process"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for all results"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Custom experiment name (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data quality validation"
    )
    
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use reduced settings for quick testing"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main workflow function."""
    args = parse_arguments()
    
    # Create experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        tags = ["workflow"]
        if len(args.gauge_ids) == 1:
            tags.append("single")
        else:
            tags.append("multi")
        if args.quick_mode:
            tags.append("quick")
        experiment_name = create_experiment_name("tft", extra_tags=tags)
    
    # Setup output directory
    exp_output_dir = args.output_dir / experiment_name
    ensure_directory(exp_output_dir)
    
    # Setup logging
    log_file = exp_output_dir / "logs" / "workflow.log"
    setup_logging(level=args.log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TFT workflow: {experiment_name}")
    logger.info(f"Processing {len(args.gauge_ids)} gauge(s): {args.gauge_ids}")
    logger.info(f"Output directory: {exp_output_dir}")
    
    start_time = time.time()
    
    try:
        # Load configuration
        if args.config.exists():
            settings = Settings.from_yaml(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            settings = Settings()
            logger.info("Using default configuration")
        
        # Apply quick mode settings
        if args.quick_mode:
            logger.info("Applying quick mode settings...")
            settings.model.n_epochs = 5
            settings.model.batch_size = 8
            settings.model.hidden_size = 32
            settings.model.lstm_layers = 1
            settings.training.early_stopping_patience = 3
        
        # Update static parameters based on target
        settings.update_static_parameters()
        
        # Set random seed for reproducibility
        set_random_seed(settings.model.random_state)
        
        # Save configuration for this experiment
        config_save_path = exp_output_dir / "config_used.yaml"
        settings.to_yaml(config_save_path)
        logger.info(f"Saved configuration to {config_save_path}")
        
        # Create output subdirectories
        model_dir = ensure_directory(exp_output_dir / "models")
        results_dir = ensure_directory(exp_output_dir / "results")
        plots_dir = ensure_directory(exp_output_dir / "plots")
        
        # Step 1: Data Loading and Preparation
        logger.info("Step 1: Loading and preparing data...")
        target_series, covariate_series = prepare_model_data(
            settings, 
            area_filter=args.gauge_ids
        )
        
        # Step 2: Data Quality Validation
        if not args.skip_validation:
            logger.info("Step 2: Validating data quality...")
            quality_report = validate_data_quality(
                target_series,
                min_length=settings.model.input_chunk_length + settings.model.output_chunk_length,
                max_missing_ratio=0.1
            )
            
            # Save quality report
            quality_path = results_dir / "data_quality_report.yaml"
            from src.utils.helpers import save_dict_to_yaml
            save_dict_to_yaml(quality_report, quality_path)
            
            if not quality_report["is_valid"]:
                logger.warning("Data quality issues found:")
                for issue in quality_report["issues"]:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("Data quality validation passed")
        
        # Step 3: Data Preprocessing
        logger.info("Step 3: Preprocessing data...")
        target_series = fill_missing_values(target_series, method="linear", limit=7)
        if covariate_series is not None:
            covariate_series = fill_missing_values(covariate_series, method="linear", limit=7)
        
        # Step 4: Data Splitting
        logger.info("Step 4: Splitting data...")
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
        
        # Log data info
        if isinstance(train_target, list):
            logger.info(f"Number of series: {len(train_target)}")
            avg_train_len = sum(len(ts) for ts in train_target) / len(train_target)
            avg_val_len = sum(len(ts) for ts in val_target) / len(val_target)
            avg_test_len = sum(len(ts) for ts in test_target) / len(test_target)
            logger.info(f"Average lengths - Train: {avg_train_len:.0f}, Val: {avg_val_len:.0f}, Test: {avg_test_len:.0f}")
        else:
            logger.info(f"Single series - Train: {len(train_target)}, Val: {len(val_target)}, Test: {len(test_target)}")
        
        # Step 5: Model Training
        logger.info("Step 5: Training TFT model...")
        model_wrapper = TFTModelWrapper(settings)
        
        model_wrapper.train(
            target_series=train_target,
            val_series=val_target,
            covariates=train_covariates,
            val_covariates=val_covariates
        )
        
        # Save model
        model_path = model_dir / f"tft_model_{experiment_name}.pkl"
        model_wrapper.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training history
        history = model_wrapper.get_training_history()
        if history["train"]:
            history_df = pd.DataFrame(history)
            history_path = results_dir / "training_history.csv"
            history_df.to_csv(history_path, index_label="epoch")
            logger.info(f"Training history saved to {history_path}")
        
        # Step 6: Model Prediction
        logger.info("Step 6: Making predictions...")
        all_predictions = []
        all_targets = []
        
        if isinstance(test_target, list):
            for i, test_ts in enumerate(test_target):
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
        
        # Step 7: Model Evaluation
        logger.info("Step 7: Evaluating model performance...")
        metrics_df = model_wrapper.evaluate(all_predictions, all_targets)
        
        # Save detailed metrics
        metrics_path = results_dir / "evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        
        # Calculate and save summary statistics
        summary_stats = {}
        for metric in ["NSE", "KGE", "RMSE", "correlation"]:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    summary_stats[f"{metric}_mean"] = float(values.mean())
                    summary_stats[f"{metric}_median"] = float(values.median())
                    summary_stats[f"{metric}_std"] = float(values.std())
                    summary_stats[f"{metric}_min"] = float(values.min())
                    summary_stats[f"{metric}_max"] = float(values.max())
        
        summary_path = results_dir / "summary_statistics.csv"
        pd.Series(summary_stats).to_csv(summary_path, header=["value"])
        
        # Step 8: Save Predictions
        logger.info("Step 8: Saving predictions...")
        predictions_data = []
        
        for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
            gauge_id = f"gauge_{i}"
            if target.static_covariates is not None:
                gauge_id = str(target.static_covariates.index[0])
            
            # Combine predictions and targets
            pred_df = pred.pd_dataframe()
            pred_df["gauge_id"] = gauge_id
            pred_df["type"] = "prediction"
            
            target_df = target.pd_dataframe()
            target_df["gauge_id"] = gauge_id
            target_df["type"] = "target"
            
            predictions_data.extend([pred_df, target_df])
        
        # Save combined predictions
        all_data = pd.concat(predictions_data, ignore_index=True)
        predictions_path = results_dir / "predictions_and_targets.csv"
        all_data.to_csv(predictions_path)
        logger.info(f"Predictions saved to {predictions_path}")
        
        # Final Summary
        elapsed_time = time.time() - start_time
        logger.info(f"Workflow completed successfully in {format_duration(elapsed_time)}")
        
        # Print final summary
        print("\n" + "="*60)
        print(f"TFT WORKFLOW COMPLETED: {experiment_name}")
        print("="*60)
        print(f"Processed gauges: {len(args.gauge_ids)}")
        print(f"Total runtime: {format_duration(elapsed_time)}")
        print(f"Output directory: {exp_output_dir}")
        
        if summary_stats:
            print("\nPerformance Summary:")
            print("-" * 20)
            for metric in ["NSE", "KGE", "RMSE"]:
                if f"{metric}_mean" in summary_stats:
                    mean_val = summary_stats[f"{metric}_mean"]
                    std_val = summary_stats.get(f"{metric}_std", 0)
                    print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
        
        print(f"\nFiles generated:")
        print(f"  - Model: {model_path}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Predictions: {predictions_path}")
        print(f"  - Config: {config_save_path}")
        
        if history["train"]:
            print(f"  - Training history: {history_path}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Workflow failed - {str(e)}")
        print(f"Check log file: {log_file}")
        raise


if __name__ == "__main__":
    main()

import sys
import os
from pathlib import Path
import pandas as pd
import mlflow
from imblearn.over_sampling import SMOTE

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline.tabular_modeling import build_preprocessor, build_classification_models, split_features_target
from src.pipeline.experiment_tracking import (
    run_experiment, 
    register_best_model,
    run_cross_validation,
    compare_models,
    select_best_model
)


def main():
    raw_path = PROJECT_ROOT / "data" / "processed" / "creditcard_processed.csv"
    if not raw_path.exists():
        print(f"Data file not found at {raw_path}")
        return

    print(f"Loading credit card data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Optional quick run downsampling via env var QUICK or --quick flag
    if os.environ.get("QUICK") == "1" or "--quick" in sys.argv:
        n = min(80000, len(df))
        df = df.sample(n=n, random_state=42)
        print(f"[quick] Downsampled to {len(df)} rows for faster iteration")

    # Ensure target is int {0,1}
    if df["Class"].dtype != int:
        df["Class"] = df["Class"].astype(int)

    # Feature/target split
    drop_cols = []  # nothing special to drop; all features are numeric already
    X_train, X_test, y_train, y_test = split_features_target(
        df, target="Class", drop_cols=drop_cols, test_size=0.2, random_state=42, stratify=True
    )

    # Print class distribution before resampling (train split only)
    train_dist = y_train.value_counts().sort_index()
    train_pct = (y_train.value_counts(normalize=True).sort_index() * 100).round(3)
    print("Training class distribution BEFORE resampling:")
    print({int(k): int(v) for k, v in train_dist.items()})
    print({int(k): float(v) for k, v in train_pct.items()})

    # Apply SMOTE only on the training split to handle class imbalance
    # Note: SMOTE operates on numeric features; creditcard.csv features are numeric.
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Print class distribution after resampling
    res_dist = y_train_res.value_counts().sort_index()
    res_pct = (y_train_res.value_counts(normalize=True).sort_index() * 100).round(3)
    print("Training class distribution AFTER SMOTE resampling:")
    print({int(k): int(v) for k, v in res_dist.items()})
    print({int(k): float(v) for k, v in res_pct.items()})

    # Determine columns for preprocessor
    # For creditcard.csv everything except target is numeric
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = []

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # Build models (LogReg, DT, RF, GBDT, optional XGB)
    models = build_classification_models(preprocessor=preprocessor)

    # MLflow local tracking
    experiment_name = "CreditCard_Fraud_Models"
    mlflow.set_tracking_uri("file:./mlruns")

    # Check if cross-validation mode is enabled
    use_cv = "--use-cv" in sys.argv
    
    # Store results for comparison
    all_model_results = []

    for model_name, model_pipeline in models.items():
        print(f"Running experiment for {model_name}...")

        # Small param grids
        param_grid = None
        if "log_reg" in model_name:
            param_grid = {"model__C": [0.1, 1.0, 3.0]}
        elif "decision_tree" in model_name:
            param_grid = {"model__max_depth": [5, 10, None], "model__min_samples_split": [2, 5]}
        elif "random_forest" in model_name:
            param_grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
        elif "gradient_boosting" in model_name:
            param_grid = {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}
        elif "xgb" in model_name:
            param_grid = {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1]}

        if use_cv:
            # Cross-validation mode
            print(f"  Running 5-fold cross-validation for {model_name}...")
            cv_results = run_cross_validation(
                model=model_pipeline,
                X=X_train_res,
                y=y_train_res,
                cv=5,
                random_state=42
            )
            
            # Log CV results to MLflow
            with mlflow.start_run(run_name=f"{model_name}__smote__cv"):
                mlflow.log_param("cv_folds", cv_results['n_folds'])
                mlflow.log_param("resampling", "SMOTE")
                
                # Log aggregated metrics
                for metric, stats in cv_results['aggregated'].items():
                    mlflow.log_metric(f"{metric}_mean", stats['mean'])
                    mlflow.log_metric(f"{metric}_std", stats['std'])
                
                # Log individual fold results
                for fold_idx, fold_metrics in enumerate(cv_results['fold_results']):
                    for metric, value in fold_metrics.items():
                        mlflow.log_metric(f"fold_{fold_idx}_{metric}", value)
                
                mlflow.sklearn.log_model(model_pipeline, "model")
            
            print(f"  CV Results for {model_name}:")
            for metric, stats in cv_results['aggregated'].items():
                print(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            all_model_results.append({
                'model_name': f"{model_name}__smote",
                'metrics': cv_results,
                'is_cv': True
            })
        else:
            # Standard train/test mode
            # Include resampling info in run name for traceability
            trained_model, metrics = run_experiment(
                experiment_name=experiment_name,
                model_name=f"{model_name}__smote",
                model=model_pipeline,
                X_train=X_train_res,
                y_train=y_train_res,
                X_test=X_test,
                y_test=y_test,
                param_grid=param_grid,
                search_type="grid",
            )
            
            all_model_results.append({
                'model_name': f"{model_name}__smote",
                'metrics': metrics,
                'is_cv': False
            })

    # Model comparison and selection
    print("\n" + "="*80)
    print("MODEL COMPARISON AND SELECTION")
    print("="*80)
    
    comparison_df = compare_models(all_model_results)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison report
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    comparison_path = output_dir / "model_comparison_creditcard.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison report saved to: {comparison_path}")
    
    # Select best model with interpretability consideration
    selected_model, justification = select_best_model(
        all_model_results,
        primary_metric='f1',
        performance_threshold=0.02
    )
    
    print("\n" + "-"*80)
    print("FINAL MODEL SELECTION")
    print("-"*80)
    print(f"Selected Model: {selected_model}")
    print(f"Justification: {justification}")
    print("-"*80)

    print("\nRegistering best model by F1...")
    register_best_model(experiment_name, metric="f1_score")


if __name__ == "__main__":
    print("evidence that raining is done for task 2")
    main()

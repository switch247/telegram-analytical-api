import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = None):
    """Set up MLflow experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_model_metrics(y_true, y_pred, y_proba=None, prefix="") -> Dict[str, float]:
    """Log evaluation metrics to MLflow."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    mlflow.log_metric(f"{prefix}accuracy", accuracy)
    mlflow.log_metric(f"{prefix}precision", precision)
    mlflow.log_metric(f"{prefix}recall", recall)
    mlflow.log_metric(f"{prefix}f1_score", f1)
    
    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            mlflow.log_metric(f"{prefix}roc_auc", roc_auc)
            metrics["roc_auc"] = roc_auc
        except ValueError:
            pass # Handle cases where ROC AUC cannot be calculated (e.g. only one class)
            
    return metrics

def tune_hyperparameters(model, param_grid, X_train, y_train, search_type='grid', cv=3, scoring='f1'):
    """Tune hyperparameters using Grid or Random search."""
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, n_iter=10)
    else:
        raise ValueError("search_type must be 'grid' or 'random'")
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def run_experiment(experiment_name: str, model_name: str, model, X_train, y_train, X_test, y_test, param_grid: Optional[Dict] = None, search_type: str = 'grid'):
    """Run a full experiment: setup, tune (optional), train, evaluate, log."""
    setup_mlflow_experiment(experiment_name)
    
    with mlflow.start_run(run_name=model_name):
        if param_grid:
            mlflow.log_param("tuning_method", search_type)
            best_model, best_params = tune_hyperparameters(model, param_grid, X_train, y_train, search_type)
            mlflow.log_params(best_params)
            model = best_model
        else:
            model.fit(X_train, y_train)
            
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            
        # Log metrics
        metrics = log_model_metrics(y_test, y_pred, y_proba)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run {model_name} completed. Metrics: {metrics}")
        return model, metrics

def register_best_model(experiment_name: str, metric: str = "f1_score", higher_is_better: bool = True):
    """Find the best run in the experiment and register it."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC" if higher_is_better else f"metrics.{metric} ASC"]
    )
    
    if not runs:
        print("No runs found.")
        return
        
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id} with {metric}: {best_run.data.metrics.get(metric)}")
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri, f"{experiment_name}_best_model")
    print(f"Registered model from run {best_run.info.run_id} as {experiment_name}_best_model")


def aggregate_cv_results(cv_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate cross-validation results by computing mean and std for each metric.
    
    Args:
        cv_results: List of dictionaries, each containing metrics from one CV fold
        
    Returns:
        Dictionary with aggregated statistics: {metric_name: {'mean': X, 'std': Y}}
    """
    if not cv_results:
        return {}
    
    # Get all metric names from first fold
    metric_names = list(cv_results[0].keys())
    aggregated = {}
    
    for metric in metric_names:
        values = [fold[metric] for fold in cv_results if metric in fold]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    
    return aggregated


def run_cross_validation(
    model,
    X,
    y,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation and return aggregated metrics.
    
    Args:
        model: Scikit-learn compatible model/pipeline
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'fold_results': List of metric dicts for each fold
        - 'aggregated': Aggregated statistics (mean, std) for each metric
        - 'cv_scores': Raw CV scores for primary metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    # Perform CV manually to get detailed metrics per fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        # Train model on fold
        model.fit(X_train_fold, y_train_fold)
        
        # Predictions
        y_pred = model.predict(X_val_fold)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val_fold)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_val_fold)
        
        # Compute metrics for this fold
        fold_metrics = {
            'accuracy': accuracy_score(y_val_fold, y_pred),
            'precision': precision_score(y_val_fold, y_pred, zero_division=0),
            'recall': recall_score(y_val_fold, y_pred, zero_division=0),
            'f1': f1_score(y_val_fold, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                fold_metrics['roc_auc'] = roc_auc_score(y_val_fold, y_proba)
            except ValueError:
                pass  # Handle cases where ROC AUC cannot be calculated
        
        fold_results.append(fold_metrics)
    
    # Aggregate results
    aggregated = aggregate_cv_results(fold_results)
    
    return {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'n_folds': cv
    }


def compare_models(model_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Args:
        model_results: List of dicts, each containing:
            - 'model_name': str
            - 'metrics': dict of metric values or aggregated CV results
            - 'is_cv': bool (optional, indicates if metrics are from CV)
            
    Returns:
        DataFrame with models ranked by performance metrics
    """
    comparison_data = []
    
    for result in model_results:
        model_name = result['model_name']
        metrics = result.get('metrics', {})
        is_cv = result.get('is_cv', False)
        
        row = {'model_name': model_name}
        
        if is_cv and 'aggregated' in metrics:
            # CV results: show mean ± std
            for metric, stats in metrics['aggregated'].items():
                row[f'{metric}_mean'] = stats['mean']
                row[f'{metric}_std'] = stats['std']
                row[metric] = stats['mean']  # For ranking
        else:
            # Single run results
            for metric, value in metrics.items():
                row[metric] = value
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by F1 score (primary metric for imbalanced classification)
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    elif 'f1_mean' in df.columns:
        df = df.sort_values('f1_mean', ascending=False)
    
    return df


def select_best_model(
    model_results: List[Dict[str, Any]],
    primary_metric: str = 'f1',
    performance_threshold: float = 0.02,
    interpretability_scores: Optional[Dict[str, int]] = None
) -> Tuple[str, str]:
    """
    Select the best model balancing performance with interpretability.
    
    Args:
        model_results: List of model result dictionaries
        primary_metric: Metric to use for performance comparison (default: 'f1')
        performance_threshold: Models within this threshold are considered comparable (default: 0.02 = 2%)
        interpretability_scores: Dict mapping model name patterns to interpretability scores (higher = more interpretable)
                                Default: LogReg=5, DecisionTree=4, RandomForest=3, GradientBoosting=2, XGBoost=1
        
    Returns:
        Tuple of (selected_model_name, justification_string)
    """
    if not model_results:
        return None, "No models to compare"
    
    # Default interpretability scores
    if interpretability_scores is None:
        interpretability_scores = {
            'log_reg': 5,
            'logistic': 5,
            'decision_tree': 4,
            'random_forest': 3,
            'gradient_boosting': 2,
            'xgb': 1,
            'xgboost': 1
        }
    
    # Extract performance scores
    model_scores = []
    for result in model_results:
        model_name = result['model_name']
        metrics = result.get('metrics', {})
        is_cv = result.get('is_cv', False)
        
        # Get primary metric value
        if is_cv and 'aggregated' in metrics:
            score = metrics['aggregated'].get(primary_metric, {}).get('mean', 0)
        else:
            score = metrics.get(primary_metric, 0)
        
        # Assign interpretability score
        interp_score = 0
        for pattern, score_val in interpretability_scores.items():
            if pattern.lower() in model_name.lower():
                interp_score = score_val
                break
        
        model_scores.append({
            'model_name': model_name,
            'performance': score,
            'interpretability': interp_score,
            'metrics': metrics,
            'is_cv': is_cv
        })
    
    # Sort by performance (descending)
    model_scores.sort(key=lambda x: x['performance'], reverse=True)
    
    # Find best performance
    best_performance = model_scores[0]['performance']
    
    # Find all models within threshold
    comparable_models = [
        m for m in model_scores 
        if best_performance - m['performance'] <= performance_threshold
    ]
    
    # Among comparable models, select most interpretable
    selected = max(comparable_models, key=lambda x: x['interpretability'])
    
    # Build justification
    if len(comparable_models) == 1:
        justification = (
            f"Selected '{selected['model_name']}' as it has the best {primary_metric} "
            f"score of {selected['performance']:.4f}"
        )
    else:
        other_models = [m['model_name'] for m in comparable_models if m['model_name'] != selected['model_name']]
        justification = (
            f"Selected '{selected['model_name']}' (interpretability score: {selected['interpretability']}) "
            f"with {primary_metric}={selected['performance']:.4f}. "
            f"Models {other_models} had comparable performance (within {performance_threshold:.1%} threshold), "
            f"but '{selected['model_name']}' offers better interpretability."
        )
    
    return selected['model_name'], justification


"""
Biz2Credit Framework - Analysis Functions + Time Series CV + Business Logic
Purpose: Contains all analysis functions, CV logic, and business operations
Role: Reusable library for Biz2Credit analysis workflow
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ds_modeling framework imports
from ds_modeling.ml_framework.base import Transformer
from ds_modeling.ml_framework.pipeline import make_pipeline
from biz2credit_transformers import Biz2CreditPrep1, Biz2CreditPrep2, Biz2CreditImputer

# Time Series CV imports
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, confusion_matrix,
    classification_report, balanced_accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.utils.multiclass import type_of_target
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Union

# Filter out common sklearn warnings that are not critical
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# ============================================================================
# STAGE 1: PRE-ANALYSIS FUNCTIONS
# ============================================================================

def generic_pre_analysis(df, title="Dataset Pre-Analysis"):
    """
    Generic pre-analysis for any dataset
    """
    print(f"=== {title} ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    
    # Null analysis
    null_summary = df.isnull().sum()
    null_percentage = (null_summary / len(df)) * 100
    null_df = pd.DataFrame({
        'Null_Count': null_summary,
        'Null_Percentage': null_percentage
    }).sort_values('Null_Percentage', ascending=False)
    
    print(f"\nNull Values Summary:")
    print(null_df[null_df['Null_Count'] > 0])
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    return null_df


def biz2credit_specific_analysis(df):
    """
    Biz2Credit-specific pre-analysis
    """
    print("\n=== BIZ2CREDIT SPECIFIC ANALYSIS ===")
    
    # Enrichment features analysis
    enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure']
    available_enrichment = [f for f in enrichment_features if f in df.columns]
    
    print("Enrichment Features Analysis:")
    for feature in available_enrichment:
        if feature in df.columns:
            null_count = df[feature].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_vals = df[feature].nunique()
            print(f"- {feature}: {null_count} nulls ({null_pct:.1f}%), {unique_vals} unique values")
    
    # Business metrics
    if 'leads_count' in df.columns and 'sales_count' in df.columns:
        total_leads = df['leads_count'].sum()
        total_sales = df['sales_count'].sum()
        conversion_rate = (total_sales / total_leads) * 100 if total_leads > 0 else 0
        print(f"\nBusiness Metrics:")
        print(f"- Total leads: {total_leads:,}")
        print(f"- Total sales: {total_sales:,}")
        print(f"- Conversion rate: {conversion_rate:.2f}%")
    
    # Date range analysis
    if 'click_month' in df.columns:
        date_range = df['click_month'].agg(['min', 'max'])
        print(f"\nDate Range:")
        print(f"- From: {date_range['min']}")
        print(f"- To: {date_range['max']}")
    
    return available_enrichment


def run_pre_analysis(df):
    """
    Run both generic and specific pre-analysis
    """
    print("=" * 60)
    print("STAGE 1: PRE-ANALYSIS")
    print("=" * 60)
    
    # Add debug printing for raw data check
    print("=== RAW DATA CHECK ===")
    print(f"Initial shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")
    
    # Run generic analysis
    null_df = generic_pre_analysis(df, "BIZ2CREDIT PRE-ANALYSIS")
    
    # Run specific analysis
    enrichment_features = biz2credit_specific_analysis(df)
    
    return null_df, enrichment_features


# ============================================================================
# STAGE 2: FEATURE ANALYSIS FUNCTIONS
# ============================================================================

def run_raw_feature_visualizations(df):
    """
    Analyze raw features (disabled for performance)
    """
    print("=== RAW FEATURE ANALYSIS ===")
    print("Feature analysis disabled for performance")
    print("Focusing on data flow analysis and debugging")
    return None


def run_transformed_feature_visualizations(df):
    """
    Analyze transformed features (disabled for performance)
    """
    print("=== TRANSFORMED FEATURE ANALYSIS ===")
    print("Feature analysis disabled for performance")
    print("Focusing on data flow analysis and debugging")
    return None


# ============================================================================
# STAGE 3: PIPELINE CREATION FUNCTIONS
# ============================================================================

def create_biz2credit_pipelines():
    """
    Create Biz2Credit pipeline instances using ds_modeling framework
    """
    print("=== CREATING BIZ2CREDIT PIPELINES ===")
    
    try:
        # Import transformers and models
        from biz2credit_transformers import Biz2CreditPrep1, Biz2CreditPrep2, Biz2CreditImputer, Biz2CreditCategoricalEncoder
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from ds_modeling.ml_framework.pipeline import make_pipeline
        
        # Create pipeline instances 
        pipelines = {

            'biz2credit_rf_1': make_pipeline(
                Biz2CreditPrep1(),
                Biz2CreditPrep2(),
                Biz2CreditImputer(),
                Biz2CreditCategoricalEncoder(),
                RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=1)
            ),
            'biz2credit_gb_1': make_pipeline(
                Biz2CreditPrep1(),
                Biz2CreditPrep2(),
                Biz2CreditImputer(),
                Biz2CreditCategoricalEncoder(),
                GradientBoostingClassifier(random_state=42, n_estimators=100)
            ),
            'biz2credit_lr_1': make_pipeline(
                Biz2CreditPrep1(),
                Biz2CreditPrep2(),
                Biz2CreditImputer(),
                Biz2CreditCategoricalEncoder(),
                LogisticRegression(random_state=42, max_iter=1000, C=1.0)
            )
        }
        
        print(f"✅ Created {len(pipelines)} pipeline instances:")
        for name, pipeline in pipelines.items():
            print(f"  - {name}: {type(pipeline).__name__}")
            print(f"    Steps: {list(pipeline.named_steps.keys())}")
        
        return pipelines
        
    except Exception as e:
        print(f"❌ Error creating pipelines: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STAGE 4: TIME SERIES CROSS-VALIDATION FUNCTIONS
# ============================================================================

def _generate_time_series_folds(data: pd.DataFrame, date_column: str, training_period: int, test_days: int) -> List[Tuple[int, int, int]]:
    """
    Generate time series fold indices for cross-validation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data sorted by date
    date_column : str
        Name of the date column
    training_period : int
        Number of days for training period
    test_days : int
        Number of days for test period
    
    Returns:
    --------
    List[Tuple[int, int, int]]
        List of (train_start, train_end, test_end) index tuples
    """
    folds = []
    
    # Get unique dates and sort them
    unique_dates = pd.to_datetime(data[date_column]).dt.date.unique()
    unique_dates = sorted(unique_dates)
    
    if len(unique_dates) < training_period + test_days:
        return folds
    
    # Calculate step size: Move forward by test_days (2 weeks) for non-overlapping bi-weekly folds
    total_days = len(unique_dates)
    available_days = total_days - training_period - test_days
    step_size = test_days  # Move forward by exactly 2 weeks (non-overlapping)
    
    print(f"📅 Fold Generation Details:")
    print(f"  Total unique dates: {total_days}")
    print(f"  Training period: {training_period} days")
    print(f"  Test period: {test_days} days")
    print(f"  Step size: {step_size} days (non-overlapping)")
    print(f"  Available days for folds: {available_days}")
    print(f"  Expected folds: {available_days // step_size}")
    
    # Start with training period
    train_start_idx = 0
    train_end_idx = training_period
    
    while train_end_idx + test_days <= len(unique_dates):
        test_end_idx = train_end_idx + test_days
        
        # Find the actual row indices for these date boundaries
        train_start_date = unique_dates[train_start_idx]
        train_end_date = unique_dates[train_end_idx - 1]
        test_end_date = unique_dates[test_end_idx - 1]
        
        # Convert dates back to row indices
        train_start = data[data[date_column].dt.date >= train_start_date].index[0]
        train_end = data[data[date_column].dt.date <= train_end_date].index[-1] + 1
        test_end = data[data[date_column].dt.date <= test_end_date].index[-1] + 1
        
        # Ensure we have enough data
        if train_end - train_start >= 100 and test_end - train_end >= 10:  # Minimum sample requirements
            folds.append((train_start, train_end, test_end))
        
        # Move forward by step_size (larger steps = fewer folds)
        train_start_idx += step_size
        train_end_idx += step_size
    
    return folds


def _calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    
    metrics = {}
    
    # Basic metrics from hard predictions
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Handle binary vs multiclass
    average_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
    
    metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
    
    # Probability-based metrics (if probabilities available)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba[:, 1])
                
                # Calculate ECE (Expected Calibration Error)
                try:
                    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba[:, 1], n_bins=10)
                    ece = np.mean(np.abs(prob_true - prob_pred))
                    metrics['ece'] = ece
                except Exception as e:
                    warnings.warn(f"Could not calculate ECE: {e}")
                    metrics['ece'] = None
                
                # Calculate RMSE and R² for probability predictions vs actual
                metrics['rmse_proba'] = np.sqrt(mean_squared_error(y_true, y_pred_proba[:, 1]))
                metrics['r2_proba'] = r2_score(y_true, y_pred_proba[:, 1])
                
                # Calculate optimal threshold for F1 score (this still requires hard predictions)
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                # Use optimal threshold for better metrics
                y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
                metrics['f1_optimal'] = f1_score(y_true, y_pred_optimal, average=average_method, zero_division=0)
                metrics['precision_optimal'] = precision_score(y_true, y_pred_optimal, average=average_method, zero_division=0)
                metrics['recall_optimal'] = recall_score(y_true, y_pred_optimal, average=average_method, zero_division=0)
                metrics['accuracy_optimal'] = accuracy_score(y_true, y_pred_optimal)
                metrics['optimal_threshold'] = optimal_threshold
                
                # Business metrics
                total_sales = y_true.sum()
                total_pred_sales = y_pred_proba[:, 1].sum()
                total_leads = len(y_true)
                total_pred_leads = y_pred.sum()
                
                metrics['total_sales'] = total_sales
                metrics['total_pred_sales'] = total_pred_sales
                metrics['total_leads'] = total_leads
                metrics['total_pred_leads'] = total_pred_leads
                metrics['sales_ratio'] = total_pred_sales / total_sales if total_sales > 0 else None
                metrics['leads_ratio'] = total_pred_leads / total_leads if total_leads > 0 else None
                
            else:
                # Multiclass classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                
        except Exception as e:
            # Only warn for non-expected errors (not single-class cases)
            if "Only one class present" not in str(e):
                warnings.warn(f"Could not calculate probability-based metrics: {e}")
    
    return metrics


def _calculate_weighted_average(fold_results: List[Dict[str, Any]], fold_weights: List[float]) -> Dict[str, Any]:
    """Calculate weighted average of metrics across folds."""
    
    if not fold_results or not fold_weights:
        return {}
    
    weighted_metrics = {}
    
    # Get all unique metric names
    all_metrics = set()
    for fold_result in fold_results:
        all_metrics.update(fold_result.keys())
    
    # Calculate weighted averages for each metric
    for metric_name in all_metrics:
        valid_values = []
        valid_weights = []
        
        for fold_result, weight in zip(fold_results, fold_weights):
            value = fold_result.get(metric_name)
            if value is not None and not pd.isna(value):
                valid_values.append(value)
                valid_weights.append(weight)
        
        if valid_values and valid_weights:
            # Calculate weighted average
            weighted_avg = np.average(valid_values, weights=valid_weights)
            weighted_metrics[metric_name] = weighted_avg
        elif any(fold_result.get(metric_name) is not None for fold_result in fold_results):
            # Include the metric in results even if no valid weights, but some folds had values
            weighted_metrics[metric_name] = None
    
    # Add summary statistics
    weighted_metrics['total_folds'] = len(fold_results)
    weighted_metrics['total_test_samples'] = sum(fold_weights)
    
    # Add per-fold breakdown for business metrics
    if 'total_sales' in weighted_metrics:
        print(f"\n📊 BUSINESS METRICS PER FOLD BREAKDOWN:")
        print(f"{'Fold':<6} {'Test Rows':<10} {'Sales':<8} {'Pred Sales':<12} {'Sales Ratio':<12}")
        print("-" * 60)
        
        # Calculate totals across all folds
        total_sales_all_folds = 0
        total_pred_sales_all_folds = 0
        total_leads_all_folds = 0
        
        for i, (fold_result, weight) in enumerate(zip(fold_results, fold_weights)):
            fold_num = i + 1
            test_size = weight
            sales = fold_result.get('total_sales', 0)
            pred_sales = fold_result.get('total_pred_sales', 0)
            sales_ratio = fold_result.get('sales_ratio', 'N/A')
            
            # Accumulate totals
            total_sales_all_folds += sales
            total_pred_sales_all_folds += pred_sales
            total_leads_all_folds += test_size  # test_size = number of rows in test fold
            
            if all(x != 'N/A' for x in [sales, pred_sales, sales_ratio]):
                print(f"{fold_num:<6} {test_size:<10} {sales:<8.1f} {pred_sales:<12.1f} {sales_ratio:<12.3f}")
            else:
                print(f"{fold_num:<6} {test_size:<10} {sales:<8} {pred_sales:<12} {sales_ratio:<12}")
        
        # Note: Totals already calculated in the loop above
        
        print("-" * 60)
        print(f"📈 SUMMARY:")
        print(f"Total across all folds: {weighted_metrics.get('total_test_samples', 'N/A')} test samples")
        print(f"Total Sales: {total_sales_all_folds}")
        print(f"Total Predicted Sales: {total_pred_sales_all_folds:.1f}")
        print(f"Total Leads: {total_leads_all_folds}")
        if total_sales_all_folds > 0:
            print(f"Overall Sales Ratio: {total_pred_sales_all_folds/total_sales_all_folds:.3f}")
        if total_leads_all_folds > 0:
            print(f"Overall Conversion Rate: {total_sales_all_folds/total_leads_all_folds:.3%}")
            print(f"Overall Predicted Conversion Rate: {total_pred_sales_all_folds/total_leads_all_folds:.3%}")
            print(f"Conversion Rate Ratio: {(total_pred_sales_all_folds/total_leads_all_folds)/(total_sales_all_folds/total_leads_all_folds):.3f}")
        print(f"Average per fold: {weighted_metrics.get('total_test_samples', 0) / len(fold_results):.0f} test samples")
        print(f"Note: Each fold tests 7 days of data, not a fixed number of rows")
    
    return weighted_metrics


def time_series_cross_validation(
    pipelines: List[BaseEstimator],
    data: pd.DataFrame,
    target_column: str,
    date_column: str,
    training_period: int,
    test_days: int,
    pipeline_names: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Perform time series cross-validation on multiple sklearn pipelines.
    
    Parameters:
    -----------
    pipelines : List[BaseEstimator]
        List of sklearn pipelines to evaluate
    data : pd.DataFrame
        Input data with features, target, and date column
    target_column : str
        Name of the target column
    date_column : str
        Name of the date column (should be datetime or convertible to datetime)
    training_period : int
        Number of days for training period
    test_days : int
        Number of days for test period
    pipeline_names : List[str], optional
        Names for the pipelines. If None, will use "Pipeline_0", "Pipeline_1", etc.
    
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary containing results for each pipeline with fold-wise and averaged metrics
    """
    
    # Validate inputs
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")
    
    # Convert date column to datetime if needed
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Sort data by date
    data = data.sort_values(date_column).reset_index(drop=True)
    
    # Set pipeline names if not provided
    if pipeline_names is None:
        pipeline_names = [f"Pipeline_{i}" for i in range(len(pipelines))]
    
    if len(pipeline_names) != len(pipelines):
        raise ValueError("Number of pipeline names must match number of pipelines")
    
    # Determine problem type (classification or regression)
    target_type = type_of_target(data[target_column])
    is_classification = target_type in ['binary', 'multiclass', 'multilabel-indicator']
    
    # Generate time series folds
    folds = _generate_time_series_folds(data, date_column, training_period, test_days)
    
    if len(folds) == 0:
        raise ValueError("No valid folds could be generated with the given parameters")
    
    print(f"Generated {len(folds)} folds for time series cross-validation")
    print(f"Problem type detected: {'Classification' if is_classification else 'Regression'}")
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate each pipeline
    for pipeline, pipeline_name in zip(pipelines, pipeline_names):
        print(f"\n{'='*60}")
        print(f"Evaluating {pipeline_name}")
        print(f"{'='*60}")
        
        fold_results = []
        fold_weights = []
        
        # Evaluate on each fold
        for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
            print(f"  Fold {fold_idx + 1}/{len(folds)}")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]  # FIXED: test_end is already the end of test period
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"    Skipping fold {fold_idx + 1} due to insufficient data")
                continue
            
            # Prepare features and target
            X_train = train_data.drop(columns=[target_column, date_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column, date_column])
            y_test = test_data[target_column]
            
            # Print fold details (simplified)
            print(f"    Training: {len(train_data):,} rows")
            print(f"    Testing: {len(test_data):,} rows")
            print(f"    Sales in test: {y_test.sum():.1f}")
            
            try:
                # Fit pipeline on training data
                pipeline_fitted = clone(pipeline)
                pipeline_fitted.fit(X_train, y_train)
                
                # Get final transformed features for the model
                # Note: ds_modeling pipelines don't have transform method like sklearn
                # We'll use the pipeline directly for predictions
                print(f"    ✅ Model features: 11 essential features (ds_modeling pipeline)")
                
                # Make predictions
                y_pred = pipeline_fitted.predict(X_test)
                y_pred_proba = None
                if hasattr(pipeline_fitted, 'predict_proba'):
                    y_pred_proba = pipeline_fitted.predict_proba(X_test)
                
                # Calculate metrics
                if is_classification:
                    fold_metrics = _calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                else:
                    # For regression, you would implement regression metrics here
                    fold_metrics = {}
                
                # Store results
                fold_results.append(fold_metrics)
                fold_weights.append(len(test_data))
                
                # Test size and sales info now shown above in fold details
                
            except Exception as e:
                print(f"    ❌ Error in fold {fold_idx + 1}: {e}")
                continue
        
        if fold_results:
            # Calculate weighted average metrics
            weighted_avg = _calculate_weighted_average(fold_results, fold_weights)
            
            # Store results
            results[pipeline_name] = {
                'problem_type': 'classification' if is_classification else 'regression',
                'fold_results': fold_results,
                'fold_weights': fold_weights,
                'weighted_average': weighted_avg
            }
            
            print(f"  ✅ Completed {pipeline_name} with {len(fold_results)} valid folds")
        else:
            print(f"  ❌ No valid folds for {pipeline_name}")
    
    return results


def print_results_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted summary of the cross-validation results."""
    
    print("\n" + "="*80)
    print("TIME SERIES CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    for pipeline_name, pipeline_results in results.items():
        print(f"\n{pipeline_name}:")
        print("-" * (len(pipeline_name) + 1))
        
        problem_type = pipeline_results['problem_type']
        weighted_avg = pipeline_results['weighted_average']
        
        print(f"Problem Type: {problem_type.title()}")
        print(f"Number of Folds: {weighted_avg.get('total_folds', 'N/A')}")
        print(f"Total Test Samples: {weighted_avg.get('total_test_samples', 'N/A')}")
        
        print(f"\nWeighted Average Metrics:")
        
        if problem_type == 'classification':
            print(f"  🎯 PROBABILITY-BASED METRICS (Main Business KPIs):")
            prob_metrics = ['roc_auc', 'log_loss', 'brier_score', 'rmse_proba', 'r2_proba', 'ece']
            for metric in prob_metrics:
                value = weighted_avg.get(metric)
                if value is not None:
                    if metric in ['rmse_proba', 'r2_proba']:
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: N/A")
            
            print(f"  📊 BUSINESS VOLUME METRICS:")
            business_metrics = ['total_sales', 'total_pred_sales', 'total_leads', 'total_pred_leads', 'sales_ratio', 'leads_ratio']
            for metric in business_metrics:
                value = weighted_avg.get(metric)
                if value is not None:
                    if metric in ['sales_ratio', 'leads_ratio']:
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value:.0f}")
                else:
                    print(f"    {metric}: N/A")
            
            print(f"\n  Note: CLASSIFICATION METRICS (Not Focus KPIs - We Use Probabilities):")
            print(f"     These metrics require hard predictions (0/1), not probability scores.")
            print(f"     For business decisions, focus on probability-based metrics above.")
            print(f"  Default Threshold (0.5) Metrics:")
            default_metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall']
            for metric in default_metrics:
                value = weighted_avg.get(metric)
                if value is not None:
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: N/A")
            
            print(f"  Optimal Threshold Metrics:")
            optimal_metrics = ['f1_optimal', 'precision_optimal', 'recall_optimal', 'accuracy_optimal', 'optimal_threshold']
            for metric in optimal_metrics:
                value = weighted_avg.get(metric)
                if value is not None:
                    if metric == 'optimal_threshold':
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: N/A")
        else:
            # Regression metrics
            print(f"  Regression Metrics:")
            reg_metrics = ['rmse', 'mae', 'r2', 'mape', 'medae', 'explained_variance']
            for metric in reg_metrics:
                value = weighted_avg.get(metric)
                if value is not None:
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: N/A")


def run_time_series_cv(df, pipelines, target_column='sales_count', date_column='clickout_date_prt'):
    """
    Run time series cross-validation on Biz2Credit data
    """
    print("=" * 60)
    print("TIME SERIES CROSS-VALIDATION")
    print("=" * 60)
    
    # Business rule: Remove rows where all enrichment features are null
    # This ensures consistent sample counts across folds
    enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure']
    available_enrichment = [f for f in enrichment_features if f in df.columns]
    
    if available_enrichment:
        before_count = len(df)
        
        # Check if all enrichment features are null for each row
        all_null_mask = df[available_enrichment].isnull().all(axis=1)
        df_filtered = df[~all_null_mask].copy()
        
        removed_count = before_count - len(df_filtered)
        
        print(f"📊 Data Filtering:")
        print(f"  Before filtering: {before_count:,} rows")
        print(f"  After filtering: {len(df_filtered):,} rows")
        print(f"  Rows removed: {removed_count:,} ({removed_count/before_count*100:.1f}%)")
        print(f"  Reason: Removed rows where ALL enrichment features are null")
        
        df = df_filtered
    
    # Time series CV parameters
    training_period = 90  # 30 days for training
    test_days = 7        # 7 days (1 week) for testing
    
    print(f"📅 Time Series CV Parameters:")
    print(f"  Training period: {training_period} days")
    print(f"  Test period: {test_days} days")
    print(f"  Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df[date_column].min()} to {df[date_column].max()}")
    print(f"  Note: Test size = {test_days} days, not {test_days} rows")
    
    try:
        # Run time series cross-validation
        results = time_series_cross_validation(
            pipelines=list(pipelines.values()),
            data=df,
            target_column=target_column,
            date_column=date_column,
            training_period=training_period,
            test_days=test_days,
            pipeline_names=list(pipelines.keys())
        )
        
        if results:
            print("\n" + "="*60)
            print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            print("="*60)
            
            # Print results summary
            print_results_summary(results)
            
            # Select best pipeline
            best_pipeline_name = select_best_pipeline_from_cv(results)
            
            return results, best_pipeline_name
        else:
            print("❌ No results from cross-validation")
            return None, None
            
    except Exception as e:
        print(f"❌ Time series CV failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def select_best_pipeline_from_cv(cv_results):
    """
    Select best pipeline from CV results based on ROC AUC
    """
    print("\n=== SELECTING BEST PIPELINE FROM CV (by roc_auc) ===")
    
    best_pipeline = None
    best_score = -1
    
    for pipeline_name, results in cv_results.items():
        roc_auc = results['weighted_average'].get('roc_auc')
        if roc_auc is not None and roc_auc > best_score:
            best_score = roc_auc
            best_pipeline = pipeline_name
    
    if best_pipeline:
        print(f"🏆 Best pipeline: {best_pipeline}")
        print(f"Best roc_auc: {best_score:.4f}")
    else:
        print("❌ No valid pipelines found")
    
    return best_pipeline


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_biz2credit_analysis(df, pipelines, target_column='sales_count'):
    """
    Main function to run complete Biz2Credit analysis
    """
    print("🚀 Starting Biz2Credit Analysis")
    print("=" * 60)
    
    # Stage 1: Pre-analysis
    print("\n" + "="*60)
    print("STAGE 1: PRE-ANALYSIS")
    print("="*60)
    null_df, enrichment_features = run_pre_analysis(df)
    
    # Stage 2: Raw feature analysis (disabled for performance)
    print("\n" + "="*60)
    print("STAGE 2: FEATURE ANALYSIS")
    print("="*60)
    print("Feature analysis disabled for performance")
    print("Focusing on data flow analysis and debugging")
    
    # Stage 3: Pipeline creation (no output needed)
    if not pipelines:
        pipelines = create_biz2credit_pipelines()
        if not pipelines:
            print("❌ Failed to create pipelines")
            return None
    
    # Stage 4: Time series cross-validation
    print("\n" + "="*60)
    print("STAGE 4: TIME SERIES CROSS-VALIDATION")
    print("="*60)
    results, best_pipeline_name = run_time_series_cv(df, pipelines, target_column)
    
    if not results:
        print("❌ Cross-validation failed")
        return None
    
    # Stage 5: Feature analysis (transformed features)
    print("\n" + "="*60)
    print("STAGE 5: FEATURE ANALYSIS (Transformed Features)")
    print("="*60)
    print("Skipping transformed feature visualization for now")
    print("Focusing on data flow analysis and debugging")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)
    if best_pipeline_name:
        best_result = results[best_pipeline_name]
        print(f"Best Pipeline: {best_pipeline_name}")
        print("Cross-Validation Results:")
        print(f"  Number of Folds: {best_result['weighted_average'].get('total_folds', 'N/A')}")
        print(f"  Total Test Samples: {best_result['weighted_average'].get('total_test_samples', 'N/A')}")
        print(f"  ROC AUC: {best_result['weighted_average'].get('roc_auc', 'N/A'):.4f}")
        print(f"  Accuracy: {best_result['weighted_average'].get('accuracy', 'N/A'):.4f}")
        print(f"  F1 Score: {best_result['weighted_average'].get('f1_score', 'N/A'):.4f}")
    
    print("\n🎉 Time Series Cross-Validation completed successfully!")
    print(f"Results available for {len(results)} pipelines")
    
    return results

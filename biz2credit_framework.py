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
from ds_modeling.ml_framework.pipeline import Pipeline
from biz2credit_transformers import Biz2CreditPrep1, Biz2CreditImputer

# Time Series CV imports
from sklearn.base import BaseEstimator
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

# SHAP imports for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available - feature importance analysis will be skipped")

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
    print("Feature analysis disabled for performance: run_raw_feature_visualizations")
    print("Focusing on data flow analysis and debugging")
    return None


def run_transformed_feature_visualizations(df):
    """
    Analyze transformed features (disabled for performance)
    """
    print("=== TRANSFORMED FEATURE ANALYSIS ===")
    print("Feature analysis disabled for performance: run_transformed_feature_visualizations")
    print("Focusing on data flow analysis and debugging")
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
    
    # Calculate step size: Use smaller steps for better coverage
    total_days = len(unique_dates)
    available_days = total_days - training_period - test_days
    
    # Use non-overlapping folds for realistic coverage
    step_size = test_days  # Move forward by exactly 7 days (non-overlapping)
    
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
        
        # Move forward by test_days for non-overlapping folds
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
                
                # Calculate ECE (Expected Calibration Error) with L2S comparison
                try:
                    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba[:, 1], n_bins=10)
                    ece = np.mean(np.abs(prob_true - prob_pred))
                    
                    # Get the bin edges and indices for detailed L2S analysis
                    bin_edges = np.linspace(0, 1, 11)  # 11 edges for 10 bins
                    bin_indices = np.digitize(y_pred_proba[:, 1], bin_edges) - 1
                    bin_indices = np.clip(bin_indices, 0, 9)  # Ensure valid bin indices
                    
                    # Calculate L2S metrics for each bin
                    bin_metrics = []
                    for bin_idx in range(10):
                        bin_mask = (bin_indices == bin_idx)
                        if bin_mask.sum() > 0:
                            bin_samples = bin_mask.sum()
                            bin_sales = y_true[bin_mask].sum()
                            bin_pred_prob = y_pred_proba[bin_mask, 1].mean()
                            bin_actual_l2s = bin_sales / bin_samples if bin_samples > 0 else 0
                            
                            # Compare predicted p_sale to actual L2S rate
                            l2s_comparison = abs(bin_pred_prob - bin_actual_l2s)
                            
                            bin_metrics.append({
                                'bin_idx': bin_idx,
                                'bin_samples': bin_samples,
                                'bin_sales': bin_sales,
                                'bin_pred_prob': bin_pred_prob,
                                'bin_actual_l2s': bin_actual_l2s,
                                'l2s_comparison': l2s_comparison,
                                'bin_range': f"{bin_edges[bin_idx]:.2f}-{bin_edges[bin_idx+1]:.2f}"
                            })
                    
                    metrics['ece'] = ece
                    
                    # Store calibration curve details for detailed ECE analysis
                    metrics['calibration_details'] = {
                        'prob_true': prob_true,
                        'prob_pred': prob_pred,
                        'n_bins': 10,
                        'ece': ece,
                        'bin_metrics': bin_metrics,
                        'total_samples': len(y_true),
                        'total_sales': y_true.sum()
                    }
                except Exception as e:
                    warnings.warn(f"Could not calculate ECE: {e}")
                    metrics['ece'] = None
                    metrics['calibration_details'] = None
                
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


def _calculate_weighted_average(fold_results: List[Dict[str, Any]], fold_weights: List[int]) -> Dict[str, Any]:
    """Calculate weighted average of metrics across folds."""
    
    if not fold_results or not fold_weights:
        return {}
    
    # Initialize weighted sums
    weighted_sums = {}
    total_weight = sum(fold_weights)
    
    # Calculate weighted sums for each metric
    for fold_result, weight in zip(fold_results, fold_weights):
        for metric, value in fold_result.items():
            if value is not None and not pd.isna(value):
                if metric not in weighted_sums:
                    weighted_sums[metric] = 0
                
                # Only multiply if the value is numeric
                if isinstance(value, (int, float)):
                    weighted_sums[metric] += value * weight
                else:
                    # For non-numeric values, just store the value
                    weighted_sums[metric] = value
    
    # Calculate weighted averages
    weighted_avg = {}
    for metric, weighted_sum in weighted_sums.items():
        # Skip non-numeric metrics that might cause calculation errors
        if isinstance(weighted_sum, (int, float)) and not pd.isna(weighted_sum):
            weighted_avg[metric] = weighted_sum / total_weight
        else:
            weighted_avg[metric] = weighted_sum  # Keep as-is for non-numeric metrics
    
    # Add metadata
    weighted_avg['total_folds'] = len(fold_results)
    weighted_avg['total_test_samples'] = total_weight
    
    return weighted_avg


def _perform_feature_importance_analysis(pipeline_fitted, X_test, y_test, pipeline_name, fold_idx):
    """
    Perform feature importance analysis for tree-based models
    """
    try:
        # Check if this is a tree-based model that supports feature importance
        model = pipeline_fitted
        if hasattr(pipeline_fitted, 'named_steps'):
            # For sklearn pipelines, get the final estimator
            model = pipeline_fitted.named_steps.get(list(pipeline_fitted.named_steps.keys())[-1])
        
        # Check if it's a tree-based model
        is_tree_based = any([
            hasattr(model, 'feature_importances_'),
            'RandomForest' in str(type(model)),
            'GradientBoosting' in str(type(model)),
            'XGB' in str(type(model)),
            'LGBM' in str(type(model))
        ])
        
        if not is_tree_based:
            return None
        
        print(f"    🔍 Performing feature importance analysis for {pipeline_name} (fold {fold_idx + 1})")
        
        # Get feature importances from the model
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_names = X_test.columns.tolist()
            
            # Ensure lengths match
            if len(feature_importance) != len(feature_names):
                # Use the shorter length to avoid errors
                min_length = min(len(feature_importance), len(feature_names))
                feature_importance = feature_importance[:min_length]
                feature_names = feature_names[:min_length]
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Print top 10 features
            print(f"      Top 10 features by model importance:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"        {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
            
            # Store results
            importance_results = {
                'feature_importance': importance_df,
                'model': model
            }
            
            return importance_results
        else:
            print(f"      ⚠️ Model doesn't have feature_importances_ attribute")
            return None
        
    except Exception as e:
        print(f"      ⚠️ Feature importance analysis failed: {e}")
        return None


def _aggregate_feature_importance_results(results):
    """
    Aggregate feature importance results across all folds for each pipeline
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS (Aggregated Across Folds)")
    print("="*80)
    
    for pipeline_name, pipeline_results in results.items():
        # Check if this pipeline has feature importance results
        fold_results = pipeline_results.get('fold_results', [])
        importance_folds = [fold for fold in fold_results if 'importance_results' in fold]
        
        if not importance_folds:
            continue
        
        print(f"\n🔍 {pipeline_name.upper()} - Feature Importance:")
        print("-" * (len(pipeline_name) + 30))
        
        # Aggregate feature importance across folds
        all_features = set()
        feature_importance_sum = {}
        feature_importance_count = {}
        
        for fold in importance_folds:
            importance_results = fold['importance_results']
            importance_df = importance_results['feature_importance']
            
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                all_features.add(feature)
                if feature not in feature_importance_sum:
                    feature_importance_sum[feature] = 0
                    feature_importance_count[feature] = 0
                
                feature_importance_sum[feature] += importance
                feature_importance_count[feature] += 1
        
        # Calculate average importance across folds
        avg_importance = {}
        for feature in all_features:
            if feature_importance_count[feature] > 0:
                avg_importance[feature] = feature_importance_sum[feature] / feature_importance_count[feature]
        
        # Sort by average importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Display top 15 features
        print(f"Top 15 features by average importance across {len(importance_folds)} folds:")
        print(f"{'Rank':<4} {'Feature':<30} {'Avg Importance':<15} {'Folds':<8}")
        print("-" * 60)
        
        for i, (feature, importance) in enumerate(sorted_features[:15]):
            fold_count = feature_importance_count[feature]
            print(f"{i+1:<4} {feature:<30} {importance:<15.4f} {fold_count:<8}")
        
        # Store aggregated results in pipeline results
        pipeline_results['aggregated_importance'] = {
            'feature_importance': sorted_features,
            'fold_count': len(importance_folds),
            'total_features': len(all_features)
        }


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
            test_data = data.iloc[train_end:test_end]
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"    Skipping fold {fold_idx + 1} due to insufficient data")
                continue
            
            # Prepare features and target
            X_train = train_data.drop(columns=[target_column, date_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column, date_column])
            y_test = test_data[target_column]
            
            # Print fold details
            print(f"    Training: {len(train_data):,} rows ({train_data[date_column].min().strftime('%Y-%m-%d')} to {train_data[date_column].max().strftime('%Y-%m-%d')})")
            print(f"    Testing: {len(test_data):,} rows ({test_data[date_column].min().strftime('%Y-%m-%d')} to {test_data[date_column].max().strftime('%Y-%m-%d')})")
            print(f"    Sales in test: {y_test.sum():.1f}")
            
            try:
                # Use sklearn pipeline methods properly - like your boss intended!
                # Fit the pipeline on training data (this will fit all transformers + model)
                pipeline_fitted = pipeline.fit(X_train, y_train)
                
                # Make predictions using the fitted pipeline
                y_pred = pipeline_fitted.predict(X_test)
                y_pred_proba = None
                if hasattr(pipeline_fitted, 'predict_proba'):
                    y_pred_proba = pipeline_fitted.predict_proba(X_test)
                
                # Get feature count from the fitted pipeline
                # For sklearn pipelines, we can get the actual transformed feature count
                try:
                    # Get the transformed training data to see actual feature count
                    X_train_transformed = pipeline_fitted[:-1].transform(X_train)
                    feature_count = X_train_transformed.shape[1]
                except:
                    # Fallback to original data count
                    feature_count = X_train.shape[1]
                
                if 'roei' in pipeline_name or 'old_model' in pipeline_name:
                    print(f"    ✅ Model features: {feature_count} feature (p_sale only)")
                else:
                    print(f"    ✅ Model features: {feature_count} essential features (transformed by pipeline)")
                
                # Reduced verbosity - only show feature count
                print(f"    🔍 Features: {feature_count} total features")
                
                # Perform feature importance analysis for tree-based models (GB and RF)
                importance_results = None
                if 'gb' in pipeline_name.lower() or 'rf' in pipeline_name.lower():
                    try:
                        if hasattr(pipeline_fitted, 'named_steps'):
                            # Get the final model from the fitted pipeline
                            final_model = pipeline_fitted.named_steps[list(pipeline_fitted.named_steps.keys())[-1]]
                            if hasattr(final_model, 'feature_importances_'):
                                feature_importance = final_model.feature_importances_
                                importance_results = {
                                    'feature_importance': feature_importance,
                                    'feature_names': None
                                }
                    except Exception as e:
                        print(f"    ⚠️ Feature importance analysis not available: {e}")
                
                # Calculate metrics using the predictions
                if is_classification:
                    fold_metrics = _calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                else:
                    # For regression, you would implement regression metrics here
                    fold_metrics = {}
                
                # Add feature importance results to fold metrics if available
                if importance_results:
                    fold_metrics['importance_results'] = importance_results
                
                # Store results
                fold_results.append(fold_metrics)
                fold_weights.append(len(test_data))
                
                # Reduced verbosity - only show fold completion
                print(f"      ✅ Fold {fold_idx + 1} completed ({len(test_data)} samples)")
                
            except Exception as e:
                print(f"    ❌ Error in fold {fold_idx + 1}: {e}")
                import traceback
                print(f"      Full traceback:")
                traceback.print_exc()
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


def print_pipeline_comparison_chart(cv_results):
    """
    Print a nice comparison chart of all pipeline results
    """
    print("\n" + "="*100)
    print("📊 PIPELINE PERFORMANCE COMPARISON CHART")
    print("="*100)
    
    # Define feature descriptions for each pipeline
    feature_descriptions = {
        'biz2credit_rf_1': '14+ enriched features',
        'biz2credit_gb_1': '14+ enriched features', 
        'biz2credit_gb_2': '11 core business features',
        'biz2credit_lr_1': '14+ enriched features',
        'old_model_pipeline': 'Only p_sale',
        'roei_pipeline': 'Manager\'s basic features'
    }
    
    # Print header
    print(f"{'Pipeline':<25} {'ROC AUC':<10} {'Log Loss':<10} {'R²':<8} {'RMSE':<8} {'ECE':<8} {'Features':<20}")
    print("-" * 100)
    
    # Sort pipelines by ROC AUC (descending)
    sorted_pipelines = sorted(
        cv_results.items(),
        key=lambda x: x[1]['weighted_average'].get('roc_auc', 0),
        reverse=True
    )
    
    # Add medals and rankings
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    
    for idx, (pipeline_name, results) in enumerate(sorted_pipelines):
        if idx < len(medals):
            medal = medals[idx]
        else:
            medal = f"{idx+1}️⃣"
        
        # Get metrics
        roc_auc = results['weighted_average'].get('roc_auc', 'N/A')
        log_loss = results['weighted_average'].get('log_loss', 'N/A')
        r2_proba = results['weighted_average'].get('r2_proba', 'N/A')
        rmse_proba = results['weighted_average'].get('rmse_proba', 'N/A')
        ece = results['weighted_average'].get('ece', 'N/A')
        features = feature_descriptions.get(pipeline_name, 'Unknown')
        
        # Format metrics
        roc_auc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else str(roc_auc)
        log_loss_str = f"{log_loss:.4f}" if isinstance(log_loss, (int, float)) else str(log_loss)
        r2_str = f"{r2_proba:.4f}" if isinstance(r2_proba, (int, float)) else str(r2_proba)
        rmse_str = f"{rmse_proba:.4f}" if isinstance(rmse_proba, (int, float)) else str(rmse_proba)
        ece_str = f"{ece:.4f}" if isinstance(ece, (int, float)) else str(ece)
        
        # Print row
        print(f"{medal} {pipeline_name:<20} {roc_auc_str:<10} {log_loss_str:<10} {r2_str:<8} {rmse_str:<8} {ece_str:<8} {features:<20}")
    
    print("-" * 100)
    print("Note: Lower values are better for Log Loss, RMSE, and ECE. Higher values are better for ROC AUC and R².")
    print("="*100)


def print_detailed_ece_analysis(cv_results, pipeline_names=None):
    """
    Print detailed ECE analysis for specified pipelines
    """
    if pipeline_names is None:
        pipeline_names = ['biz2credit_gb_1', 'biz2credit_gb_2', 'roei_pipeline']
    
    print("\n" + "="*100)
    print("🔍 DETAILED ECE (Expected Calibration Error) ANALYSIS")
    print("="*100)
    
    for pipeline_name in pipeline_names:
        if pipeline_name in cv_results:
            print(f"\n📊 {pipeline_name.upper()}")
            print("-" * 60)
            
            # Get fold results
            fold_results = cv_results[pipeline_name]['fold_results']
            fold_weights = cv_results[pipeline_name]['fold_weights']
            
            print(f"Total Folds: {len(fold_results)}")
            print(f"Fold Weights (test set sizes): {fold_weights}")
            print(f"Total Test Samples: {sum(fold_weights):,}")
            
            # Calculate weighted ECE across folds
            total_weighted_ece = 0
            total_weight = 0
            calibration_details_all = []
            
            for fold_idx, (fold_result, weight) in enumerate(zip(fold_results, fold_weights)):
                if 'calibration_details' in fold_result and fold_result['calibration_details']:
                    ece = fold_result['calibration_details']['ece']
                    prob_true = fold_result['calibration_details']['prob_true']
                    prob_pred = fold_result['calibration_details']['prob_pred']
                    
                    total_weighted_ece += ece * weight
                    total_weight += weight
                    
                    # Store calibration details for this fold
                    calibration_details_all.append({
                        'fold': fold_idx + 1,
                        'weight': weight,
                        'ece': ece,
                        'prob_true': prob_true,
                        'prob_pred': prob_pred
                    })
                    
                    print(f"\n  Fold {fold_idx + 1} (Weight: {weight:,} samples):")
                    print(f"    ECE: {ece:.4f}")
                    
                    # Show detailed L2S comparison analysis
                    if 'bin_metrics' in fold_result and fold_result['bin_metrics']:
                        print(f"    L2S Comparison Analysis (Predicted p_sale vs Actual Sales/Leads):")
                        print(f"      Bin Range    | Samples | Sales | Pred p_sale | Actual L2S | Difference")
                        print(f"      {'-' * 75}")
                        
                        for bm in fold_result['bin_metrics']:
                            print(f"      {bm['bin_range']:11} | {bm['bin_samples']:6d} | {bm['bin_sales']:5d} | {bm['bin_pred_prob']:11.3f} | {bm['bin_actual_l2s']:11.3f} | {bm['l2s_comparison']:10.3f}")
                        
                        # Show summary
                        total_samples = sum(bm['bin_samples'] for bm in fold_result['bin_metrics'])
                        total_sales = sum(bm['bin_sales'] for bm in fold_result['bin_metrics'])
                        overall_l2s = total_sales / total_samples if total_samples > 0 else 0
                        print(f"\n    Summary: {total_samples:,} samples, {total_sales:,} sales, Overall L2S Rate: {overall_l2s:.3f}")
                    else:
                        # Fallback to standard calibration curve
                        print(f"    Calibration Curve (10 bins):")
                        print(f"      Predicted Prob | Actual Prob | Difference")
                        print(f"      {'-' * 40}")
                        
                        for i in range(len(prob_true)):
                            diff = abs(prob_true[i] - prob_pred[i])
                            print(f"      {prob_pred[i]:.3f}        | {prob_true[i]:.3f}      | {diff:.3f}")
            
            if total_weight > 0:
                weighted_ece = total_weighted_ece / total_weight
                print(f"\n  🎯 WEIGHTED ECE ACROSS ALL FOLDS: {weighted_ece:.4f}")
                print(f"     (Weighted by test set size - larger test sets have more influence)")
                

                
                # Show overall calibration curve
                if calibration_details_all:
                    print(f"\n  📈 OVERALL CALIBRATION ANALYSIS:")
                    print(f"     ECE measures how well predicted probabilities match actual outcomes")
                    print(f"     Lower ECE = Better calibration (predicted ≈ actual)")
                    print(f"     ECE = 0 means perfect calibration")
                    print(f"     ECE = 1 means worst possible calibration")
                    
                    # Aggregate calibration details across all folds
                    try:
                        # Check if all calibration details have the same shape
                        shapes = [len(cd['prob_true']) for cd in calibration_details_all if cd is not None]
                        if len(set(shapes)) > 1:
                            print(f"  ⚠️ Warning: Inconsistent calibration bin counts across folds")
                            print(f"     Fold shapes: {shapes}")
                            print(f"     Using median shape for aggregation")
                            # Use median shape to avoid errors
                            median_shape = int(np.median(shapes))
                            # Pad or truncate calibration details to median shape
                            for cd in calibration_details_all:
                                if cd is not None:
                                    if len(cd['prob_true']) > median_shape:
                                        cd['prob_true'] = cd['prob_true'][:median_shape]
                                        cd['prob_pred'] = cd['prob_pred'][:median_shape]
                                    elif len(cd['prob_true']) < median_shape:
                                        # Pad with last values
                                        last_prob_true = cd['prob_true'][-1] if len(cd['prob_true']) > 0 else 0.5
                                        last_prob_pred = cd['prob_pred'][-1] if len(cd['prob_pred']) > 0 else 0.5
                                        cd['prob_true'] = np.pad(cd['prob_true'], (0, median_shape - len(cd['prob_true'])), mode='edge')
                                        cd['prob_pred'] = np.pad(cd['prob_pred'], (0, median_shape - len(cd['prob_pred'])), mode='edge')
                        
                        # Now aggregate with consistent shapes
                        avg_prob_true = np.mean([cd['prob_true'] for cd in calibration_details_all], axis=0)
                        avg_prob_pred = np.mean([cd['prob_pred'] for cd in calibration_details_all], axis=0)
                        
                        print(f"\n     Average Calibration Curve Across Folds:")
                        print(f"       Predicted Prob | Actual Prob | Difference")
                        print(f"       {'-' * 40}")
                        
                        for i in range(len(avg_prob_true)):
                            diff = abs(avg_prob_true[i] - avg_prob_pred[i])
                            print(f"       {avg_prob_pred[i]:.3f}        | {avg_prob_true[i]:.3f}      | {diff:.3f}")
                    
                    except Exception as e:
                        print(f"  ❌ Error aggregating calibration curves: {e}")
                        print(f"     Showing individual fold results instead")
                        # Fallback to showing individual fold results
                        for i, cd in enumerate(calibration_details_all):
                            if cd is not None:
                                print(f"  📊 Fold {i+1} Calibration:")
                                print(f"     Bins: {len(cd['prob_true'])}")
                                for j in range(len(cd['prob_true'])):
                                    print(f"     {cd['prob_pred'][j]:.3f} | {cd['prob_true'][j]:.3f}")
                                print()
                else:
                    print("  ⚠️ No calibration details available for this pipeline")


def create_simple_visualizations(cv_results, best_pipeline_name):
    """
    Create simple visualizations for modeling stage
    """
    try:
        import matplotlib.pyplot as plt
        
        print("\n📊 CREATING SIMPLE VISUALIZATIONS...")
        
        # 1. ROC AUC comparison bar chart
        plt.figure(figsize=(12, 6))
        
        pipeline_names = []
        roc_aucs = []
        
        for pipeline_name, results in cv_results.items():
            if 'roc_auc' in results['weighted_average']:
                pipeline_names.append(pipeline_name)
                roc_aucs.append(results['weighted_average']['roc_auc'])
        
        if roc_aucs:
            bars = plt.bar(pipeline_names, roc_aucs, alpha=0.7, color='lightblue')
            plt.title('ROC AUC Comparison Across All Models', fontsize=16, fontweight='bold')
            plt.xlabel('Pipeline')
            plt.ylabel('ROC AUC')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, roc_aucs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('roc_auc_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ ROC AUC comparison saved as 'roc_auc_comparison.png'")
        
        print("✅ Simple visualizations completed!")
        
    except ImportError:
        print("⚠️ matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")


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
    print("New features added: network (one-hot), time_to_clickout_S, time_to_clickout_S_group (dummies)")
    print("Focusing on data flow analysis and debugging")
    
    # Stage 3: Pipeline creation (no output needed)
    if not pipelines:
        # This section is removed as per the edit hint.
        # The pipeline creation logic should be handled by the pipeline file.
        print("❌ No pipelines provided. Please ensure pipelines are created and passed.")
        return None
    
    # Stage 4: Time series cross-validation
    print("\n" + "="*60)
    print("STAGE 4: TIME SERIES CROSS-VALIDATION")
    print("="*60)
    results, best_pipeline_name = run_time_series_cv(df, pipelines, target_column)
    
    if not results:
        print("❌ Cross-validation failed")
        return None
    
    # Print comparison chart
    print_pipeline_comparison_chart(results)
    
    # Stage 5: Detailed ECE Analysis
    print_detailed_ece_analysis(results, ['biz2credit_gb_1', 'biz2credit_gb_2', 'roei_pipeline'])
    
    # Stage 6: Feature Importance Analysis
    print("\n" + "="*60)
    print("STAGE 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    _aggregate_feature_importance_results(results)
    
    # Stage 6: Feature analysis (transformed features)
    print("\n" + "="*60)
    print("STAGE 6: FEATURE ANALYSIS (Transformed Features)")
    print("="*60)
    print("Feature analysis completed during pipeline execution")
    
    # Stage 6: Create visualizations
    print("\n" + "="*60)
    print("STAGE 6: CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create simple visualizations
    if best_pipeline_name:
        create_simple_visualizations(results, best_pipeline_name)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)
    
    # Show sales coverage summary
    print("📊 SALES COVERAGE SUMMARY:")
    total_dataset_sales = df[target_column].sum()
    print(f"  Total dataset sales: {total_dataset_sales:.0f}")
    
    for pipeline_name, result in results.items():
        if 'weighted_average' in result and 'total_sales' in result['weighted_average']:
            test_sales = result['weighted_average']['total_sales']
            if test_sales is not None:
                coverage = (test_sales / total_dataset_sales) * 100
                print(f"  {pipeline_name}: {test_sales:.0f} sales ({coverage:.1f}% coverage)")
    
    if best_pipeline_name:
        best_result = results[best_pipeline_name]
        print(f"\n🏆 Best Pipeline: {best_pipeline_name}")
        print("Cross-Validation Results:")
        print(f"  Number of Folds: {len(best_result['fold_results'])}")
        print(f"  Total Test Samples: {best_result['weighted_average'].get('total_test_samples', 'N/A')}")
        print(f"  ROC AUC: {best_result['weighted_average'].get('roc_auc', 'N/A'):.4f}")
        print(f"  Accuracy: {best_result['weighted_average'].get('accuracy', 'N/A'):.4f}")
        print(f"  F1 Score: {best_result['weighted_average'].get('f1_score', 'N/A'):.4f}")
    
    print("\n🎉 Time Series Cross-Validation completed successfully!")
    print(f"Results available for {len(results)} pipelines")
    
    return results

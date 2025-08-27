"""
Biz2Credit Framework
Contains only essential functions for time series CV and metrics calculation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from ds_modeling.ml_framework.evaluation_metrics import BINARY_CLASSIFICATION_BOTH_PROBA_METRICS
from ece_utils import expected_calibration_error
import warnings

# Filter out common sklearn warnings
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# ============================================================================
def run_pre_analysis(df):
    """Run basic data analysis and print summary"""
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
    
    return null_df

# ============================================================================

def analyze_enrichment_features(df_company, enrichment_features: list[str] = None):
    """Analyze enrichment features availability and quality with detailed insights"""
    if enrichment_features is None:
        return
    
    print(f"\n🔍 Available enrichment features after company-specific filtering:")
    
    for feature in enrichment_features:
        if feature in df_company.columns:
            null_count = df_company[feature].isnull().sum()
            total_count = len(df_company)
            null_rate = null_count / total_count * 100
            print(f"   ✅ {feature}: {null_count:,}/{total_count:,} nulls ({null_rate:.1f}%)")
            
            # Analyze feature type and provide insights
            if df_company[feature].dtype == 'object':
                # Categorical feature analysis
                print(f"      📊 Categorical Analysis - Top 10 Categories:")
                print(f"      {'Category':<30} {'Leads':<8} {'L2S':<8}")
                print(f"      {'-' * 30} {'-' * 8} {'-' * 8}")
                
                # Group by category and count leads + sum sales
                cat_analysis = df_company.groupby(feature).agg({
                    'sales_count': ['count', 'sum']
                }).round(2)
                cat_analysis.columns = ['leads', 'sales_sum']
                # Calculate L2S ratio: sales_sum / leads
                cat_analysis['l2s'] = (cat_analysis['sales_sum'] / cat_analysis['leads']).round(4)
                cat_analysis = cat_analysis.sort_values('leads', ascending=False).head(10)
                
                for cat, row in cat_analysis.iterrows():
                    cat_str = str(cat)[:28] + ".." if len(str(cat)) > 30 else str(cat)
                    print(f"      {cat_str:<30} {row['leads']:<8.0f} {row['l2s']:<8.4f}")
                    
            else:
                # Numerical feature analysis
                print(f"      📊 Numerical Analysis - Correlation with Sales:")
                
                # Calculate correlation with sales_count
                correlation = df_company[feature].corr(df_company['sales_count'])
                
                # Group by feature value and analyze sales trend
                if df_company[feature].nunique() <= 20:  # If few unique values, show grouping
                    num_analysis = df_company.groupby(df_company[feature]).agg({
                        'sales_count': ['count', 'sum']
                    }).round(2)
                    num_analysis.columns = ['leads', 'sales_sum']
                    # Calculate L2S ratio: sales_sum / leads
                    num_analysis['l2s'] = (num_analysis['sales_sum'] / num_analysis['leads']).round(4)
                    num_analysis = num_analysis.sort_values('leads', ascending=False).head(5)
                    
                    print(f"      {'Value':<15} {'Leads':<8} {'L2S':<8}")
                    print(f"      {'-' * 15} {'-' * 8} {'-' * 8}")
                    for val, row in num_analysis.iterrows():
                        val_str = str(val)[:13] + ".." if len(str(val)) > 15 else str(val)
                        print(f"      {val_str:<15} {row['leads']:<8.0f} {row['l2s']:<8.4f}")
                
                # Show correlation and trend
                trend = "↗️ ASC" if correlation > 0.1 else "↘️ DESC" if correlation < -0.1 else "➡️ NONE"
                print(f"      📈 Correlation with sales: {correlation:.3f} ({trend})")
                
        else:
            print(f"   ❌ {feature}: Column not found")
        
        print()  # Add spacing between features

# ============================================================================

def _calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray = None,
    y_pred_proba: np.ndarray = None
) -> dict[str, float]:
    """Calculate essential probability-based classification metrics with ECE."""
    
    metrics = {}
    
    # Probability-based metrics only (if probabilities available)
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            # Binary classification - essential KPIs only
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['log_loss'] = BINARY_CLASSIFICATION_BOTH_PROBA_METRICS['log_loss'](y_true, y_pred_proba)
            metrics['r2_proba'] = BINARY_CLASSIFICATION_BOTH_PROBA_METRICS['r2_score'](y_true, y_pred_proba[:, 1])
            metrics['rmse_proba'] = BINARY_CLASSIFICATION_BOTH_PROBA_METRICS['root_mean_squared_error'](y_true, y_pred_proba[:, 1])
            
            # Calculate ECE using team's function (keeping this as requested)
            ece = expected_calibration_error(y_true, y_pred_proba[:, 1], n_bins=5)
            metrics['ece'] = ece
            
            # TEMPORARY DEBUG: Print ECE bins for first fold only
            if len(y_true) > 100:  # Only for test sets (not training)
                print(f"🔍 ECE DEBUG - Bins: {ece:.4f} (n_bins=5)")
                # Show actual bin details for debugging
                df_debug = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba[:, 1]})
                
                # Debug probability distribution
                unique_probs = df_debug['y_pred_proba'].nunique()
                print(f"🔍 Probability Distribution: {len(df_debug)} samples, {unique_probs} unique values")
                print(f"🔍 Min: {df_debug['y_pred_proba'].min():.6f}, Max: {df_debug['y_pred_proba'].max():.6f}")
                
                try:
                    df_debug['y_pred_bin'] = pd.qcut(df_debug['y_pred_proba'], q=5, labels=False, duplicates='drop')
                    bin_details = df_debug.groupby('y_pred_bin').agg({
                        'y_pred_proba': ['mean', 'count'],
                        'y_true': 'mean'
                    }).round(4)
                    print(f"🔍 ECE Bin Details:\n{bin_details}")
                except Exception as e:
                    print(f"🔍 Quantile binning failed: {e}")
                    # Fallback to manual binning
                    df_debug['y_pred_bin'] = pd.cut(df_debug['y_pred_proba'], bins=5, labels=False, include_lowest=True)
                    bin_details = df_debug.groupby('y_pred_bin').agg({
                        'y_pred_proba': ['mean', 'count'],
                        'y_true': 'mean'
                    }).round(4)
                    print(f"🔍 Fallback Bin Details:\n{bin_details}")
            
        else:
            # Multiclass classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            metrics['log_loss'] = BINARY_CLASSIFICATION_BOTH_PROBA_METRICS['log_loss'](y_true, y_pred_proba)
    
    return metrics

# ============================================================================

def _generate_time_series_folds(data: pd.DataFrame, date_column: str, train_days: int = 90, test_days: int = 7, step_days: int = 7) -> list[tuple[int, int, int]]:
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
    
    # Ensure data is sorted by date
    data_sorted = data.sort_values(date_column).reset_index(drop=True)
    
    # Get unique dates and sort them
    unique_dates = pd.to_datetime(data_sorted[date_column]).dt.date.unique()
    unique_dates = sorted(unique_dates)
    
    if len(unique_dates) < train_days + test_days:
        return folds
    
    # Calculate step size: Use smaller steps for better coverage
    total_days = len(unique_dates)
    available_days = total_days - train_days - test_days
    
    # Use non-overlapping folds for realistic coverage
    step_size = step_days  # Move forward by exactly step_days (non-overlapping)
    
    print(f"📅 Generating {available_days // step_size} time series folds ({train_days} days train, {test_days} days test, {step_days} days step)")
    
    # Start with training period
    train_start_idx = 0
    train_end_idx = train_days
    
    while train_end_idx + test_days <= len(unique_dates):
        test_end_idx = train_end_idx + test_days
        
        # Find the actual row indices for these date boundaries
        train_start_date = unique_dates[train_start_idx]
        train_end_date = unique_dates[train_start_idx + train_days - 1]
        test_start_date = unique_dates[train_start_idx + train_days]
        test_end_date = unique_dates[train_start_idx + train_days + test_days - 1]
        
        # Convert dates back to row indices
        train_start = data_sorted[data_sorted[date_column].dt.date >= train_start_date].index[0]
        train_end = data_sorted[data_sorted[date_column].dt.date <= train_end_date].index[-1] + 1
        test_start = data_sorted[data_sorted[date_column].dt.date >= test_start_date].index[0]
        test_end = data_sorted[data_sorted[date_column].dt.date <= test_end_date].index[-1] + 1
        
        # Ensure we have enough data
        if train_end - train_start >= 100 and test_end - test_start >= 10:  # Minimum sample requirements
            folds.append((train_start, train_end, test_start, test_end))
        
        # Move forward by step_days for non-overlapping folds
        train_start_idx += step_size
        train_end_idx += step_size
    
    return folds

# ============================================================================

def _calculate_weighted_average(metrics_list: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    """Calculate weighted average of metrics across folds."""
    if not metrics_list or not weights:
        return {}
    
    weighted_metrics = {}
    total_weight = sum(weights)
    
    # Get all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Calculate weighted average for each metric
    for metric in all_metrics:
        weighted_sum = 0
        valid_weights = 0
        
        for metrics, weight in zip(metrics_list, weights):
            if metric in metrics and metrics[metric] is not None:
                weighted_sum += metrics[metric] * weight
                valid_weights += weight
        
        if valid_weights > 0:
            weighted_metrics[metric] = weighted_sum / valid_weights
        else:
            weighted_metrics[metric] = None
    
    return weighted_metrics

# ============================================================================

def time_series_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    date_column: str,
    train_days: int = 90,
    test_days: int = 7,
    step_days: int = 7,
    data_sorted: pd.DataFrame = None
) -> dict[str, any]:
    """Perform time series cross-validation."""
    
    # Generate folds
    folds = _generate_time_series_folds(X, date_column, train_days, test_days, step_days)
    
    if not folds:
        print("❌ No valid folds generated. Check your data and date column.")
        return {}
    
    fold_results = []
    all_metrics = []
    weights = []
    
    print(f"🔄 Running {len(folds)} time series folds...")
    
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{len(folds)}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")
        
        # Split data for this fold
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"    ❌ Skipping fold {fold_idx + 1} - empty train or test set")
            continue
        
        # Clone pipeline for this fold
        fold_pipeline = clone(pipeline)
        
        # Fit and predict
        fold_pipeline.fit(X_train, y_train)
       # y_pred = fold_pipeline.predict(X_test)
        y_pred_proba = fold_pipeline.predict_proba(X_test)
        
        # Calculate test metrics
        test_metrics = _calculate_classification_metrics(y_test, None, y_pred_proba)
        
        # Calculate train metrics (on training data)
        y_train_pred = fold_pipeline.predict(X_train)
        y_train_pred_proba = fold_pipeline.predict_proba(X_train)
        train_metrics = _calculate_classification_metrics(y_train, y_train_pred, y_train_pred_proba)
        
        # Store results
        weight = len(y_test)
        fold_results.append({
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'weight': weight,
            'fold_idx': fold_idx
        })
        
        all_metrics.append(test_metrics)
        weights.append(weight)
        
        # Print both train and test metrics
        test_roc_auc = test_metrics.get('roc_auc')
        test_ece = test_metrics.get('ece')
        train_roc_auc = train_metrics.get('roc_auc')
        train_ece = train_metrics.get('ece')
        
        test_roc_auc_str = f"{test_roc_auc:.4f}" if test_roc_auc is not None else "N/A"
        test_ece_str = f"{test_ece:.4f}" if test_ece is not None else "N/A"
        train_roc_auc_str = f"{train_roc_auc:.4f}" if train_roc_auc is not None else "N/A"
        train_ece_str = f"{train_ece:.4f}" if train_ece is not None else "N/A"
        
        # Get dates for this fold
        train_start, train_end, test_start, test_end = folds[fold_idx]
        train_dates = data_sorted.iloc[train_start:train_end][date_column]
        test_dates = data_sorted.iloc[test_start:test_end][date_column]
        
        train_start_date = train_dates.min().strftime('%Y-%m-%d')
        train_end_date = train_dates.max().strftime('%Y-%m-%d')
        test_start_date = test_dates.min().strftime('%Y-%m-%d')
        test_end_date = test_dates.max().strftime('%Y-%m-%d')
        
        print(f"✅ Fold {fold_idx + 1}:")
        print(f"   Train: {train_start_date} to {train_end_date} (rows {train_start}-{train_end}, size: {train_end - train_start})")
        print(f"   Test:  {test_start_date} to {test_end_date} (rows {test_start}-{test_end}, size: {test_end - test_start})")
        print(f"   Metrics: Train ROC AUC: {train_roc_auc_str}, Test ROC AUC: {test_roc_auc_str}, Test RMSE: {test_metrics.get('rmse_proba', 'N/A'):.4f}")
    
    if not all_metrics:
        print("❌ No successful folds completed")
        return {}
    
    # Calculate weighted averages
    weighted_average = _calculate_weighted_average(all_metrics, weights)
    
    return {
        'fold_results': fold_results,
        'weighted_average': weighted_average,
        'total_folds': len(folds)
    }

# ============================================================================

def print_simple_results_summary(results: dict[str, any], pipeline_name: str):
    """Print simple results summary for a pipeline."""
    if not results or 'weighted_average' not in results:
        print(f"❌ No results for {pipeline_name}")
        return
    
    metrics = results['weighted_average']
    total_folds = results.get('total_folds', 0)
    
    print(f"\n📊 {pipeline_name.upper()} - SUMMARY:")
    print(f"   Folds: {total_folds}")
    print(f"   ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    print(f"   Log Loss: {metrics.get('log_loss', 'N/A'):.4f}")
    print(f"   R²: {metrics.get('r2_proba', 'N/A'):.4f}")
    print(f"   RMSE: {metrics.get('rmse_proba', 'N/A'):.4f}")
    print(f"   ECE: {metrics.get('ece', 'N/A'):.4f}")
    
    # Show 1-2 fold details with ECE bins
    if 'fold_results' in results and len(results['fold_results']) > 0:
        print(f"\n   📈 Sample Fold Details:")
        for i, fold in enumerate(results['fold_results'][:2]):  # Show first 2 folds
            test_metrics = fold['test_metrics']
            train_metrics = fold['train_metrics']
            weight = fold['weight']
            print(f"     Fold {fold['fold_idx'] + 1} ({weight} samples):")
            print(f"       Train - ROC AUC: {train_metrics.get('roc_auc', 'N/A'):.4f}, RMSE: {train_metrics.get('rmse_proba', 'N/A'):.4f}")
            print(f"       Test  - ROC AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}, RMSE: {test_metrics.get('rmse_proba', 'N/A'):.4f}")

# ============================================================================

def run_time_series_cv(
    pipelines: dict[str, any],
    X: pd.DataFrame,
    y: pd.Series,
    date_column: str = 'clickout_date_prt',
    enrichment_features: list[str] = None,
    train_days: int = 90,
    test_days: int = 7,
    step_days: int = 7
) -> dict[str, any]:
    """Run time series cross-validation for all pipelines."""
    
    print("🚀 STARTING TIME SERIES CROSS-VALIDATION")
    print("=" * 60)
    
    # Remove rows where all enrichment features are null (if provided)
    if enrichment_features:
        available_features = [f for f in enrichment_features if f in X.columns]
        if available_features:
            # Check if all enrichment features are null for each row
            all_null_mask = X[available_features].isnull().all(axis=1)
            initial_rows = len(X)
            X = X[~all_null_mask]
            y = y[~all_null_mask]
            removed_rows = initial_rows - len(X)
            
            if removed_rows > 0:
                print(f"📊 Removed {removed_rows:,} rows where all enrichment features were null")
                print(f"📊 Remaining data: {len(X):,} rows")
    
    all_results = {}
    
    for pipeline_name, pipeline in pipelines.items():
        print(f"\n🎯 Testing pipeline: {pipeline_name}")
        print("-" * 50)
        
        results = time_series_cross_validation(pipeline, X, y, date_column, train_days, test_days, step_days, data_sorted=X.sort_values(date_column).reset_index(drop=True))
        all_results[pipeline_name] = results
        
        # Print simple summary
        print_simple_results_summary(results, pipeline_name)
        
        # 🌳 Analyze feature importance ONLY for RF model
        if pipeline_name == 'optimized_rf':
            analyze_feature_importance(pipeline, X, pipeline_name)
    
    return all_results

# ============================================================================

def analyze_feature_importance(pipeline, X: pd.DataFrame, pipeline_name: str):
    """Analyze feature importance for Random Forest models."""
    try:
        # Get the final estimator (last step in pipeline)
        final_estimator = pipeline.steps[-1][1]
        
        # Check if it's a Random Forest
        if hasattr(final_estimator, 'feature_importances_'):
            # Get feature names from the pipeline
            feature_names = []
            for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the final estimator
                if hasattr(step_transformer, 'get_feature_names_out'):
                    feature_names.extend(step_transformer.get_feature_names_out())
                elif hasattr(step_transformer, 'get_feature_names'):
                    feature_names.extend(step_transformer.get_feature_names())
            
            # If we couldn't get feature names, use generic ones
            if not feature_names:
                feature_names = [f'feature_{i}' for i in range(len(final_estimator.feature_importances_))]
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': final_estimator.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🌳 FEATURE IMPORTANCE ANALYSIS - {pipeline_name.upper()}")
            print("=" * 60)
            print("Top 15 Most Important Features:")
            print("-" * 60)
            
            for idx, row in importance_df.head(15).iterrows():
                print(f"{idx+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
            
            print("-" * 60)
            print(f"Total features analyzed: {len(importance_df)}")
            
            return importance_df
            
    except Exception as e:
        print(f"⚠️ Could not analyze feature importance for {pipeline_name}: {e}")
        return None

# ============================================================================

def print_final_comparison(results: dict[str, any]):
    """Print final comparison of all pipelines."""
    print("\n" + "=" * 80)
    print("🏁 FINAL PIPELINE COMPARISON")
    print("=" * 80)
    
    # Sort pipelines by RMSE (lower is better, so reverse=False)
    sorted_pipelines = sorted(
        [(name, results[name]) for name in results.keys() if results[name] and 'weighted_average' in results[name]],
        key=lambda x: x[1]['weighted_average'].get('rmse_proba', float('inf')),
        reverse=False
    )
    
    if not sorted_pipelines:
        print("❌ No valid results to compare")
        return
    
    # Print header
    print(f"{'Pipeline':<25} {'ROC AUC':<10} {'Log Loss':<10} {'R²':<8} {'RMSE':<8} {'ECE':<8}")
    print("-" * 80)
    
    # Print results with medals
    medals = ['1', '2', '3', '4','5']
    
    for idx, (pipeline_name, result) in enumerate(sorted_pipelines):
        metrics = result['weighted_average']
        
        if idx < len(medals):
            medal = medals[idx]
        else:
            medal = f"{idx+1}️⃣"
        
        roc_auc = metrics.get('roc_auc')
        log_loss = metrics.get('log_loss')
        r2 = metrics.get('r2_proba')
        rmse = metrics.get('rmse_proba')
        ece = metrics.get('ece')
        
        # Format metrics safely
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        log_loss_str = f"{log_loss:.4f}" if log_loss is not None else "N/A"
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"
        
        print(f"{medal} {pipeline_name:<22} {roc_auc_str:<10} {log_loss_str:<10} {r2_str:<8} {rmse_str:<8} {ece_str:<8}")
    
    print("-" * 80)
    print("📊 Lower values are better for Log Loss, RMSE, and ECE")
    print("🎯 Higher values are better for ROC AUC and R²")
    print("🏆 Models are ranked by RMSE (lower = better)")

# ============================================================================

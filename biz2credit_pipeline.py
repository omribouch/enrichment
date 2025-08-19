"""
Biz2Credit Pipeline - Main execution script
"""

import pandas as pd
import numpy as np
# Import standard sklearn models instead of problematic ds_modeling wrappers
try:
    from ds_modeling.ml_framework.pipeline import make_pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"⚠️ ds_modeling pipeline not available: {e}")

try:
    from ds_modeling.ml_framework.prdictors_wrappers.classifiers.linear import SkLogisticRegression
    LOGISTIC_AVAILABLE = True
except ImportError as e:
    LOGISTIC_AVAILABLE = False
    print(f"⚠️ SkLogisticRegression not available: {e}")

# Import standard XGBoost instead of wrapper
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    print("✅ XGBoost available - using standard XGBClassifier")
except ImportError as e:
    XGB_AVAILABLE = False
    print(f"⚠️ XGBoost not available: {e}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Fallback models
from sklearn.linear_model import LogisticRegression # Added for fallback
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve

from biz2credit_transformers import Biz2CreditPrep1, Biz2CreditPrep1_2, Biz2CreditPrep_keep_additional_features, Biz2CreditImputer, Biz2CreditCategoricalEncoder, FeaturesAdder, build_pip1e, FinalNumericFilter, FeatureSelector
from biz2credit_data_handler import Biz2CreditDataHandler
from biz2credit_framework import run_biz2credit_analysis

# ============================================================================
# CONFIGURATION: Toggle between best params vs hyperparameter tuning
# ============================================================================

# 🎯 SET THIS TO FALSE if you want to run hyperparameter tuning again
USE_BEST_PARAMS = True

# 🏆 BEST PARAMETERS FROM YOUR SUCCESSFUL TUNING (ROC AUC: 0.9220, ECE: 0.1288)
BEST_GB_PARAMS = {
    'n_estimators': 200,        # Best from tuning
    'learning_rate': 0.1,       # Best from tuning  
    'max_depth': 4,             # Best from tuning
    'min_samples_split': 5,     # Best from tuning
    'min_samples_leaf': 2,      # Best from tuning
    'subsample': 0.9,           # Best from tuning
    'max_features': 'sqrt',      # Best from tuning
    'random_state': 42
}

# ============================================================================
# PIPELINE TESTING FUNCTION
# ============================================================================

def test_pipeline(pipeline, name):
    """Test if a pipeline can be created and has the right structure"""
    try:
        if hasattr(pipeline, 'named_steps'):
            steps = list(pipeline.named_steps.keys())
            print(f"✅ {name}: {len(steps)} steps - {steps}")
            return True
        elif hasattr(pipeline, 'estimator'):
            print(f"✅ {name}: GridSearchCV with {type(pipeline.estimator).__name__}")
            return True
        else:
            print(f"❌ {name}: Unknown pipeline structure")
            return False
    except Exception as e:
        print(f"❌ {name}: Error - {e}")
        return False

# ============================================================================
# HYPERPARAMETER TUNING FUNCTIONS
# ============================================================================

def create_light_gradient_boosting():
    """Create a lighter Gradient Boosting with hyperparameter tuning (optimized for speed)"""
    base_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=False),
        Biz2CreditPrep1_2(),
        Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=['p_sale']),
        Biz2CreditImputer(),
        Biz2CreditCategoricalEncoder(),
        FinalNumericFilter(),
        GradientBoostingClassifier(random_state=42)
    )

    # Define hyperparameter grid for Gradient Boosting (MINIMAL for speed)
    param_distributions = {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.15],
        'gradientboostingclassifier__max_depth': [3, 4],
        'gradientboostingclassifier__min_samples_split': [2, 5],
        'gradientboostingclassifier__subsample': [0.8, 0.9],
        'gradientboostingclassifier__max_features': ['sqrt', None]
    }

    # Create RandomizedSearchCV pipeline (MUCH FASTER)
    tuned_pipeline = RandomizedSearchCV(
        base_pipeline, 
        param_distributions, 
        n_iter=2,  # Only test 2 random combinations
        cv=2,      # Reduce CV folds
        scoring='roc_auc', # ECE
        n_jobs=1,  # Single job to avoid parallel issues
        verbose=0,
        random_state=42
    )
    
    return tuned_pipeline

# ============================================================================
# MAIN PIPELINE CREATION
# ============================================================================

# Define additional features list
additional_features_list = ['network', 'time_to_clickout_s_group']  # Removed time_to_clickout_s (too granular)

def main():
    """Main execution function"""
    
    print("🚀 Starting Biz2Credit Pipeline Analysis")
    print("=" * 60)
    
    # Load real Biz2Credit data using data handler
    print("📊 Loading Biz2Credit data using data handler...")
    
    try:
        # Create data handler and load data
        data_handler = Biz2CreditDataHandler()
        df_biz2credit = data_handler.load_data()
        
        # Check if data is empty
        if df_biz2credit is None or df_biz2credit.empty:
            print("❌ No data loaded. Exiting.")
            return
        
        print(f"✅ Data loaded successfully: {df_biz2credit.shape}")
        
        # Show feature information from the raw data
        data_handler.show_feature_info()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Exiting due to data loading failure.")
        return
    
    print("\n🔧 Creating pipeline instances...")
    
    # Create pipeline instances using available models
    pipelines = {}
    
    # Keep only the ESSENTIAL models for speed
    print("🚀 Creating ESSENTIAL pipelines only (for speed)...")
    
    # Gradient Boosting with additional features (your main model) - TOGGLE MODE
    if USE_BEST_PARAMS:
        # 🚀 FAST MODE: Use best parameters (instant results)
        pipelines['biz2credit_gb_2'] = make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=additional_features_list),  # Step 3: WITH additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            GradientBoostingClassifier(**BEST_GB_PARAMS)    # Use best parameters
        )
        print("✅ Created biz2credit_gb_2 (FAST MODE - Best parameters from tuning)")
        
    else:
        # 🔍 TUNING MODE: Run hyperparameter optimization
        base_gb_pipeline = make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=additional_features_list),  # Step 3: WITH additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            GradientBoostingClassifier(random_state=42)     # Base model for tuning
        )
        
        # 🎯 ENHANCED hyperparameter tuning for your main model
        param_distributions = {
            # Core performance parameters
            'gradientboostingclassifier__n_estimators': [100, 150, 200, 300],
            'gradientboostingclassifier__learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],
            'gradientboostingclassifier__max_depth': [3, 4, 5, 6],
            
            # Regularization to prevent overfitting
            'gradientboostingclassifier__min_samples_split': [2, 5, 10],
            'gradientboostingclassifier__min_samples_leaf': [1, 2, 4],
            'gradientboostingclassifier__subsample': [0.8, 0.85, 0.9, 0.95],
            'gradientboostingclassifier__max_features': ['sqrt', 'log2', None]
        }
        
        # Create tuned version of your main model
        pipelines['biz2credit_gb_2'] = RandomizedSearchCV(
            base_gb_pipeline,
            param_distributions,
            n_iter=12,        # Good exploration without being too slow
            cv=3,             # Reliable 3-fold CV
            scoring='roc_auc',
            n_jobs=1,         # Single job to avoid parallel issues
            verbose=0,        # Reduce verbose output
            random_state=42,
            refit=True
        )
        
        print("✅ Created biz2credit_gb_2 (TUNING MODE - 12 iterations, 3-fold CV)")
    
    # ROEI pipeline - simple features (baseline) - NO TUNING
    pipelines['roei_pipeline'] = make_pipeline(
        FeaturesAdder(),
        Biz2CreditImputer(),                            # Add imputation
        Biz2CreditCategoricalEncoder(),                 # Add categorical encoding
        FinalNumericFilter(),                           # Ensure numeric only
        GradientBoostingClassifier(n_estimators=100, random_state=42)  # sklearn
    )
    print("✅ Created roei_pipeline (baseline - NO TUNING)")
    
    # ============================================================================
    # HYPERPARAMETER TUNED PIPELINES (OPTIMIZED VERSION)
    # ============================================================================
    
    print("\n🚀 Creating OPTIMIZED HYPERPARAMETER TUNED pipelines...")
    
    # Tuned Random Forest - SET GOOD PARAMS AS DEFAULT
    try:
        # Use the parameters that already beat ROEI (ECE: 0.1115)
        pipelines['optimized_rf'] = make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),
            Biz2CreditPrep1_2(),
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=['p_sale']),
            Biz2CreditImputer(),
            Biz2CreditCategoricalEncoder(),
            FinalNumericFilter(),
            RandomForestClassifier(
                n_estimators=200,      # Good param from tuning
                max_depth=10,          # Good param from tuning
                random_state=42
            )
        )
        print("✅ Created optimized_rf with GOOD PARAMS AS DEFAULT (already beats ROEI!)")
    except Exception as e:
        print(f"⚠️ Failed to create optimized Random Forest: {e}")
    
    # Tuned Gradient Boosting - REDUCED TUNING (avoid overfitting)
    try:
        pipelines['light_tuned_gb'] = create_light_gradient_boosting()
        print("✅ Created light_tuned_gb with LIGHT hyperparameter optimization")
    except Exception as e:
        print(f"⚠️ Failed to create light tuned Gradient Boosting: {e}")
    
    print(f"\n🎯 Total pipelines created: {len(pipelines)}")
    print("📊 Pipeline types:")
    print("  - biz2credit_gb_2 (your main model - ENHANCED TUNING)")
    print("  - ROEI pipeline (baseline - NO TUNING)")
    print("  - optimized_rf (GOOD PARAMS AS DEFAULT - already beats ROEI!)")
    print("  - light_tuned_gb (LIGHT tuning - avoid overfitting)")
    
    # Test all pipelines to ensure they work
    print("\n🧪 Testing pipeline structures...")
    working_pipelines = {}
    for name, pipeline in pipelines.items():
        if test_pipeline(pipeline, name):
            working_pipelines[name] = pipeline
        else:
            print(f"⚠️ Removing {name} from analysis due to errors")
    
    print(f"\n✅ Working pipelines: {len(working_pipelines)}/{len(pipelines)}")
    
    print("\n🔧 HYPERPARAMETER TUNING STRATEGY:")
    if USE_BEST_PARAMS:
        print("  - biz2credit_gb_2: FAST MODE (Best parameters from tuning)")
    else:
        print("  - biz2credit_gb_2: TUNING MODE (12 iterations, 3-fold CV)")
    print("  - Random Forest: GOOD PARAMS AS DEFAULT (already beats ROEI!)")
    print("  - Gradient Boosting: LIGHT tuning (avoid overfitting)")
    print("  - Goal: Beat ROEI on ECE & RMSE with smart tuning")
    print("  - SPEED: Optimized for your needs!")
    
    # ============================================================================
    # PERFORMANCE COMPARISON FUNCTION
    # ============================================================================
    
    def compare_tuned_vs_untuned(results):
        """Compare tuned vs untuned pipeline performance focusing on ECE and RMSE"""
        print("\n" + "="*80)
        print("🏆 CALIBRATION & PREDICTION ACCURACY COMPARISON (ECE & RMSE)")
        print("="*80)
        print("💡 Why ECE & RMSE matter more than ROC AUC:")
        print("  - ECE: How well predicted probabilities match actual outcomes")
        print("  - RMSE: How accurate the probability predictions are")
        print("  - These directly impact business decisions and risk assessment")
        print("  - ROC AUC only measures ranking, not prediction accuracy")
        print("="*80)
        
        # Debug: Print the structure of results to understand the format
        print("\n🔍 DEBUG: Results structure:")
        for name, result in results.items():
            print(f"  {name}: {type(result)}")
            if hasattr(result, 'keys'):
                print(f"    Keys: {list(result.keys())}")
            else:
                print(f"    Content: {result}")
        
        # Group results by type
        basic_pipelines = {}
        tuned_pipelines = {}
        roei_pipeline = {}
        
        for name, result in results.items():
            if 'tuned_' in name or 'optimized_' in name or 'light_' in name:
                tuned_pipelines[name] = result
            elif 'roei' in name:
                roei_pipeline[name] = result
            else:
                basic_pipelines[name] = result
        
        # Show ROEI performance (baseline)
        if roei_pipeline:
            print("\n🎯 ROEI BASELINE PERFORMANCE (Target to beat):")
            for name, result in roei_pipeline.items():
                # Extract metrics from weighted_average
                weighted_avg = result.get('weighted_average', {})
                ece = weighted_avg.get('ece', 'N/A')
                rmse = weighted_avg.get('rmse_proba', 'N/A')
                if isinstance(ece, (int, float)) and isinstance(rmse, (int, float)):
                    print(f"  {name}: ECE = {ece:.4f}, RMSE = {rmse:.4f}")
                else:
                    print(f"  {name}: ECE = {ece}, RMSE = {rmse}")
        
        # Show best basic pipeline
        if basic_pipelines:
            # Find best by ECE (lower is better)
            best_basic_ece = min(basic_pipelines.items(), 
                               key=lambda x: x[1].get('weighted_average', {}).get('ece', float('inf')) if isinstance(x[1].get('weighted_average', {}).get('ece'), (int, float)) else float('inf'))
            # Find best by RMSE (lower is better)
            best_basic_rmse = min(basic_pipelines.items(), 
                                key=lambda x: x[1].get('weighted_average', {}).get('rmse_proba', float('inf')) if isinstance(x[1].get('weighted_average', {}).get('rmse_proba'), (int, float)) else float('inf'))
            
            print(f"\n📊 BEST BASIC PIPELINES:")
            ece = best_basic_ece[1].get('weighted_average', {}).get('ece', 'N/A')
            rmse = best_basic_ece[1].get('weighted_average', {}).get('ece', 'N/A')
            if isinstance(ece, (int, float)):
                print(f"  Best ECE: {best_basic_ece[0]} (ECE = {ece:.4f})")
            else:
                print(f"  Best ECE: {best_basic_ece[0]} (ECE = {ece})")
                
            rmse = best_basic_rmse[1].get('weighted_average', {}).get('rmse_proba', 'N/A')
            if isinstance(rmse, (int, float)):
                print(f"  Best RMSE: {best_basic_rmse[0]} (RMSE = {rmse:.4f})")
            else:
                print(f"  Best RMSE: {best_basic_rmse[0]} (RMSE = {rmse})")
        
        # Show tuned pipeline performance
        if tuned_pipelines:
            print(f"\n🚀 HYPERPARAMETER TUNED PIPELINES:")
            for name, result in tuned_pipelines.items():
                weighted_avg = result.get('weighted_average', {})
                ece = weighted_avg.get('ece', 0)
                rmse = weighted_avg.get('rmse_proba', 0)
                
                if isinstance(ece, (int, float)) and isinstance(rmse, (int, float)):
                    print(f"  {name}: ECE = {ece:.4f}, RMSE = {rmse:.4f}")
                    
                    # Compare with ROEI on ECE and RMSE
                    if roei_pipeline:
                        roei_weighted_avg = list(roei_pipeline.values())[0].get('weighted_average', {})
                        roei_ece = roei_weighted_avg.get('ece', 0)
                        roei_rmse = roei_weighted_avg.get('rmse_proba', 0)
                        
                        if isinstance(roei_ece, (int, float)) and isinstance(roei_rmse, (int, float)):
                            ece_improvement = roei_ece - ece  # Lower ECE is better
                            rmse_improvement = roei_rmse - rmse  # Lower RMSE is better
                            
                            print(f"    📊 vs ROEI:")
                            if ece_improvement > 0:
                                print(f"      🎉 ECE: BEATS ROEI by +{ece_improvement:.4f}!")
                            else:
                                print(f"      📉 ECE: Loses to ROEI by {abs(ece_improvement):.4f}")
                                
                            if rmse_improvement > 0:
                                print(f"      🎉 RMSE: BEATS ROEI by +{rmse_improvement:.4f}!")
                            else:
                                print(f"      📉 RMSE: Loses to ROEI by {abs(rmse_improvement):.4f}")
                        else:
                            print(f"    ⚠️ Cannot compare with ROEI (invalid values)")
                else:
                    print(f"  {name}: ECE = {ece}, RMSE = {rmse}")
        
        # Find the overall winner by ECE (most important for business)
        all_results = {**basic_pipelines, **tuned_pipelines, **roei_pipeline}
        if all_results:
            # Filter out results with valid ECE values
            valid_results = {k: v for k, v in all_results.items() 
                           if isinstance(v.get('weighted_average', {}).get('ece'), (int, float))}
            if valid_results:
                winner = min(valid_results.items(), key=lambda x: x[1].get('weighted_average', {}).get('ece', float('inf')))
                print(f"\n🏆 OVERALL WINNER (by ECE): {winner[0]}")
                print(f"  Best ECE: {winner[1].get('weighted_average', {}).get('ece'):.4f}")
                print(f"  💡 Lower ECE = Better probability calibration = Better business decisions!")
            else:
                print(f"\n⚠️ No valid ECE results to determine winner")
        
        print("="*80)
    
    print(f"\n✅ Created {len(pipelines)} pipeline instances:")
    for name, pipeline in pipelines.items():
        print(f"  - {name}: {type(pipeline).__name__}")
        if hasattr(pipeline, 'named_steps'):
            print(f"    Steps: {list(pipeline.named_steps.keys())}")
        else:
            print(f"    Steps: {pipeline}")
    
    # Run the complete analysis
    print("\n🚀 Running complete Biz2Credit analysis...")
    
    try:
        all_results = run_biz2credit_analysis(
            df=df_biz2credit,
            pipelines=working_pipelines,  # Use only working pipelines
            target_column='sales_count'
        )
        
        if all_results and 'error' not in all_results:
            # Run the hyperparameter tuning comparison
            compare_tuned_vs_untuned(all_results)
            print(f"\n✅ Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed - check error details above")
            
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

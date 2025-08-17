"""
Biz2Credit Pipeline - Main execution script
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_squared_error, r2_score
from sklearn.calibration import calibration_curve

from biz2credit_transformers import Biz2CreditPrep1, Biz2CreditPrep1_2, Biz2CreditPrep_keep_additional_features, Biz2CreditImputer, Biz2CreditCategoricalEncoder, FeaturesAdder, build_pip1e, FinalNumericFilter
from biz2credit_data_handler import Biz2CreditDataHandler
from biz2credit_framework import run_biz2credit_analysis

# Define additional features list
additional_features_list = ['network', 'time_to_clickout_s_group', 'time_to_clickout_s']

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
    
    # Create pipeline instances
    pipelines = {
        # Main models with all features
        'biz2credit_rf_1': make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=additional_features_list),  # Step 3: Keep additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        
        'biz2credit_gb_1': make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=False, additional_features_list=additional_features_list),  # Step 3: NO additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        ),
        
        'biz2credit_gb_2': make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=additional_features_list),  # Step 3: WITH additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        ),
        
        'biz2credit_lr_1': make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=False),           # Step 1: Create p_sale, keep all columns
            Biz2CreditPrep1_2(),                            # Step 2: Enrichment preprocessing
            Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=additional_features_list),  # Step 3: Keep additional features
            Biz2CreditImputer(),                            # Step 4: Imputation
            Biz2CreditCategoricalEncoder(),                 # Step 5: Categorical encoding (business quality, age, revenue)
            FinalNumericFilter(),                           # Step 6: Final filtering (numeric only)
            LogisticRegression(random_state=42, max_iter=1000)
        ),
        
        # Baseline model - only p_sale
        'old_model_pipeline': make_pipeline(
            Biz2CreditPrep1(keepOnlypSale=True),            # Step 1: Create p_sale, keep only p_sale
            LogisticRegression(random_state=42, max_iter=1000)
        ),
        
        # ROEI pipeline - simple features
        'roei_pipeline': make_pipeline(
            FeaturesAdder(),
            GradientBoostingClassifier(n_estimators=100, random_state=42)
        )
    }
    
    print(f"✅ Created {len(pipelines)} pipeline instances:")
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
            pipelines=pipelines,  # Pass the created pipelines
            target_column='sales_count'
        )
        
        if all_results and 'error' not in all_results:
            print(f"\n✅ Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed - check error details above")
            
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

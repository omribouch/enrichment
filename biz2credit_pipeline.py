"""
Biz2Credit Pipeline - Main execution file
This is the main file to run your Biz2Credit analysis using the new framework
"""

import pandas as pd
import numpy as np
from biz2credit_framework import run_biz2credit_analysis
# Import data handler
from biz2credit_data_handler import Biz2CreditDataHandler
from biz2credit_transformers import build_pip1e, FeaturesAdder
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run Biz2Credit analysis
    """
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
    
    # Create pipeline instances (like your boss's example)
    print("\n🔧 Creating pipeline instances...")
    
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
            ),
            'roei_pipeline': build_pip1e()
        }
        
        print(f"✅ Created {len(pipelines)} pipeline instances:")
        for name, pipeline in pipelines.items():
            print(f"  - {name}: {type(pipeline).__name__}")
            print(f"    Steps: {list(pipeline.named_steps.keys())}")
        
    except Exception as e:
        print(f"❌ Error creating pipelines: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run the complete analysis
    print("\n🚀 Running complete Biz2Credit analysis...")
    
    try:
        all_results = run_biz2credit_analysis(
            df=df_biz2credit,
            pipelines=pipelines,  # Pass the created pipelines
            target_column='sales_count'
        )
        
        if all_results and 'error' not in all_results:
            print(f"\n🎉 Time Series Cross-Validation completed successfully!")
            print(f"Results available for {len(all_results)} pipelines")
            
            
        else:
            print(f"\n❌ Analysis failed - check error details above")
            
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

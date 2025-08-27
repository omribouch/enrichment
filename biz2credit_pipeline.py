"""
Biz2Credit Pipeline - Clean and Organized Version

This pipeline creates and tests company-specific models for Biz2Credit data.
Currently focused on 'ni' company with all enrichment features.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from biz2credit_transformers import (
    Biz2CreditPrep1, 
    Biz2CreditPrep1_2, 
    Biz2CreditPrep3,
    Biz2CreditPrep_keep_additional_features,
    Biz2CreditImputer, 
    Biz2CreditCategoricalEncoder, 
    FinalNumericFilter, 
    FeaturesAdder, 
    AvocadoModel,
    RemoveUserRankFeatures,
    UserRankOnlyFilter
)
from biz2credit_framework import (
    run_time_series_cv,
    print_final_comparison,
    run_pre_analysis,
    analyze_enrichment_features
    #,print_detailed_ece_analysis,
    #_aggregate_feature_importance_results,
    # create_simple_visualizations
)
from biz2credit_data_handler import Biz2CreditDataHandler

#  PIPELINE CREATION
# ============================================================================

def create_pipelines():
    """Create pipelines for comparison"""
    pipelines = {}
    
    # Define essential features once at the beginning
    enrichment_features = [
        'age_of_business_months', 
        'application_annual_revenue', 
        'business_legal_structure', 
        'loan_purpose', 
        'industry', 
        'sub_industry', 
        'users_prob_sale'
    ]
    
    other_essential_features = [
        'company', 
        'network', 
        'time_to_clickout_s_group', 
        'clickout_date_prt', 
        'leads_count', 
        'sales_count'
    ]
    
    # 1. BASELINE: Only p_sale (Step 1 only)
    baseline_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=True),  # Step 1: Create p_sale
        #Biz2CreditImputer(),                            # Step 2: Imputation
        FinalNumericFilter(),                           # Step 3: Final filtering
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    )
    # Set essential features for consistency (though not used for keepOnlypSale=True)
    # baseline_pipeline.steps[0][1].set_essential_features(enrichment_features, other_essential_features)
    pipelines['baseline_psale'] = baseline_pipeline
    
    # 1. ROEI PIPELINE: Uses Biz2CreditPrep1 for consistent p_sale creation, keeps all features
    roei_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=False, filter_to_essential=True, enrichment_features=enrichment_features, other_essential_features=other_essential_features),  # Step 1: Create p_sale, filter to essential
        FeaturesAdder(),                                # Step 2: Add features (keeps all features)
        FinalNumericFilter(),                           # Step 3: Final filtering
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    )
    
    # No need to call set_essential_features anymore - features are passed in constructor
    pipelines['roei_pipeline'] = roei_pipeline
    
    # 2. ENHANCED MODEL: All features with best params (INCLUDES new enrichment features)
    gb2_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=False, filter_to_essential=True, enrichment_features=enrichment_features, other_essential_features=other_essential_features),  # Step 1: Create p_sale, filter to essential
        Biz2CreditPrep1_2(),                            # Step 2: Original enrichment preprocessing
        Biz2CreditPrep3(),                              # Step 3: NEW enrichment features (loan_purpose, industry, sub_industry, users_prob_sale)
        Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=['p_sale']),  # Step 4: WITH additional features
        Biz2CreditImputer(),                            # Step 5: Imputation
        Biz2CreditCategoricalEncoder(),                 # Step 6: Categorical encoding
        FinalNumericFilter(),                           # Step 7: Final filtering
        GradientBoostingClassifier(
            n_estimators=200,        # Best from tuning
            learning_rate=0.1,       # Best from tuning  
            max_depth=4,             # Best from tuning
            min_samples_split=5,     # Best from tuning
            min_samples_leaf=2,      # Best from tuning
            subsample=0.9,           # Best from tuning
            max_features='sqrt',      # Best from tuning
            random_state=42
        )
    )
    
    # No need to call set_essential_features anymore - features are passed in constructor
    pipelines['biz2credit_gb_2'] = gb2_pipeline
    
    # 3. RANDOM FOREST: Good default params (INCLUDES new enrichment features)
    rf_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=False, filter_to_essential=True, enrichment_features=enrichment_features, other_essential_features=other_essential_features),  # Step 1: Create p_sale, filter to essential
        Biz2CreditPrep1_2(),                            # Step 2: Original enrichment preprocessing
        Biz2CreditPrep3(),                              # Step 3: NEW enrichment features (loan_purpose, industry, sub_industry, users_prob_sale)
        Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=['p_sale']),  # Step 4: WITH additional features
        Biz2CreditImputer(),                            # Step 5: Imputation
        Biz2CreditCategoricalEncoder(),                 # Step 6: Categorical encoding
        FinalNumericFilter(),                           # Step 7: Final filtering
        RandomForestClassifier(
            n_estimators=200,      
            max_depth=10,          
            random_state=42
        )
    )
    
    # No need to call set_essential_features anymore - features are passed in constructor
    pipelines['optimized_rf'] = rf_pipeline
    
    # AVOCADO MODEL: Custom rule-based model
    pipelines['avocado_model'] = make_pipeline(
        AvocadoModel()                                   # Custom avocado model with rules
    )
    
# 2 additional New Pipelines:
    # 1. USER_RANK_ONLY: Only users_prob_sale feature (no p_sale, no other features)
    user_rank_only_pipeline = make_pipeline(
        UserRankOnlyFilter(),  # Keep only users_prob_sale features
        RandomForestClassifier(
            n_estimators=200,      
            max_depth=10,          
            random_state=42
        )
    )
    pipelines['user_rank_only'] = user_rank_only_pipeline

    # 2.
    rf_without_user_rank_pipeline = make_pipeline(
        Biz2CreditPrep1(keepOnlypSale=False, filter_to_essential=True, enrichment_features=enrichment_features, other_essential_features=other_essential_features),  # Step 1: Create p_sale, filter to essential
        Biz2CreditPrep1_2(),                            # Step 2: Original enrichment preprocessing
        Biz2CreditPrep3(),                              # Step 3: NEW enrichment features (loan_purpose, industry, sub_industry, users_prob_sale)
        Biz2CreditPrep_keep_additional_features(keep_new_features=True, additional_features_list=['p_sale']),  # Step 4: WITH additional features
        RemoveUserRankFeatures(),
        Biz2CreditImputer(),                            # Step 5: Imputation
        Biz2CreditCategoricalEncoder(),                 # Step 6: Categorical encoding
        FinalNumericFilter(),                           # Step 7: Final filtering
        RandomForestClassifier(
            n_estimators=200,      
            max_depth=10,          
            random_state=42
        )
    )
    pipelines['rf_without_user_rank'] = rf_without_user_rank_pipeline


    return pipelines

# ============================================================================
# MAIN EXECUTION

if __name__ == "__main__":
    print("Biz2Credit Pipeline")
    print("=" * 60)
    
    # Load data (all companies, no filtering)
    print("Loading Biz2Credit data...")
    data_handler = Biz2CreditDataHandler()
    df = data_handler.load_data()
    
    if df is None or df.empty:
        print("❌ No data loaded. Exiting.")
        exit(1)
    print(f"✅ Data loaded: {df.shape}")
    
    # Skip the aggressive feature selection for now - keep all columns
    print("\n🔍 Keeping all columns for pipeline processing...")
    print(f"📊 Available columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    
    # Ensure we have the required date column
    if 'clickout_date_prt' not in df.columns:
        print("❌ Required date column 'clickout_date_prt' not found!")
        exit(1)
    
    # Convert date column to datetime for time series CV
    df['clickout_date_prt'] = pd.to_datetime(df['clickout_date_prt'])
    print(f"📅 Date column converted to datetime: {df['clickout_date_prt'].dtype}")
    
    # Get the same feature lists used in pipeline creation
    pipelines = create_pipelines()
    
    # Define the same feature lists here for consistency (instead of extracting from pipeline)
    enrichment_features = [
        'age_of_business_months', 
        'application_annual_revenue', 
        'business_legal_structure', 
        'loan_purpose', 
        'industry', 
        'sub_industry', 
        'users_prob_sale'
    ]
    
    other_essential_features = [
        'company', 
        'network', 
        'time_to_clickout_s_group', 
        'clickout_date_prt', 
        'leads_count', 
        'sales_count'
    ]
    
    print(f"\n📋 Using enrichment features: {enrichment_features}")
    print(f"📋 Using other essential features: {other_essential_features}")
    
    # Analyze enrichment features
    analyze_enrichment_features(df, enrichment_features)
    
    # Stage 1: Pre-analysis
    print("\n" + "="*60)
    print("STAGE 1: PRE-ANALYSIS")
    print("="*60)
    run_pre_analysis(df)
    # HERE OPTIONAL TO ADD VISUALIZATIONS AGAIN LATER.

    # For now, only run the 'user_rank_only' pipeline
    pipelines_to_run = {k: v for k, v in pipelines.items() } #if k == 'user_rank_only'
    print(f"\n🎯 Using {len(pipelines_to_run)} pipelines: {list(pipelines_to_run.keys())}")
    # Stage 2: Time Series Cross-Validation
    print("\n" + "="*60)
    print("STAGE 2: TIME SERIES CROSS-VALIDATION")
    print("="*60)
    
    # Prepare X and y - remove target variable from features
    X = df.drop(['sales_count'], axis=1, errors='ignore')
    y = df['sales_count']

    # Run time series CV with default parameters (90 days train, 7 days test, 7 days step)
    results = run_time_series_cv(pipelines_to_run, X, y, 'clickout_date_prt', enrichment_features)
    
    if results:
        # Print final comparison
        print_final_comparison(results)
        
        print("\n🎉 Time Series Cross-Validation completed successfully!")
        print(f"Results available for {len(results)} pipelines")
    else:
        print("❌ Analysis failed")

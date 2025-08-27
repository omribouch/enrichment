"""
Biz2Credit Data Handler
Handles data loading, preprocessing, and preparation for the pipeline

Current Configuration:
- Vertical ID: 5fe352b14e39dc2cce94d6fb
- Product ID: 11441  
- Partner ID: 11487
- Process Name: biz2credit (lead)
- Transaction Month: 2025-01
- Clickout Date: 2025-01-01


BI Configuration:
- Vertical ID: 64e33e7be3cbc4ce1041a30f
- Product ID: 13465  
- Partner ID: 13589
- Process Name: bi_biz2credit_lead

To change parameters, update the default values in load_data() and get_data() methods.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ds_aws_services import CachedAthenaApi
import os
os.environ['disk_caching'] = 'true'  # Enable caching for faster execution


class Biz2CreditDataHandler:
    """
    Handles Biz2Credit data loading and preparation
    """
    
    def __init__(self, null_threshold: float = 40.0):
        self.df_raw = None
        self.df_processed = None
        self.feature_columns = None
        self.target_column = 'sales_count'
        self.null_threshold = null_threshold
        
        # Define essential features that will be used by the models
        self.essential_features = [
            # Core features needed for p_sale calculation
            'normalized_p_cr_lead', 'normalized_p_cr_sale', 'p_cr_lead', 'p_cr_sale',
            'leads_count', 'sales_count',
            
            # Enrichment features
            'age_of_business_months', 'application_annual_revenue', 'business_legal_structure',
            'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale',
            
            # Company features (will be one-hot encoded)
            'company',
            
            # New features from query
            'network', 'time_to_clickout_s', 'time_to_clickout_s_group',
            
            # Additional features that might be needed
            'channel_click_id', 'visit_iid',
            
            # Date column needed for time series CV
            'clickout_date_prt'
        ]

    def get_data(
        self,
        partner_ids: list = [11487],  # Only ni company
        product_ids: list = [11441],  # Only ni product
        process_names: list = ['biz2credit (lead)'],  # Only ni process
        transaction_month_prt: str = '2025-01',
        vertical_ids: list = ['5fe352b14e39dc2cce94d6fb'],  # Only ni vertical
        clickout_date_prt: str = '2025-01-01'
        # BI COMPANY PARAMETERS (for future use):
        # partner_ids: [13589]  # bi company
        # product_ids: [13465]  # bi product  
        # process_names: ['bi_biz2credit_lead']  # bi process
        # vertical_ids: ['64e33e7be3cbc4ce1041a30f']  # bi vertical
    ) -> pd.DataFrame:
        print("🔍 get_data() method is executing!")
        
        query_biz = f"""
        WITH enrichment_data AS (
        SELECT 
            f.subid,
            c.visit_iid as en_visit_iid,
            c.channel_click_id as en_channel_click_id,
            process_name,
            f.partner_name,
            f.rn,
            min(f.rn) over (partition by c.visit_iid) as min_rn,
            max(f.customer_status)      AS business_legal_structure,
            max(f.age_group)            AS age_of_business_months, 
            max(f.net_revenue)          AS application_annual_revenue,
            count(distinct f.age_group) as requests_num,
            max(f.loanpurpose) as loan_purpose,
            max(sku) as industry,
            max(os) as sub_industry,
            max(user_rank) as users_prob_sale
    
    FROM dlk_visitor_funnel_dwh_production.enrich_conversions_flatten f
    LEFT JOIN dlk_visitor_funnel_dwh_production.chart_funnel c
        ON f.subid = c.cid
    WHERE f.partner_id in ({','.join(map(str, partner_ids))})
      AND process_name in ({','.join(f"'{p}'" for p in process_names)})
      AND transaction_month_prt >= '{transaction_month_prt}'
      AND f.vertical_id in ({','.join(f"'{v}'" for v in vertical_ids)})
      AND c.vertical_id in ({','.join(f"'{v}'" for v in vertical_ids)})
      AND c.product_Id in ({','.join(map(str, product_ids))})
    GROUP BY 
       f.subid, c.visit_iid, c.channel_click_id, process_name, f.partner_name, f.rn
),
enrichment_final AS (
    SELECT 
        en_visit_iid,
        en_channel_click_id,
        max(business_legal_structure) as business_legal_structure,
        max(age_of_business_months) as age_of_business_months,
        max(application_annual_revenue) as application_annual_revenue,
        max(loan_purpose) as loan_purpose,  
        max(industry) as industry,
        max(sub_industry) as sub_industry,
        max(users_prob_sale) as users_prob_sale 

    FROM enrichment_data
    WHERE rn = min_rn
    GROUP BY en_visit_iid, en_channel_click_id
),
prediction_data AS (
                       SELECT   channel_click_id,
                                visit_iid,
                                product_name,
                                max(company) as company,
                                max(coalesce(channel_country_code, country_code)) as country_code,
                                max(site_name) as site_name,
                                max(model_version) as model_version,
                                max(model_id) as model_id,
                                max(traffic_join) as channel,
                                max(coalesce(source_join, utm_source)) as source,
                                max(bucket_group) as bucket_group,
                                max(campaign_name) as campaign,
                                max(ad_group_name) as ad_group_name,
                                max(match_type) as match_type, 
                                max(agent_browser) as agent_browser,
                                max(agent_platform) as agent_platform,
                                min(clickout_timestamp) as clickout_timestamp,
                                max(model_run_id) as model_run_id,
                                max(run_id_prt) as run_id_prt,
                                max(page_type_name) as page_type_name,
                                max(gclid) as gclid,
                                max(topic) as topic,
                                max(ppc_account_name) as ppc_account_name,
                                max(landing_page_uri) as landing_page_uri,
                                max(traffic_source_name) as traffic_source_name,
                                max(pli_segment_name) as pli_segment_name,
                                max(agent_os) as agent_os,
                                max(pli_vertical_name) as pli_vertical_name,
                                min(conversion_month_prt) as conversion_month_prt,
                                min(clickout_date_prt) as clickout_date_prt,
                                min(visit_timestamp) AS visit_timestamp,
                                max(normalized_p_cr_lead) AS normalized_p_cr_lead,
                                max(avg_conversion_lag_sale) AS avg_conversion_lag_sale,
                                max(normalized_p_cr_sale) AS normalized_p_cr_sale,
                                min(p_conversion_time_sale) AS p_conversion_time_sale,
                                max(predicted_commission) AS predicted_commission,
                                max(estimated_earnings_usd) AS estimated_earnings_usd,
                                max(estimated_conversions) AS estimated_conversions,
                                max(conversion_count) AS conversion_count,
                                max(leads_count) AS leads_count,
                                max(qualified_leads_count) AS qualified_leads_count,
                                max(sales_count) AS sales_count,
                                max(p_cr_lead) as p_cr_lead,
                                max(p_cr_sale) AS p_cr_sale,
                                max(coalesce(network,'x')) as network,
                                max(date_diff('second', visit_timestamp, clickout_timestamp)) as time_to_clickout_s,
                                case 
                                    when max(date_diff('second', visit_timestamp, clickout_timestamp)) < 10 then '<10'
                                    when max(date_diff('second', visit_timestamp, clickout_timestamp)) >= 10 and max(date_diff('second', visit_timestamp, clickout_timestamp)) < 30 then '10=<Second<30'
                                    when max(date_diff('second', visit_timestamp, clickout_timestamp)) >= 30 and max(date_diff('second', visit_timestamp, clickout_timestamp)) < 60 then '30=<Second<60'  
                                    else '>=60' 
                                end as time_to_clickout_s_group

FROM dlk_mlmodels_production.v_multilabel_conversions_predictions_fast_longer
WHERE product_id in ({','.join(map(str, product_ids))})
  AND vertical_id in ({','.join(f"'{v}'" for v in vertical_ids)})
  AND clickout_date_prt >= '{clickout_date_prt}'
  and cid is not null
group by channel_click_id, visit_iid, product_name, company, country_code
                             )
select distinct p.*, 
           age_of_business_months,
           application_annual_revenue,
           business_legal_structure,
           loan_purpose,
           industry,
           sub_industry,
           users_prob_sale
from prediction_data as p
LEFT join enrichment_final as en
on (p.channel_click_id = en.en_channel_click_id and p.visit_iid = en.en_visit_iid)
"""
        
        print("🔍 About to execute SQL query...")
        raw = CachedAthenaApi().execute_fetch(query_biz)
        print("🔍 SQL query executed successfully!")
        
        df_biz_enrich = pd.DataFrame(raw)
        print(f"🔍 DataFrame created with shape: {df_biz_enrich.shape}")
        print(f"🔍 Columns: {list(df_biz_enrich.columns)}")
        
        return df_biz_enrich

    def load_data(
        self,
        partner_ids: list = [11487], #11487,13589  # ni company
        product_ids: list = [11441], #[11441, 13465],    # ni product
        process_names: list = ['biz2credit (lead)'],# ['bi_biz2credit_lead','biz2credit (lead)'], 
        transaction_month_prt: str = '2025-01',
        vertical_ids: list = ['5fe352b14e39dc2cce94d6fb'],  # ni vertical , 64e33e7be3cbc4ce1041a30f bi vertical
        clickout_date_prt: str = '2025-01-01'
    ) -> pd.DataFrame:
        """
        Load Biz2Credit data using your SQL query
        """
        print("Loading Biz2Credit data...")
        
        try:
            self.df_raw = self.get_data(
                partner_ids=partner_ids,  # Only ni partner ID
                product_ids=product_ids,  # Only ni product ID
                process_names=process_names,  # Only ni process name
                transaction_month_prt=transaction_month_prt,
                vertical_ids=vertical_ids,  # Only ni vertical ID
                clickout_date_prt=clickout_date_prt
            )
            print('data is loaded on try : load_Data')
            # Apply leads filter
            initial_shape = self.df_raw.shape
            self.df_raw = self.df_raw[self.df_raw['leads_count'] >= 1]
            
            print(f"Data loaded: {self.df_raw.shape[0]:,} rows")
            print(f"Sales rate: {self.df_raw['sales_count'].sum()/self.df_raw['leads_count'].sum()*100:.1f}%")
            
            # Feature definitions moved to pipeline - data handler only loads data
            
            # Don't filter here - let Biz2CreditPrep1 handle filtering
            print("📊 Keeping all columns for pipeline processing - Biz2CreditPrep1 will handle filtering")
            

            
            return self.df_raw
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            raise
    

    def _analyze_company_categories(self):
        """
        Analyze company categories and identify big categories for one-hot encoding
        """
        if 'company' not in self.df_raw.columns:
            return
            
        # Get company counts for analysis
        company_counts = self.df_raw['company'].value_counts()
        total_companies = len(self.df_raw)
        
        print(f"\n🏢 COMPANY CATEGORY ANALYSIS:")
        print("-" * 50)
        print(f"Total companies: {total_companies:,}")
        print(f"Unique categories: {len(company_counts)}")
        
        print(f"\nTop Company Categories (for one-hot encoding consideration):")
        print(f"{'Category':<40} {'Count':<10} {'Percentage':<10}")
        print("-" * 70)
        
        for category, count in company_counts.head(15).items():
            percentage = (count / total_companies) * 100
            # Truncate long category names
            category_display = str(category)[:40] + "..." if len(str(category)) > 40 else str(category)
            print(f"{category_display:<40} {count:<10} {percentage:<10.1f}%")
        
        # Identify big categories (suggested threshold: >1% of total)
        threshold = total_companies * 0.01  # 1% threshold
        big_categories = company_counts[company_counts > threshold]
        
        print(f"\n📊 BIG CATEGORIES (>1% = {threshold:.0f} companies) for one-hot encoding:")
        print("-" * 60)
        for category, count in big_categories.items():
            percentage = (count / total_companies) * 100
            category_display = str(category)[:50] + "..." if len(str(category)) > 50 else str(category)
            print(f"✅ {category_display:<53} {count:>6} ({percentage:>5.1f}%)")
        
        # Store big categories for later use in transformers
        self.big_company_categories = big_categories.index.tolist()
        print(f"\n💡 Suggestion: Create one-hot features for {len(self.big_company_categories)} big categories")
        print(f"   Smaller categories can be grouped into 'Other' category")
    
## ------------------------------
## COMMENTED OUT: Feature selection now handled by Biz2CreditPrep1 in pipeline
    # COMMENTED OUT: Feature selection now handled by Biz2CreditPrep1 in pipeline
    # Detect good processes automatically
    # process_quality, good_processes = self._detect_good_processes(self.df_raw)
    
    # Analyze enrichment features per process and make smart decisions
    # feature_decisions = self._analyze_enrichment_features_per_process(self.df_raw)
    
    # Update essential features based on smart decisions
    # self._update_essential_features_based_on_analysis(feature_decisions)
    
    # Show data summary
    # self._show_data_summary()
    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _filter_to_essential_features(
    #     self,
    #     enrichment_features=None,
    #     other_essential_features=None
    # ):
    #     """
    #     Filter the raw data to only include essential features needed for modeling.
    #     Now accepts parameters for enrichment_features (all enrichment, object or not)
    #     and other_essential_features (such as time, network, etc.) for greater flexibility.
    #     """
    #     initial_shape = self.df_raw.shape

    #     # Use provided params or fall back to instance attributes/defaults
    #     if enrichment_features is None:
    #         enrichment_features = getattr(self, "enrichment_features", [])
    #     if other_essential_features is None:
    #         other_essential_features = [
    #             'business_legal_structure', 'company', 'network', 'time_to_clickout_s_group', 'clickout_date_prt'
    #     ]

    #     # Compose the full list of features to keep
    #     features_to_keep = []
    #     # Add enrichment features that exist in the data
    #     features_to_keep += [col for col in enrichment_features if col in self.df_raw.columns]
    #     # Add other essential features that exist in the data
    #     features_to_keep += [col for col in other_essential_features if col in self.df_raw.columns]

    #     # Add target column if not already included
    #     if hasattr(self, "target_column") and self.target_column not in features_to_keep:
    #         features_to_keep.append(self.target_column)

        # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
        # # Remove duplicates while preserving order
        # seen = set()
        # features_to_keep = [x for x in features_to_keep if not (x in seen or seen.add(x))]

        # # Filter the dataframe
        # self.df_raw = self.df_raw[features_to_keep]

        # # Additional cleaning: remove any remaining problematic columns

        # # Handle clickout_date_prt - convert to datetime and keep for time series CV
        # if 'clickout_date_prt' in self.df_raw.columns:
        #     try:
        #         self.df_raw['clickout_date_prt'] = pd.to_datetime(self.df_raw['clickout_date_prt'])
        #     except Exception as e:
        #         print(f"Warning: Could not convert clickout_date_prt to datetime: {e}")
        #         # If conversion fails, remove the column
        #         self.df_raw = self.df_raw.drop(columns=['clickout_date_prt'])
        #         print(f"Removed clickout_date_prt due to conversion failure")

        # # Remove other datetime columns (but keep clickout_date_prt)
        # datetime_cols = self.df_raw.select_dtypes(include=['datetime64']).columns.tolist()
        # datetime_cols = [col for col in datetime_cols if col != 'clickout_date_prt']
        # if datetime_cols:
        #     print(f"Removing other datetime columns: {datetime_cols}")
        #         self.df_raw = self.df_raw.drop(columns=datetime_cols)

        # # Remove object columns except those in features_to_keep
        # object_cols = self.df_raw.select_dtypes(include=['object']).columns.tolist()
        # object_cols_to_remove = [col for col in object_cols if col not in features_to_keep]

        # if object_cols_to_remove:
        #         self.df_raw = self.df_raw.drop(columns=object_cols_to_remove)
        #         print(f"  Removed {len(object_cols_to_remove)} object columns: {object_cols_to_remove}")

        # print(f"  ✅ Data ready for pipeline processing - transformers will handle feature engineering")

        # # Company-specific feature filtering based on null rates
        # if 'company' in self.df_raw.columns:
        #     self._filter_features_by_company_null_keep = [col for col in object_cols if col not in features_to_keep]

        # if object_cols_to_remove:
        #     self.df_raw = self.df_raw.drop(columns=object_cols_to_remove)
        #     print(f"  Removed {len(object_cols_to_remove)} object columns: {object_cols_to_remove}")

        # print(f"  ✅ Data ready for pipeline processing - transformers will handle feature engineering")

        # # Company-specific feature filtering based on null rates
        # if 'company' in self.df_raw.columns:
        #     self._filter_features_by_company_null_rates()
        #     self._analyze_company_categories()
        # print("-" * 70)
    

    #------
    # COMMENTED OUT: No longer used - feature filtering now handled by Biz2CreditPrep1
    # def _filter_features_by_company_null_rates(self, enrichment_features):
    #     """
    #     Filter out features that have too many nulls (>60%) for specific companies
    #     This ensures each company only uses features with good data quality
    #     """
    #     if 'company' not in self.df_raw.columns:
    #         return
    #             
    #     print(f"\n🔍 COMPANY-SPECIFIC FEATURE FILTERING:")
    #     print("-" * 50)
    #     
    #     # Get unique companies
    #     companies = self.df_raw['company'].unique()
    #     
    #     
    #     
    #     # Check null rates per company for each enrichment feature
    #     features_to_remove = []
    #     
    #     for feature in enrichment_features:
    #         if feature not in self.df_raw.columns:
    #         continue
    #             
    #         print(f"\n📊 {feature}:")
    #         for company in companies:
    #             company_data = self.df_raw[self.df_raw['company'] == company]
    #             null_count = company_data[feature].isnull().sum()
    #             total_count = len(company_data)
    #             null_rate = null_count / total_count * 100
    #             
    #             status = "✅" if null_rate < 60 else "❌"
    #             print(f"   {company}: {null_count:,}/{total_count:,} nulls ({null_rate:.1f}%) {status}")
    #             
    #             # If any company has >60% nulls for this feature, mark it for removal
    #             if null_rate > 60:
    #                     if feature not in features_to_remove:
    #                         features_to_remove.append(feature)
    #                     print(f"      ⚠️  {feature} will be removed due to {company} having {null_rate:.1f}% nulls")
    #     
    #     # Remove features with poor data quality
    #     if features_to_remove:
    #         print(f"\n🗑️  Removing features with poor data quality: {features_to_remove}")
    #         self.df_raw = self.df_raw.drop(columns=features_to_remove)
    #         print(f"📊 After company-specific filtering: {self.df_raw.shape[0]:,} rows, {self.df_raw.shape[1]:,} columns")
    #     else:
    #         print(f"\n✅ All enrichment features have good data quality across all companies")
    
    # COMMENTED OUT: No longer used - encoding handled by transformers in pipeline
    # def get_big_company_categories(self):
    #     """
    #     Get the list of big company categories for one-hot encoding
    #     """
    #     if hasattr(self, 'big_company_categories'):
    #         return self.big_company_categories
    #     else:
    #         return []
    # 
    # def get_company_encoding_info(self):
    #     """
    #     Get information about company encoding strategy
    #     """
    #     if hasattr(self, 'big_company_categories'):
    #         return {
    #             'big_categories': self.big_company_categories,
    #             'total_categories': len(self.big_company_categories),
    #             'encoding_strategy': 'one-hot for big categories, group others'
    #         }
    #     else:
    #         return {
    #             'big_categories': [],
    #             'total_categories': 0,
    #             'encoding_strategy': 'not analyzed yet'
    #         }
    
    
    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _show_data_summary(self):
    #     """
    #     Show summary of loaded data
    #     """
    #     if self.df_raw is None:
    #         print("No data loaded yet")
    #         return
    #             
    #     print(f"Shape: {self.df_raw.shape}")
    #     print(f"Memory: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    #     
    #     # Show first few rows
    #     print("First few rows:")
    #     print(self.df_raw.head())
    #     
    #     # Show data types
    #     print("Data types:")
    #     print(self.df_raw.dtypes.value_counts())
    #     
    #     # Show key features with null rates
    #     print("Key features:")
    #     for col in ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure']:
    #         if col in df.columns:
    #             null_count = self.df_raw[col].isnull().sum()
    #             null_rate = (null_count / len(self.df_raw)) * 100
    #             print(f"  {col}: {null_count} nulls ({null_rate:.1f}%)")
    
    # COMMENTED OUT: No longer used - basic info handled by pipeline analysis
    # def get_raw_data(self):
    #     """
    #     Get the raw dataframe before filtering for avocado model
    #     """
    #     if self.df_raw is None:
    #         raise ValueError("No data loaded. Call load_data() first.")
    #     
    #     return self.df_raw.copy()
    # 
    # def get_basic_info(self) -> dict:
    #     """
    #         Get basic data information (no feature processing)
    #     """
    #     if self.df_raw is None:
    #         raise ValueError("No data loaded. Call load_data() first.")
    #     
    #     info = {
    #         'shape': self.df_raw.shape,
    #         'columns': list(self.df_raw.columns),
    #         'dtypes': self.df_raw.dtypes.to_dict(),
    #         'dtypes': self.df_raw.dtypes.to_dict(),
    #         'null_summary': self.df_raw.isnull().sum().to_dict(),
    #         'memory_usage': self.df_raw.memory_usage(deep=True).sum() / 1024**2
    #         }
    #     
    #         return info
    # 
    # def get_target_info(self) -> dict:
    #         """
    #         Get target column information
    #         """
    #         if self.df_raw is None:
    #         raise ValueError("No data loaded. Call load_data() first.")
    #         return
    #     
    #         if self.target_column not in self.df_raw.columns:
    #         raise ValueError(f"Target column '{self.target_column}' not found in data")
    #         
    #         
    #         target_info = {
    #         'null_count': self.df_raw[self.target_column].isnull().sum(),
    #         'null_percentage': (self.df_raw[self.target_column].isnull().sum() / len(self.df_raw)) * 100,
    #         'value_counts': self.df_raw[self.target_column].value_counts().to_dict(),
    #         'dtype': str(self.df_raw[self.target_column].dtype)
    #         }
    #         
    #         return target_info
    # 
    # def get_feature_info(self) -> dict:
    #         """
    #         Get information about all features (excluding target)
    #         """
    #         if self.df_raw is None:
    #         raise ValueError("No data loaded. Call load_data() first.")
    #         return
    #     
    #         feature_info = {}
    #         all_features = [col for col in self.df_raw.columns if col != self.target_column]
    #     
    #     for col in all_features:
    #         feature_info[col] = {
    #         'dtype': str(self.df_raw[col].dtype),
    #         'null_count': self.df_raw[col].isnull().sum(),
    #         'null_percentage': (self.df_raw[col].isnull().sum() / len(self.df_raw)) * 100,
    #         'unique_count': self.df_raw[col].nunique(),
    #         'is_numeric': pd.api.types.is_numeric_dtype(self.df_raw[col])
    #         }
    #     
    #         return feature_info
    # 
    # def show_feature_info(self):
    #         """
    #         Display feature information
    #         """
    #         feature_info = self.get_feature_info()
    #         
    #         print("\n=== FEATURE INFORMATION ===")
    #         for feature, info in feature_info.items():
    #         print(f"\n{feature}:")
    #         print(f"  Type: {info['dtype']}")
    #         print(f"  Nulls: {info['null_count']} ({info['null_percentage']:.1f}%)")
    #         print(f"  Unique values: {info['unique_count']}")
    #         print(f"  Numeric: {info['is_numeric']}")

    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _analyze_enrichment_features_per_process(self, df):
    #     """
    #     Analyze enrichment features availability per process to make smart feature selection decisions
    #     """
    #     print("\n🔍 ENRICHMENT FEATURE ANALYSIS PER PROCESS:")
    #     print("=" * 60)
    #     
    #     # Get enrichment features
    #     enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale']
    #     
    #     # Check if company column exists (equivalent to process_name in your data)
    #     if 'company' not in df.columns:
    #         print("⚠️ No company column found - using global analysis")
    #         return self._analyze_enrichment_features_global(df, enrichment_features)
    #     
    #     # Analyze per company (equivalent to process)
    #     process_analysis = {}
    #     for company in df['company'].unique():
    #         if pd.isna(company):
    #             continue
    #                 
    #         company_df = df[df['company'] == company]
    #         process_analysis[company] = {}
    #         
    #         print(f"\n📊 Company/Process: {company}")
    #         print(f"   Rows: {len(company_df):,}")
    #         
    #         for feature in enrichment_features:
    #             if feature in company_df.columns:
    #                     null_count = company_df[feature].isnull().sum()
    #                     total_count = len(company_df)
    #                     null_pct = (null_count / total_count) * 100
    #                     
    #                     process_analysis[company][feature] = {
    #                         'null_count': null_count,
    #                         'total_count': total_count,
    #                         'null_pct': null_pct,
    #                         'has_data': null_pct < 100
    #                     }
    #                     
    #                     status = "✅" if null_pct < 100 else "❌"
    #                     print(f"   {status} {feature}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%)")
    #             else:
    #                     process_analysis[company][feature] = {
    #                         'null_count': 0,
    #                         'total_count': 0,
    #                         'null_pct': 100,
    #                         'has_data': False
    #                     }
    #                     print(f"   ❌ {feature}: Column not found")
    #     
    #     # Make smart feature selection decisions
    #     return self._make_smart_feature_decisions(process_analysis, enrichment_features)
    
    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _analyze_enrichment_features_global(self, df, enrichment_features):
    #     """
    #     Fallback: Analyze enrichment features globally if no process breakdown
    #     """
    #     print("\n📊 GLOBAL ENRICHMENT FEATURE ANALYSIS:")
    #     print("=" * 50)
    #     
    #     global_analysis = {}
    #     for feature in enrichment_features:
    #         if feature in df.columns:
    #             null_count = df[feature].isnull().sum()
    #             total_count = len(df)
    #             null_pct = (null_count / total_count) * 100
    #             
    #             global_analysis[feature] = {
    #                 'null_count': null_count,
    #                 'total_count': total_count,
    #                 'null_pct': null_pct,
    #                 'has_data': null_pct < 100
    #         }
    #             
    #             status = "✅" if null_pct < 100 else "❌"
    #             print(f"{status} {feature}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%)")
    #         else:
    #             global_analysis[feature] = {
    #         'null_count': 0,
    #         'total_count': 0,
    #         'null_pct': 100,
    #         'has_data': False
    #     }
    #             print(f"❌ {feature}: Column not found")
    #     
    #     return self._make_smart_feature_decisions({'global': global_analysis}, enrichment_features)
    
    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _make_smart_feature_decisions(self, process_analysis, enrichment_features):
    #     """
    #     Make smart decisions about which enrichment features to use based on process analysis
    #     """
    #     print(f"\n🎯 FEATURE SELECTION SUMMARY (null threshold: {self.null_threshold}%):")
    #     print("=" * 60)
    #     
    #     feature_decisions = {}
    #     included_features = []
    #     excluded_features = []
    #     
    #     for feature in enrichment_features:
    #         # Check if ANY process has data for this feature
    #         any_process_has_data = any(
    #             process_data[feature]['has_data'] 
    #         for process_data in process_analysis.values()
    #     )
    #         
    #         if not any_process_has_data:
    #             # Feature is 100% null across ALL processes - exclude it
    #             feature_decisions[feature] = {
    #             'use_feature': False,
    #             'reason': '100% null across all processes',
    #             'imputation_needed': False
    #         }
    #             excluded_features.append(f"{feature} (100.0%)")
    #             continue
    #         
    #         # Feature has data in at least one process - analyze further
    #         processes_with_data = [
    #             process for process, process_data in process_analysis.items()
    #             if process_data[feature]['has_data']
    #         ]
    #         
    #         # Calculate overall null percentage across all processes
    #         total_rows = sum(process_data[feature]['total_count'] for process_data in process_analysis.values())
    #         total_nulls = sum(process_data[feature]['null_count'] for process_data in process_analysis.values())
    #         overall_null_pct = (total_nulls / total_rows) * 100 if total_rows > 0 else 100
    #         
    #         if overall_null_pct > self.null_threshold:
    #             # More than threshold nulls - exclude feature
    #             feature_decisions[feature] = {
    #             'use_feature': False,
    #             'reason': f'{overall_null_pct:.1f}% nulls overall (≥{self.null_threshold}% - too many)',
    #             'imputation_needed': False
    #         }
    #             excluded_features.append(f"{feature} ({overall_null_pct:.1f}%)")
    #         else:
    #             # <threshold nulls - use feature with imputation
    #             feature_decisions[feature] = {
    #             'use_feature': True,
    #             'reason': f'{overall_null_pct:.1f}% nulls overall (<{self.null_threshold}% - good for imputation)',
    #             'imputation_needed': overall_null_pct > 0,
    #             'processes_with_data': processes_with_data
    #         }
    #             included_features.append(f"{feature} ({overall_null_pct:.1f}%)")
    #     
    #     # Show minimal summary
    #     if included_features:
    #             print(f"✅ INCLUDED: {', '.join(included_features)}")
    #     if excluded_features:
    #             print(f"❌ REMOVED: {', '.join(excluded_features)}")
    #     
    #     return feature_decisions

    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _update_essential_features_based_on_analysis(self, feature_decisions):
    #     """
    #     Update the essential features list based on smart feature analysis decisions
    #     """
    #     print("\n🔄 UPDATING ESSENTIAL FEATURES BASED ON ANALYSIS:")
    #     print("=" * 60)
    #     
    #     # Get current enrichment features
    #     enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale']
    #     
    #     # NEW APPROACH: Don't remove features based on overall analysis
    #     # Instead, keep all potential enrichment features and let company-specific filtering decide
    #     print("🔒 KEEPING ALL ENRICHMENT FEATURES for company-specific analysis")
    #         print("   Company-specific filtering will determine which features to use")
    #     
    #     # Only remove features if they're 100% null across ALL companies AND we're certain they're useless
    #     features_to_remove = []
    #     for feature in enrichment_features:
    #         if feature in self.essential_features and not feature_decisions[feature]['use_feature']:
    #             # Only remove if it's truly 100% null across everything
    #             if feature_decisions[feature]['reason'].startswith('100% null'):
    #                     features_to_remove.append(feature)
    #                     self.essential_features.remove(feature)
    #                     print(f"❌ Removed {feature} from essential features: {feature_decisions[feature]['reason']}")
    #             else:
    #                     print(f"🔒 Keeping {feature} despite overall analysis: {feature_decisions[feature]['reason']}")
    #                     print(f"   Will let company-specific filtering decide")
    #     
    #     # Add back features that should be included
    #     features_to_add = []
    #     for feature in enrichment_features:
    #         if feature_fature_decisions[feature]['use_feature']:
    #             features_to_add.append(feature)
    #             self.essential_features.append(feature)
    #             print(f"✅ Added {feature} to essential features: {feature_decisions[feature]['reason']}")
    #     
    #     if not features_to_remove and not features_to_add:
    #             print("✅ No changes needed - all enrichment features are properly configured")
    #     
    #     print(f"\n📊 Updated essential features count: {len(self.essential_features)}")
    #     print(f"🔍 Enrichment features in use: {[f for f in enrichment_features if f in self.essential_features]}")
    #     
    #     # Store feature decisions for pipeline use
    #     self.feature_decisions = feature_decisions

    # COMMENTED OUT: No longer used - feature decisions handled by pipeline transformers
    # def get_feature_decisions(self):
    #     """
    #     Get the feature decisions made during analysis for use in pipeline transformers
    #     """
    #     if hasattr(self, 'feature_decisions'):
    #         return self.feature_decisions
    #     else:
    #         print("⚠️ Feature decisions not available - run load_data() first")
    #         return {}
    # 
    # def get_available_enrichment_features(self):
    #     """
    #         Get list of enrichment features that should be used in the pipeline
    #     """
    #     if hasattr(self, 'feature_decisions'):
    #         return [feature for feature, decision in self.feature_decisions.items() 
    #                if decision['use_feature']]
    #     else:
    #         print("⚠️ Feature decisions not available - run load_data() first")
    #         return []

    # COMMENTED OUT: No longer used - feature selection now handled by Biz2CreditPrep1
    # def _detect_good_processes(self, df):
    #     """
    #     Automatically detect which processes have good enrichment data and update query parameters
    #     """
    #     print("\n🔍 AUTOMATIC PROCESS DETECTION:")
    #     print("=" * 50)
    #     
    #     if 'company' not in df.columns:
    #         print("⚠️ No company column found - cannot detect good processes")
    #         return {}, []
    #     
    #     # Analyze each company for enrichment data quality (equivalent to process)
    #     process_quality = {}
    #     enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale']
    #     
    #     for company in df['company'].unique():
    #         if pd.isna(company):
    #             continue
    #                 
    #         company_df = df[df['company'] == company]
    #         process_quality[company] = {
    #             'row_count': len(company_df),
    #             'enrichment_coverage': 0,
    #             'features_with_data': 0
    #         }
    #         
    #         # Count features with data
    #         for feature in enrichment_features:
    #             if feature in company_df.columns:
    #                     null_pct = (company_df[feature].isnull().sum() / len(company_df)) * 100
    #                     if null_pct < 100: # Has some data
    #                         process_quality[company]['features_with_data'] += 1
    #         
    #         # Calculate enrichment coverage score
    #         process_quality[company]['enrichment_coverage'] = (
    #             process_quality[company]['features_with_data'] / len(enrichment_features)
    #         )
    #         
    #         print(f"📊 Company/Process: {company}:")
    #         print(f"   Rows: {process_quality[company]['row_count']:,}")
    #         print(f"   Features with data: {process_quality['features_with_data']}/{len(enrichment_features)}")
    #         print(f"   Coverage score: {process_quality[company]['enrichment_coverage']:.1%}")
    #     
    #     # Recommend good companies/processes
    #     good_processes = [
    #         company for company, quality in process_quality.items()
    #         if quality['enrichment_coverage'] > 0.3 and quality['row_count'] > 1000
    #         ]
    #     
    #     if good_processes:
    #         print(f"\n✅ RECOMMENDED COMPANIES/PROCESSES (good enrichment coverage):")
    #         for company in good_processes:
    #                 quality = process_quality[company]
    #                 print(f"   🎯 {company}: {quality['enrichment_coverage']:.1%} coverage, {quality['row_count']:,} rows")
    #         else:
    #             print(f"\n⚠️ No companies/processes with good enrichment coverage found")
    #     
    #     return process_quality, good_processes

    # COMMENTED OUT: No longer used - company analysis handled by pipeline
    # def analyze_ni_company_recent_data(self, months_back=3):
    #     """
    #     Analyze enrichment features specifically for company 'ni' in recent months
    #     to investigate high null percentages
    #     """
    #     print(f"\n🔍 DETAILED ANALYSIS: Company 'ni' - Last {months_back} Months")
    #     print("=" * 70)
    #     
    #     if self.df_raw is None or self.df_raw.empty:
    #         print("❌ No data loaded. Please run load_data() first.")
    #         return
    #     
    #     # Filter for company 'ni' and recent data
    #     ni_data = self.df_raw[self.df_raw['company'] == 'ni'].copy()
    #     
    #     if ni_data.empty:
    #         print("❌ No data found for company 'ni'")
    #         return
    #     
    #     # Convert clickout_date_prt to datetime if it's not already
    #     if 'clickout_date_prt' in ni_data.columns:
    #             ni_data['clickout_date_prt'] = pd.to_datetime(ni_data['clickout_date_prt'])
    #             
    #             # Get the most recent date and filter for last N months
    #             most_recent = ni_data['clickout_date_prt'].max()
    #             cutoff_date = most_recent - pd.DateOffset(months=months_back)
    #             
    #             recent_ni_data = ni_data[ni_data['clickout_date_prt'] >= cutoff_date].copy()
    #             
    #             print(f"📅 Date Analysis:")
    #             print(f"   Most recent date: {most_recent.strftime('%Y-%m-%d')}")
    #             print(f"   Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    #             print(f"   Total 'ni' data: {len(ni_data):,} rows")
    #             print(f"   Recent 'ni' data (last {months_back} months): {len(recent_ni_data):,} rows")
    #             
    #             # Analyze enrichment features for recent data
    #             enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 
    #                                  'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale']
    #             
    #             print(f"\n📊 ENRICHMENT FEATURES ANALYSIS - Recent 'ni' Data:")
    #             print("-" * 70)
    #             
    #             for feature in enrichment_features:
    #         if feature in recent_ni_data.columns:
    #             null_count = recent_ni_data[feature].isnull().sum()
    #             total_count = len(recent_ni_data)
    #             null_pct = (null_count / total_count) * 100
    #             
    #             if null_pct < 100:
    #                 # Show some sample values
    #                 non_null_values = recent_ni_data[feature].dropna()
    #                 unique_count = non_null_values.nunique()
    #                 sample_values = non_null_values.head(5).tolist()
    #                 
    #                 status = "✅" if null_pct < 40 else "⚠️" if null_pct < 60 else "❌"
    #                 print(f"{status} {feature:25}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%)")
    #                 print(f"   {'':25}  Unique values: {unique_count}, Sample: {sample_values}")
    #             else:
    #                 print(f"❌ {feature:25}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%) - NO DATA")
    #         else:
    #             print(f"❌ {feature:25}: Column not found in data")
    #     
    #     # Monthly breakdown
    #     print(f"\n📅 MONTHLY BREAKDOWN - Company 'ni':")
    #     print("-" * 70)
    #     
    #     # Only include features that exist in the data
    #     available_features = [f for f in ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 
    #                                     'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale'] 
    #                          if f in recent_ni_data.columns]
    #     
    #     if available_features:
    #         monthly_stats = recent_ni_data.groupby(recent_ni_data['clickout_date_prt'].dt.to_period('M')).agg({
    #             feature: ['count', lambda x: x.isnull().sum()] for feature in available_features
    #         }).round(2)
    #     else:
    #         print("   No enrichment features available for monthly analysis")
    #         monthly_stats = None
    #     
    #     # Flatten column names and print if available
    #     if monthly_stats is not None:
    #         monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    #         print(monthly_stats)
    #     else:
    #         print("   No monthly statistics available")
    #     
    #     # Check if there are any rows with ANY enrichment data (only for features that exist)
    #     available_features = [f for f in enrichment_features if f in recent_ni_data.columns]
    #     if available_features:
    #         any_enrichment = recent_ni_data[available_features].notna().any(axis=1)
    #         rows_with_enrichment = any_enrichment.sum()
    #     else:
    #         rows_with_enrichment = 0
    #     
    #     print(f"\n📊 OVERALL SUMMARY:")
    #     print(f"   Rows with ANY enrichment data: {rows_with_enrichment:,}/{len(recent_ni_data):,} ({rows_with_enrichment/len(recent_ni_data)*100:.1f}%)")
    #     
    #     if rows_with_enrichment > 0:
    #         print(f"   Rows with NO enrichment data: {len(recent_ni_data) - rows_with_enrichment:,}")
    #             
    #             # Show sample rows with enrichment data
    #             sample_enriched = recent_ni_data[any_enrichment].head(3)
    #             print(f"\n📋 Sample rows WITH enrichment data:")
    #             for idx, row in sample_enriched.iterrows():
    #             print(f"   Row {idx}:")
    #             for feature in available_features:
    #             if feature in row and pd.notna(row[feature]):
    #                 print(f"     {feature}: {row[feature]}")
    #         else:
    #             print(f"   All rows have NO enrichment data")
    #     
    #     else:
    #             print("❌ No clickout_date_prt column found for time-based analysis")
    
    # COMMENTED OUT: No longer used - data quality analysis handled by pipeline
    # def get_data_quality_summary(self):
    #     """
    #     Get a summary of data quality across all companies and time periods
    #     """
    #     print(f"\n🔍 DATA QUALITY SUMMARY - ALL COMPANIES & TIME PERIODS")
    #     print("=" * 70)
    #     
    #     if self.df_raw is None or self.df_raw.empty:
    #         print("❌ No data loaded. Please run load_data() first.")
    #         return
    #     
    #     # Overall enrichment features analysis
    #     enrichment_features = ['age_of_business_months', 'application_annual_revenue', 'business_legal_structure', 
    #                          'loan_purpose', 'industry', 'sub_industry', 'users_prob_sale']
    #     
    #     print(f"📊 OVERALL ENRICHMENT FEATURES (All Data):")
    #     print("-" * 50)
    #     
    #     for feature in enrichment_features:
    #         if feature in self.df_raw.columns:
    #             null_count = self.df_raw[feature].isnull().sum()
    #             total_count = len(self.df_raw)
    #             null_pct = (null_count / total_count) * 100
    #             
    #             if null_pct < 100:
    #                 non_null_values = self.df_raw[feature].dropna()
    #                 unique_count = non_null_values.nunique()
    #                 status = "✅" if null_pct < 40 else "⚠️" if null_pct < 60 else "❌"
    #                 print(f"{status} {feature:25}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%) - {unique_count} unique values")
    #             else:
    #                 print(f"❌ {feature:25}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%) - NO DATA")
    #         else:
    #             print(f"❌ {feature:25}: Column not found")
    #     
    #     # Company-specific analysis
    #     if 'company' in self.df_raw.columns:
    #         print(f"\n📊 COMPANY-SPECIFIC ANALYSIS:")
    #         print("-" * 50)
    #         
    #         for company in self.df_raw['company'].unique():
    #             if pd.isna(company):
    #                 continue
    #                 
    #             company_data = self.df_raw[self.df_raw['company'] == company]
    #             print(f"\n🏢 Company: {company} ({len(company_data):,} rows)")
    #             
    #             for feature in enrichment_features:
    #                 if feature in company_data.columns:
    #                         null_count = company_data[feature].isnull().sum()
    #                         total_count = len(company_data)
    #                         null_pct = (null_count / total_count) * 100
    #                         
    #                         if null_pct < 100:
    #                             non_null_values = company_data[feature].dropna()
    #                             unique_count = non_null_values.nunique()
    #                             status = "✅" if null_pct < 40 else "⚠️" if null_pct < 60 else "❌"
    #                             print(f"  {status} {feature:20}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%) - {unique_count} unique")
    #                         else:
    #                             print(f"  ❌ {feature:20}: {null_count:,}/{total_count:,} nulls ({null_pct:.1f}%) - NO DATA")
    #                     else:
    #                         print(f"  ❌ {feature:20}: Column not found")

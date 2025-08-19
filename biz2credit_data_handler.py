"""
Biz2Credit Data Handler
Handles data loading, preprocessing, and preparation for the pipeline
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ds_aws_services import CachedAthenaApi
import os
os.environ['disk_caching'] = 'true'


class Biz2CreditDataHandler:
    """
    Handles Biz2Credit data loading and preparation
    """
    
    def __init__(self):
        self.df_raw = None
        self.df_processed = None
        self.feature_columns = None
        self.target_column = 'sales_count'
        
        # Define essential features that will be used by the models
        self.essential_features = [
            # Core features needed for p_sale calculation
            'normalized_p_cr_lead', 'normalized_p_cr_sale', 'p_cr_lead', 'p_cr_sale',
            'leads_count', 'sales_count',
            
            # Enrichment features
            'age_of_business_months', 'application_annual_revenue', 'business_legal_structure',
            
            # New features from query
            'network', 'time_to_clickout_s', 'time_to_clickout_s_group',
            
            # Additional features that might be needed
            'channel_click_id', 'visit_iid',
            
            # Date column needed for time series CV
            'clickout_date_prt'
        ]

    def get_data(
        self,
        partner_id: int = 13589,
        product_id: int = 13465,
        process_name: str = 'bi_biz2credit_lead',
        transaction_month_prt: str = '2025-01',
        vertical_id: str = '64e33e7be3cbc4ce1041a30f',
        clickout_date_prt: str = '2025-01-01'
    ) -> pd.DataFrame:
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
        count(distinct f.age_group) as requests_num
    FROM dlk_visitor_funnel_dwh_production.enrich_conversions_flatten f
    LEFT JOIN dlk_visitor_funnel_dwh_production.chart_funnel c
        ON f.subid = c.cid
    WHERE f.partner_id = {partner_id}
      AND process_name  = '{process_name}'
      AND transaction_month_prt >= '{transaction_month_prt}'
      AND f.vertical_id = '{vertical_id}'
      AND c.vertical_id = '{vertical_id}'
      AND c.product_Id = {product_id}
    GROUP BY 
       1,2,3,4,5,6
),
enrichment_final AS (
    SELECT 
        en_visit_iid,
        en_channel_click_id,
        max(business_legal_structure) as business_legal_structure,
        max(age_of_business_months) as age_of_business_months,
        max(application_annual_revenue) as application_annual_revenue
    FROM enrichment_data
    WHERE rn = min_rn
    GROUP BY en_visit_iid, en_channel_click_id
),
prediction_data AS (
                       SELECT   channel_click_id,
                                visit_iid,
                                product_name,
                                coalesce(channel_country_code, country_code) as country_code,
                                coalesce(channel_region_code,ip_region_code) as region_code,
                                max(company) as company,
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
WHERE product_id = {product_id}
  AND vertical_id = '{vertical_id}'
  AND clickout_date_prt >= '{clickout_date_prt}'
  and cid is not null
group by 1, 2, 3, 4, 5
                             )
select distinct p.*, 
           age_of_business_months,
           application_annual_revenue,
           business_legal_structure
           
from prediction_data as p
LEFT join enrichment_final as en
on (p.channel_click_id = en.en_channel_click_id and p.visit_iid = en.en_visit_iid)
"""
        raw = CachedAthenaApi().execute_fetch(query_biz)
        df_biz_enrich = pd.DataFrame(raw)
        return df_biz_enrich

    def load_data(
        self,
        partner_id: int = 13589,
        product_id: int = 13465,
        process_name: str = 'bi_biz2credit_lead',
        transaction_month_prt: str = '2025-01',
        vertical_id: str = '64e33e7be3cbc4ce1041a30f',
        clickout_date_prt: str = '2025-01-01'
    ) -> pd.DataFrame:
        """
        Load Biz2Credit data using your SQL query
        """
        print("📊 Loading Biz2Credit data from SQL query...")
        
        try:
            print("Calling get_data() method...")
            self.df_raw = self.get_data()
            
            # Show BEFORE filtering summary
            print(f"\n=== BEFORE LEADS FILTERING ===")
            print(f"Initial shape: {self.df_raw.shape}")
            if 'leads_count' in self.df_raw.columns:
                print(f"Total leads: {self.df_raw['leads_count'].sum()}")
                print(f"Total sales: {self.df_raw['sales_count'].sum() if 'sales_count' in self.df_raw.columns else 'N/A'}")
                print(f"Sales rate: {self.df_raw['sales_count'].sum()/self.df_raw['leads_count'].sum()*100:.2f}%" if 'sales_count' in self.df_raw.columns else 'N/A')
            
            # Apply leads filter
            print(f"\n=== APPLYING LEADS FILTER ===")
            print(f"Filtering for leads_count >= 1 to improve data quality")
            
            initial_shape = self.df_raw.shape
            self.df_raw = self.df_raw[self.df_raw['leads_count'] >= 1]
            
            # Show AFTER filtering summary
            print(f"\n=== AFTER LEADS FILTERING ===")
            print(f"Final shape: {self.df_raw.shape}")
            print(f"Rows removed: {initial_shape[0] - self.df_raw.shape[0]} ({((initial_shape[0] - self.df_raw.shape[0])/initial_shape[0]*100):.1f}%)")
            
            if 'leads_count' in self.df_raw.columns:
                print(f"Total leads: {self.df_raw['leads_count'].sum()}")
                print(f"Total sales: {self.df_raw['sales_count'].sum() if 'sales_count' in self.df_raw.columns else 'N/A'}")
                print(f"Sales rate: {self.df_raw['sales_count'].sum()/self.df_raw['leads_count'].sum()*100:.2f}%" if 'sales_count' in self.df_raw.columns else 'N/A')
            
            print(f"✅ Data loaded and filtered successfully: {self.df_raw.shape}")
            
            # Filter to only essential columns
            self._filter_to_essential_features()
            
            # Create missing time-based features
            # self._create_time_based_features()
            
            # Show data summary
            self._show_data_summary()
            
            return self.df_raw
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            print("Full error traceback:")
            traceback.print_exc()
            raise
    
    def _filter_to_essential_features(self):
        """
        Filter the raw data to only include essential features needed for modeling
        """
        print(f"\n=== FILTERING TO ESSENTIAL FEATURES ===")
        initial_shape = self.df_raw.shape
        
        # Get columns that actually exist in the data
        available_features = [col for col in self.essential_features if col in self.df_raw.columns]
        missing_features = [col for col in self.essential_features if col not in self.df_raw.columns]
        
        print(f"Essential features found: {len(available_features)}/{len(self.essential_features)}")
        if missing_features:
            print(f"Missing features: {missing_features}")
        
        # Filter to only essential columns + any additional columns that might be needed
        columns_to_keep = available_features.copy()
        
        # Add target column if not already included
        if self.target_column not in columns_to_keep:
            columns_to_keep.append(self.target_column)
        
        # Filter the dataframe
        self.df_raw = self.df_raw[columns_to_keep]
        
        # Additional cleaning: remove any remaining problematic columns
        print(f"\n=== ADDITIONAL DATA CLEANING ===")
        
        # Handle clickout_date_prt - convert to datetime and keep for time series CV
        if 'clickout_date_prt' in self.df_raw.columns:
            try:
                self.df_raw['clickout_date_prt'] = pd.to_datetime(self.df_raw['clickout_date_prt'])
                print(f"Converted clickout_date_prt to datetime")
            except Exception as e:
                print(f"Warning: Could not convert clickout_date_prt to datetime: {e}")
                # If conversion fails, remove the column
                self.df_raw = self.df_raw.drop(columns=['clickout_date_prt'])
                print(f"Removed clickout_date_prt due to conversion failure")
        
        # Remove other datetime columns (but keep clickout_date_prt)
        datetime_cols = self.df_raw.select_dtypes(include=['datetime64']).columns.tolist()
        datetime_cols = [col for col in datetime_cols if col != 'clickout_date_prt']
        if datetime_cols:
            print(f"Removing other datetime columns: {datetime_cols}")
            self.df_raw = self.df_raw.drop(columns=datetime_cols)
        
        # Remove object columns (except business_legal_structure which will be encoded)
        object_cols = self.df_raw.select_dtypes(include=['object']).columns.tolist()
        # Keep business_legal_structure and the 3 additional features for encoding
        object_cols_to_keep = ['business_legal_structure', 'network', 'time_to_clickout_s_group']
        object_cols_to_remove = [col for col in object_cols if col not in object_cols_to_keep]
        
        if object_cols_to_keep:
            print(f"  Keeping {len(object_cols_to_keep)} object columns for pipeline processing: {object_cols_to_keep}")
        
        if object_cols_to_remove:
            self.df_raw = self.df_raw.drop(columns=object_cols_to_remove)
            print(f"  Removed {len(object_cols_to_remove)} object columns: {object_cols_to_remove}")
        
        print(f"  ✅ Data ready for pipeline processing - transformers will handle feature engineering")
    
    
    def _show_data_summary(self):
        """
        Show summary of loaded data
        """
        if self.df_raw is None:
            print("No data loaded yet")
            return
            
        print("\n=== DATA SUMMARY ===")
        print(f"Shape: {self.df_raw.shape}")
        print(f"Memory usage: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show first few rows
        print("\nFirst few rows:")
        print(self.df_raw.head())
        
        # Show data types
        print("\nData types:")
        print(self.df_raw.dtypes.value_counts())
        
        # Create comprehensive feature analysis
        print("\n=== FEATURE ANALYSIS (Sorted by Null Rate) ===")
        
        feature_analysis = []
        for col in self.df_raw.columns:
            null_count = self.df_raw[col].isnull().sum()
            null_rate = (null_count / len(self.df_raw)) * 100
            dtype = self.df_raw[col].dtype
            unique_count = self.df_raw[col].nunique()
            
            # Determine if numeric
            is_numeric = pd.api.types.is_numeric_dtype(dtype)
            
            if is_numeric:
                # Numeric features
                non_null_data = self.df_raw[col].dropna()
                if len(non_null_data) > 0:
                    min_val = non_null_data.min()
                    max_val = non_null_data.max()
                    mean_val = non_null_data.mean()
                    feature_info = {
                        'column': col,
                        'type': str(dtype),
                        'nulls': null_count,
                        'null_rate': null_rate,
                        'unique_values': unique_count,
                        'is_numeric': True,
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'top_values': None
                    }
                else:
                    feature_info = {
                        'column': col,
                        'type': str(dtype),
                        'nulls': null_count,
                        'null_rate': null_rate,
                        'unique_values': unique_count,
                        'is_numeric': True,
                        'min': 'N/A',
                        'max': 'N/A',
                        'mean': 'N/A',
                        'top_values': None
                    }
            else:
                # Categorical features
                value_counts = self.df_raw[col].value_counts()
                top_values = value_counts.head(3).index.tolist()
                feature_info = {
                    'column': col,
                    'type': str(dtype),
                    'nulls': null_count,
                    'null_rate': null_rate,
                    'unique_values': unique_count,
                    'is_numeric': False,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'top_values': top_values
                }
            
            feature_analysis.append(feature_info)
        
        # Sort by null rate (descending)
        feature_analysis.sort(key=lambda x: x['null_rate'], reverse=True)
        
        # Display feature analysis
        for feature in feature_analysis:
            print(f"\n{feature['column']}:")
            print(f"  Type: {feature['type']}")
            print(f"  Nulls: {feature['nulls']} ({feature['null_rate']:.1f}%)")
            print(f"  Unique values: {feature['unique_values']}")
            
            if feature['is_numeric']:
                print(f"  Numeric: True")
                print(f"  Min: {feature['min']}")
                print(f"  Max: {feature['max']}")
                print(f"  Mean: {feature['mean']:.2f}" if isinstance(feature['mean'], (int, float)) else f"  Mean: {feature['mean']}")
            else:
                print(f"  Numeric: False")
                if feature['top_values']:
                    print(f"  Top values: {feature['top_values']}")
                else:
                    print(f"  Top values: None")
    
    def get_basic_info(self) -> dict:
        """
        Get basic data information (no feature processing)
        """
        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.df_raw.shape,
            'columns': list(self.df_raw.columns),
            'dtypes': self.df_raw.dtypes.to_dict(),
            'null_summary': self.df_raw.isnull().sum().to_dict(),
            'memory_usage': self.df_raw.memory_usage(deep=True).sum() / 1024**2
        }
        
        return info
    
    def get_target_info(self) -> dict:
        """
        Get target column information
        """
        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            return
        
        if self.target_column not in self.df_raw.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        
        target_info = {
            'null_count': self.df_raw[self.target_column].isnull().sum(),
            'null_percentage': (self.df_raw[self.target_column].isnull().sum() / len(self.df_raw)) * 100,
            'value_counts': self.df_raw[self.target_column].value_counts().to_dict(),
            'dtype': str(self.df_raw[self.target_column].dtype)
        }
        
        return target_info
    
    def get_feature_info(self) -> dict:
        """
        Get information about all features (excluding target)
        """
        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data() first.")
            return
        
        feature_info = {}
        all_features = [col for col in self.df_raw.columns if col != self.target_column]
        
        for col in all_features:
            feature_info[col] = {
                'dtype': str(self.df_raw[col].dtype),
                'null_count': self.df_raw[col].isnull().sum(),
                'null_percentage': (self.df_raw[col].isnull().sum() / len(self.df_raw)) * 100,
                'unique_count': self.df_raw[col].nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(self.df_raw[col])
            }
        
        return feature_info
    
    def show_feature_info(self):
        """
        Display feature information
        """
        feature_info = self.get_feature_info()
        
        print("\n=== FEATURE INFORMATION ===")
        for feature, info in feature_info.items():
            print(f"\n{feature}:")
            print(f"  Type: {info['dtype']}")
            print(f"  Nulls: {info['null_count']} ({info['null_percentage']:.1f}%)")
            print(f"  Unique values: {info['unique_count']}")
            print(f"  Numeric: {info['is_numeric']}")

"""
Custom transformers for Biz2Credit data preprocessing
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Transformer(BaseEstimator, TransformerMixin):
    """Base class for all custom transformers"""
    
    def fit(self, X, y=None):
        return self._fit(X, y)
    
    def transform(self, X):
        return self._transform(X)
    
    def _fit(self, X, y=None):
        raise NotImplementedError
    
    def _transform(self, X):
        raise NotImplementedError

class Biz2CreditPrep1(Transformer):
    """First preprocessing step: creates p_sale and optionally filters to p_sale only"""
    
    def __init__(self, keepOnlypSale=False):
        self.keepOnlypSale = keepOnlypSale
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Create p_sale using normalized values if available, fallback to regular
        if 'normalized_p_cr_sale' in x.columns and 'normalized_p_cr_lead' in x.columns:
            x['p_sale'] = x['normalized_p_cr_sale'] / (x['normalized_p_cr_lead'] + 1e-8)
        elif 'p_cr_sale' in x.columns and 'p_cr_lead' in x.columns:
            x['p_sale'] = x['p_cr_sale'] / (x['p_cr_lead'] + 1e-8)
        else:
            print(f"Warning: Neither normalized_p_cr_sale/normalized_p_cr_lead nor p_cr_sale/p_cr_lead found")
            x['p_sale'] = 0.0
        
        # Clip p_sale to reasonable range
        x['p_sale'] = x['p_sale'].clip(0, 1)
        
        if self.keepOnlypSale:
            return x[['p_sale']]
        else:
            # For multi-step pipeline: keep ALL columns (both numeric and object)
            # The next transformers will handle the object columns appropriately
            return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass

class Biz2CreditPrep1_2(Transformer):
    """Second preprocessing step: handles age, revenue, and null indicators"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()

        # Age enrichment - ONLY convert months to years
        if 'age_of_business_months' in x.columns:
            age_max = x['age_of_business_months'].max()
            # Remove verbose logging - just keep essential info
            
            # Convert months to years (keep as numeric)
            x['age_of_business_years'] = x['age_of_business_months'] / 12
            
            # Create null indicator
            x['isnull_age'] = x['age_of_business_months'].isnull().astype(int)
            
            # Drop original months column
            x = x.drop(columns=['age_of_business_months'])

        # Revenue enrichment  
        if 'application_annual_revenue' in x.columns:
            x['log_annual_revenue'] = np.log1p(x['application_annual_revenue'])
            x['isnull_revenue'] = x['application_annual_revenue'].isnull().astype(int)
        
        # Business legal structure enrichment
        if 'business_legal_structure' in x.columns:
            x['business_legal_structure'] = x['business_legal_structure'].fillna('Unknown')
            
            # Create 3 quality tiers (high, mid, low)
            x['business_quality_high'] = (x['business_legal_structure'] == 'Corporation').astype(int)
            x['business_quality_mid'] = (x['business_legal_structure'] == 'Limited Liability Company').astype(int)
            x['business_quality_low'] = (x['business_legal_structure'].isin(['Sole Proprietorship', 'Unknown'])).astype(int)
            
            x = x.drop(columns=['business_legal_structure'])
        
        # Interaction terms
        if 'p_sale' in x.columns:
            if 'isnull_age' in x.columns:
                x['p_sale_x_isnull_age'] = x['p_sale'] * x['isnull_age']
            if 'isnull_revenue' in x.columns:
                x['p_sale_x_isnull_revenue'] = x['p_sale'] * x['isnull_revenue']
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll add the method for consistency
        pass

class Biz2CreditPrep_keep_additional_features(Transformer):
    """Handles additional features (network, time_to_clickout_s, time_to_clickout_s_group)"""
    
    def __init__(self, additional_features_list=None, keep_new_features=True):
        self.additional_features_list = additional_features_list or ['network', 'time_to_clickout_s', 'time_to_clickout_s_group']
        self.keep_new_features = keep_new_features
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        if not self.keep_new_features:
            # Drop additional features if not keeping them
            for feature in self.additional_features_list:
                if feature in x.columns:
                    x = x.drop(columns=[feature])
            return x
        else:   # Process additional features if keeping them
            
            if 'network' in x.columns:
                # Fill NaN values with 'x' (most common after g, o, s)
                if x['network'].isnull().any():
                    x['network'] = x['network'].fillna('x')
                
                # Group 1: Google network
                x['network_g'] = (x['network'] == 'g').astype(int)
                # Group 2: Other specific networks
                x['network_o'] = (x['network'] == 'o').astype(int)
                # Group 3: Combined s, x, and any other networks (prevents feature mismatch)
                x['network_sx'] = (x['network'].isin(['s', 'x'])).astype(int)
                
                x = x.drop(columns=['network'])
            
            if 'time_to_clickout_s' in x.columns: # Removing
                # Fill NaN with median
                # if x['time_to_clickout_s'].isnull().any():
                #     x['time_to_clickout_s'] = x['time_to_clickout_s'].fillna(x['time_to_clickout_s'].median())
            
                # Remove time_to_clickout_s (too granular and noisy)
                x = x.drop(columns=['time_to_clickout_s'])


            if 'time_to_clickout_s_group' in x.columns:
                # Fill NaN values with '<10' (most common)
                if x['time_to_clickout_s_group'].isnull().any():
                    x['time_to_clickout_s_group'] = x['time_to_clickout_s_group'].fillna('<10')
                
                # Create CLEAN feature names that XGBoost will accept (no special characters)
                # Map the original values to clean names
                time_group_mapping = {
                    '<10': 'under_10_seconds',
                    '10=<Second<30': '10_to_30_seconds', 
                    '30=<Second<60': '30_to_60_seconds',
                    '>=60': 'over_60_seconds'
                }
                
                # Create clean one-hot encoded features
                for original_value, clean_name in time_group_mapping.items():
                    x[f'time_group_{clean_name}'] = (x['time_to_clickout_s_group'] == original_value).astype(int)
                
                x = x.drop(columns=['time_to_clickout_s_group'])
            

            return x
    
    def set_model_name(self, name):
        self.model_name = name
    

class Biz2CreditImputer(Transformer):
    """Imputes missing values"""
    
    def __init__(self):
        self.model_name = "Unknown"  
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        # Work with the input DataFrame directly
        x = X.copy()
        
        # Handle numeric columns
        numeric_cols = x.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if x[col].isnull().any():
                if col == 'p_sale':
                    # For p_sale, use a small positive value
                    x[col] = x[col].fillna(0.0001)
                else:
                    # For other numeric columns, use median
                    x[col] = x[col].fillna(x[col].median())
        
        # Handle categorical columns
        categorical_cols = x.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if x[col].isnull().any():
                if col == 'age_group':
                    # Use the new age categories
                    x[col] = x[col].fillna('0-1')
                else:
                    # For other categorical columns, use mode
                    mode_val = x[col].mode().iloc[0] if not x[col].mode().empty else 'Unknown'
                    x[col] = x[col].fillna(mode_val)
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    

class Biz2CreditCategoricalEncoder(Transformer):
    """One-hot encodes categorical columns with consistent feature names"""
    
    def __init__(self):
        # Set default expected categories to ensure consistency even without fitting
        self.expected_categories = {
            'age_group': ['0-1', '1-3', '3-5', '5+']  # Updated to match new grouping
        }
        self.model_name = "Unknown" 
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        # Update expected categories based on actual data if available
        if 'age_group' in X.columns:
            self.expected_categories['age_group'] = ['0-1', '1-3', '3-5', '5+']
        
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Handle age groups and encoding
        if 'age_of_business_years' in x.columns:
            # Create age groups
            x['age_group'] = pd.cut(x['age_of_business_years'], 
                                   bins=[0, 1, 3, 5, float('inf')], 
                                   labels=['0-1', '1-3', '3-5', '5+'], 
                                   include_lowest=True)
            x['age_group'] = x['age_group'].fillna('0-1')
            
            # One-hot encode age groups
            for category in ['0-1', '1-3', '3-5', '5+']:
                x[f'age_group_{category}'] = (x['age_group'] == category).astype(int)
            
            # Clean up intermediate columns
            x = x.drop(columns=['age_group', 'age_of_business_years'])
            # Remove verbose logging - just keep essential info
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll keep the method for consistency
        pass

class FeaturesAdder(Transformer):
    """Adds additional features to the dataset"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Minimal NaN handling - just fill basic NaNs without changing core logic
        if 'age_of_business_months' in x.columns and x['age_of_business_months'].isnull().any():
            x['age_of_business_months'] = x['age_of_business_months'].fillna(x['age_of_business_months'].median())
        
        if 'application_annual_revenue' in x.columns and x['application_annual_revenue'].isnull().any():
            x['application_annual_revenue'] = x['application_annual_revenue'].fillna(x['application_annual_revenue'].median())
        
        # Create p_sale feature
        if 'normalized_p_cr_sale' in x.columns and 'normalized_p_cr_lead' in x.columns:
            x['p_sale'] = x['normalized_p_cr_sale'] / (x['normalized_p_cr_lead'] + 1e-8)
        elif 'p_cr_sale' in x.columns and 'p_cr_lead' in x.columns:
            x['p_sale'] = x['p_cr_sale'] / (x['p_cr_lead'] + 1e-8)
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll add the method for consistency
        pass

class FinalNumericFilter(Transformer):
    """Final transformer that ensures only numeric columns reach the model"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Keep only numeric columns
        numeric_cols = x.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print(f"        ⚠️ No numeric columns found in FinalNumericFilter")
            return pd.DataFrame()  # Return empty DataFrame if no numeric columns
        
        x_filtered = x[numeric_cols]
        
        # Ensure we return a pandas DataFrame, not numpy array
        if not isinstance(x_filtered, pd.DataFrame):
            x_filtered = pd.DataFrame(x_filtered, columns=numeric_cols)
        
        # Remove verbose logging - just keep essential info
        return x_filtered
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass


class FeatureSelector(Transformer):
    """Selects only the most predictive features, removing noisy ones"""
    
    def __init__(self, keep_features=None):
        self.keep_features = keep_features or [
            'normalized_p_cr_lead', 'normalized_p_cr_sale', 'p_cr_lead', 'p_cr_sale',
            'leads_count', 'age_of_business_months', 'application_annual_revenue',
            'p_sale', 'business_quality_high', 'business_quality_mid', 'business_quality_low'
        ]
        self.model_name = "Unknown"
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Keep only the most predictive features
        available_features = [f for f in self.keep_features if f in x.columns]
        x_selected = x[available_features]
        
        return x_selected
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        pass


def build_pip1e(transformer_list):
    """Builds a pipeline from a list of transformers"""
    return transformer_list






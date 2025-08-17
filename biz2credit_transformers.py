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
            print(f"Model: {self.model_name}, Completed step 1: baseline mode, returning only p_sale, shape: {x.shape}")
            return x[['p_sale']]
        else:
            print(f"Model: {self.model_name}, Completed step 1: p_sale creation, shape: {x.shape}")
            print(f"      🔍 Features after step 1: {list(x.columns)}")
            return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll add the method for consistency
        pass

class Biz2CreditPrep1_2(Transformer):
    """Second preprocessing step: handles age, revenue, and null indicators"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Handle age of business (convert to months and create bins)
        if 'age_of_business_months' in x.columns:
            # Convert to months if it's in years
            if x['age_of_business_months'].max() > 1000:  # Likely in years
                x['age_of_business_months'] = x['age_of_business_months'] * 12
            
            # Create age groups using original logic
            x['age_group'] = pd.cut(x['age_of_business_months'], 
                                   bins=[0, 12, 36, 60, float('inf')], 
                                   labels=['0-1', '1-3', '3-5', '5+'], 
                                   include_lowest=True)
            x['age_group'] = x['age_group'].astype(str)
            
            # Create null indicator
            x['isnull_age'] = x['age_of_business_months'].isnull().astype(int)
        
        # Handle revenue (log transform and bins)
        if 'application_annual_revenue' in x.columns:
            # Create log revenue
            x['log_annual_revenue'] = np.log1p(x['application_annual_revenue'])
            
            # Create revenue tiers using original logic
            x['revenue_tier'] = pd.cut(x['application_annual_revenue'], 
                                     bins=[0, 100000, 500000, 1000000, float('inf')], 
                                     labels=['0-100K', '100K-500K', '500K-1M', '1M+'], 
                                     include_lowest=True)
            x['revenue_tier'] = x['revenue_tier'].astype(str)
            
            # Create null indicator
            x['isnull_revenue'] = x['application_annual_revenue'].isnull().astype(int)
        
        # Handle business legal structure null indicator
        if 'business_legal_structure' in x.columns:
            x['isnull_legal'] = x['business_legal_structure'].isnull().astype(int)
        
        # Create interaction terms
        if 'p_sale' in x.columns:
            if 'isnull_age' in x.columns:
                x['p_sale_x_isnull_age'] = x['p_sale'] * x['isnull_age']
            if 'isnull_revenue' in x.columns:
                x['p_sale_x_isnull_revenue'] = x['p_sale'] * x['isnull_revenue']
        
        print(f"Model: {self.model_name}, Completed step 2: enrichment preprocessing, shape: {x.shape}")
        print(f"      🔍 Features after step 2: {list(x.columns)}")
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
    
    def _fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        if not self.keep_new_features:
            # Drop additional features if not keeping them
            for feature in self.additional_features_list:
                if feature in x.columns:
                    x = x.drop(columns=[feature])
            print(f"Model: {self.model_name}, Completed step 3: dropped additional features, shape: {x.shape}")
            return x
        else:
            # Process additional features if keeping them
            
            if 'network' in x.columns:
                # Fill NaN values with 'x' (most common after g, o, s)
                if x['network'].isnull().any():
                    x['network'] = x['network'].fillna('x')
                
                # Use exact values from the data: g, o, s, x, other
                networks_to_encode = ['g', 'o', 's', 'x']
                
                # One-hot encode network with exact categories
                for network in networks_to_encode:
                    x[f'network_{network}'] = (x['network'] == network).astype(int)
                # Create 'other' category for any other networks (a, {Network}, etc.)
                x['network_other'] = (~x['network'].isin(networks_to_encode)).astype(int)
                x = x.drop(columns=['network'])
            
            if 'time_to_clickout_s' in x.columns:
                # Fill NaN with median
                if x['time_to_clickout_s'].isnull().any():
                    x['time_to_clickout_s'] = x['time_to_clickout_s'].fillna(x['time_to_clickout_s'].median())
            
            if 'time_to_clickout_s_group' in x.columns:
                # Fill NaN values with '<10' (most common)
                if x['time_to_clickout_s_group'].isnull().any():
                    x['time_to_clickout_s_group'] = x['time_to_clickout_s_group'].fillna('<10')
                
                # Use exact values from the data: <10, 10=<Second<30, 30=<Second<60, >=60
                time_groups_to_encode = ['<10', '10=<Second<30', '30=<Second<60', '>=60']
                
                # One-hot encode time groups with exact categories
                for time_group in time_groups_to_encode:
                    x[f'time_to_clickout_s_group_{time_group}'] = (x['time_to_clickout_s_group'] == time_group).astype(int)
                x = x.drop(columns=['time_to_clickout_s_group'])
            
            print(f"Model: {self.model_name}, Completed step 3: processed additional features, shape: {x.shape}")
            print(f"      🔍 Features after step 3: {list(x.columns)}")
            return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer no longer needs external categories, but we'll keep the method for consistency
        pass

class Biz2CreditImputer(Transformer):
    """Imputes missing values"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None):
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
                if col == 'revenue_tier':
                    x[col] = x[col].fillna('0-100K')
                elif col == 'age_group':
                    x[col] = x[col].fillna('0-1')
                else:
                    # For other categorical columns, use mode
                    mode_val = x[col].mode().iloc[0] if not x[col].mode().empty else 'Unknown'
                    x[col] = x[col].fillna(mode_val)
        
        print(f"Model: {self.model_name}, Completed step 4: imputation, shape: {x.shape}")
        print(f"      🔍 Features after step 4: {list(x.columns)}")
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll add the method for consistency
        pass

class Biz2CreditCategoricalEncoder(Transformer):
    """One-hot encodes categorical columns with consistent feature names"""
    
    def __init__(self):
        # Set default expected categories to ensure consistency even without fitting
        self.expected_categories = {
            'revenue_tier': ['0-100K', '100K-500K', '500K-1M', '1M+'],
            'age_group': ['0-1', '1-3', '3-5', '5+']
        }
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None):
        # Update expected categories based on actual data if available
        if 'revenue_tier' in X.columns:
            self.expected_categories['revenue_tier'] = ['0-100K', '100K-500K', '500K-1M', '1M+']
        
        if 'age_group' in X.columns:
            self.expected_categories['age_group'] = ['0-1', '1-3', '3-5', '5+']  # Remove 'unknown', start with '0-1'
        
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Debug: Check what columns we actually have
        print(f"      🔍 DEBUG: Columns in transformer: {list(x.columns)}")
        print(f"      🔍 DEBUG: business_legal_structure present: {'business_legal_structure' in x.columns}")
        if 'business_legal_structure' in x.columns:
            print(f"      🔍 DEBUG: business_legal_structure dtype: {x['business_legal_structure'].dtype}")
            print(f"      🔍 DEBUG: business_legal_structure unique values: {x['business_legal_structure'].unique()}")
            print(f"      🔍 DEBUG: business_legal_structure sample: {x['business_legal_structure'].head(3).tolist()}")
        
        # Handle business_legal_structure with quality grouping (3 groups)
        if 'business_legal_structure' in x.columns:
            # Fill NaN values
            if x['business_legal_structure'].isnull().any():
                x['business_legal_structure'] = x['business_legal_structure'].fillna('Unknown')
            
            # Create 3 quality groups
            x['business_quality_high'] = (x['business_legal_structure'] == 'Corporation').astype(int)
            x['business_quality_mid'] = (x['business_legal_structure'].isin(['Limited Liability Company', 'Partnership', 'LLC', 'Non Profit Corp', 'Limited Partnership'])).astype(int)
            x['business_quality_low'] = (x['business_legal_structure'].isin(['Sole Proprietorship', 'Unknown', "I don't Know", 'I just do not know'])).astype(int)
            
            # Drop original column
            x = x.drop(columns=['business_legal_structure'])
            print(f"      🔍 DEBUG: Created business quality columns and dropped original")
        else:
            print(f"      🔍 DEBUG: business_legal_structure column NOT found!")
        
        # Handle other categorical columns with expected categories
        categorical_cols = ['revenue_tier', 'age_group']
        for col in categorical_cols:
            if col in x.columns:
                # Fill NaN values before encoding
                if x[col].isnull().any():
                    if col == 'revenue_tier':
                        x[col] = x[col].fillna('0-100K')
                    elif col == 'age_group':
                        x[col] = x[col].fillna('0-1')
                
                # Create dummies with consistent categories
                if col in self.expected_categories:
                    # Use expected categories to ensure consistency
                    for category in self.expected_categories[col]:
                        x[f'{col}_{category}'] = (x[col] == category).astype(int)
                else:
                    # Fallback to regular get_dummies if no expected categories
                    dummies = pd.get_dummies(x[col], prefix=col)
                    x = pd.concat([x, dummies], axis=1)
                
                # Drop original column
                x = x.drop(columns=[col])
        
        print(f"Model: {self.model_name}, Completed step 5: categorical encoding, shape: {x.shape}")
        print(f"      🔍 DEBUG: Final columns after encoding: {list(x.columns)}")
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        # This transformer doesn't need external categories, but we'll add the method for consistency
        pass

class FeaturesAdder(Transformer):
    """Adds additional features to the dataset"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None):
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
        
        print(f"Model: {self.model_name}, Completed step 1, shape: {x.shape}")
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
    
    def _fit(self, X, y=None):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # Keep only numeric columns
        numeric_cols = x.select_dtypes(include=[np.number]).columns
        x_filtered = x[numeric_cols]
        
        # Ensure we return a pandas DataFrame, not numpy array
        if not isinstance(x_filtered, pd.DataFrame):
            x_filtered = pd.DataFrame(x_filtered, columns=numeric_cols)
        
        print(f"Model: {self.model_name}, Completed final filtering, shape: {x_filtered.shape}")
        print(f"      🔍 Final numeric features: {list(x_filtered.columns)}")
        return x_filtered
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass


def build_pip1e(transformer_list):
    """Builds a pipeline from a list of transformers"""
    return transformer_list






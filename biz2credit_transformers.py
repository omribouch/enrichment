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
    """First preprocessing step: creates p_sale and filters to essential features"""
    
    def __init__(self, keepOnlypSale=False, filter_to_essential=True, enrichment_features=None, other_essential_features=None):
        self.keepOnlypSale = keepOnlypSale
        self.filter_to_essential = filter_to_essential
        self.model_name = "Unknown"  # Will be set by framework
        self.essential_enrichment_features = enrichment_features if enrichment_features is not None else []
        self.other_essential_features = other_essential_features if other_essential_features is not None else []
    
    def set_essential_features(self, enrichment_features, other_essential_features):
        """Set the essential features to keep after p_sale generation (for backward compatibility)"""
        self.essential_enrichment_features = enrichment_features or []
        self.other_essential_features = other_essential_features or []
    
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
            x['p_sale'] = 0.0
        
        # Clip p_sale to reasonable range
        x['p_sale'] = x['p_sale'].clip(0, 1)
        
        if self.keepOnlypSale:
            return x[['p_sale']]
        elif self.filter_to_essential:
            # STEP 1: First filter to essential features + p_sale
            columns_to_keep = ['p_sale'] + self.essential_enrichment_features + self.other_essential_features
            available_columns = [col for col in columns_to_keep if col in x.columns]
            
            # Add any missing essential columns that exist in the data
            for col in columns_to_keep:
                if col not in available_columns and col in x.columns:
                    available_columns.append(col)
            
            filtered_x = x[available_columns]
            
            # STEP 2: Then remove features with >60% nulls from the essential features
            features_to_remove = []
            for col in filtered_x.columns:
                if col != 'p_sale':  # Don't remove p_sale
                    null_pct = (filtered_x[col].isnull().sum() / len(filtered_x)) * 100
                    if null_pct > 60:
                        features_to_remove.append(col)
            
            if features_to_remove:
                filtered_x = filtered_x.drop(columns=features_to_remove)
            
            return filtered_x
        else:
            # Keep all columns (for multi-step pipeline)
            return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass
    
    def get_params(self, deep=True):
        """Get parameters for this estimator - required for scikit-learn compatibility"""
        return {
            'keepOnlypSale': self.keepOnlypSale,
            'filter_to_essential': self.filter_to_essential,
            'enrichment_features': self.essential_enrichment_features,
            'other_essential_features': self.other_essential_features
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator - required for scikit-learn compatibility"""
        for key, value in params.items():
            if key == 'enrichment_features':
                self.essential_enrichment_features = value
            elif key == 'other_essential_features':
                self.other_essential_features = value
            elif hasattr(self, key):
                setattr(self, key, value)
        return self

class Biz2CreditPrep1_2(Transformer):
    """Second preprocessing step: handles ALL enrichment features (original + new)"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        

        
        # Age enrichment - convert months to years
        if 'age_of_business_months' in x.columns:
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
        
        # Interaction terms - only create if p_sale exists and is valid
        if 'p_sale' in x.columns and x['p_sale'].notna().any():
            try:
                if 'isnull_age' in x.columns:
                    x['p_sale_x_isnull_age'] = x['p_sale'] * x['isnull_age']
                if 'isnull_revenue' in x.columns:
                    x['p_sale_x_isnull_revenue'] = x['p_sale'] * x['isnull_revenue']
            except Exception as e:
                print(f"⚠️ Warning: Could not create interaction terms with p_sale: {e}")
                # Continue without interaction terms rather than crashing
        else:
            print(f"ℹ️ p_sale not available for interaction terms - skipping")
        
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
                median_time = x['time_to_clickout_s'].median()
                x['time_to_clickout_s'] = x['time_to_clickout_s'].fillna(median_time)
                
                # Create time-based features
                x['time_to_clickout_fast'] = (x['time_to_clickout_s'] < 10).astype(int)
                x['time_to_clickout_medium'] = ((x['time_to_clickout_s'] >= 10) & (x['time_to_clickout_s'] < 30)).astype(int)
                x['time_to_clickout_slow'] = (x['time_to_clickout_s'] >= 30).astype(int)
                
                x = x.drop(columns=['time_to_clickout_s'])
            
            if 'time_to_clickout_s_group' in x.columns:
                # Create one-hot encoding for time groups
                time_groups = ['<10', '10=<Second<30', '30=<Second<60', '>=60']
                for group in time_groups:
                    col_name = f'time_group_{group.replace("<", "lt_").replace(">=", "gte_").replace("=", "_").replace("Second", "sec")}'
                    x[col_name] = (x['time_to_clickout_s_group'] == group).astype(int)
                
                x = x.drop(columns=['time_to_clickout_s_group'])
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass

class Biz2CreditPrep3(Transformer):
    """Third preprocessing step: handles NEW enrichment features (loan_purpose, industry, sub_industry, users_prob_sale)"""
    
    def __init__(self):
        self.model_name = "Unknown"  # Will be set by framework
    
    def _fit(self, X, y=None, sample_weight=None, **fit_params):
        return self
    
    def _transform(self, X):
        x = X.copy()
        
        # NEW: Loan purpose enrichment - log transformation + numeric
        try:
            if 'loan_purpose' in x.columns:
                # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                x['loan_purpose'] = pd.to_numeric(x['loan_purpose'], errors='coerce').fillna(0)
                
                # Apply log transformation (like revenue)
                x['log_loan_purpose'] = np.log1p(x['loan_purpose'])
                
                # Create null indicator
                x['isnull_loan_purpose'] = (x['loan_purpose'] == 0).astype(int)
                
                # Drop original column
                x = x.drop(columns=['loan_purpose'])
        except Exception as e:
            print(f"    ⚠️ Error processing loan_purpose: {e}")
        
        # NEW: Industry enrichment - top 8 one-hot encoding
        try:
            if 'industry' in x.columns:
                # Fill NaN with 'Unknown'
                x['industry'] = x['industry'].fillna('Unknown')
                
                # Top 8 industries (until Health Care) based on your data
                top_industries = [
                    'Other Services (except Public Administration)',
                    'Construction', 
                    'Retail Trade',
                    'Transportation and Warehousing',
                    'Accommodation and Food Services',
                    'ProfessionalScientificand Technical Services',
                    'Health Care and Social Assistance',
                    'ArtsEntertainmentand Recreation'
                ]
                
                # Create one-hot features for top industries
                for industry in top_industries:
                    col_name = f'industry_{industry.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")}'
                    x[col_name] = (x['industry'] == industry).astype(int)
                
                # Create 'Other' category for remaining industries
                x['industry_Other'] = (~x['industry'].isin(top_industries)).astype(int)
                
                # Drop original column
                x = x.drop(columns=['industry'])
        except Exception as e:
            print(f"    ⚠️ Error processing industry: {e}")
        
        # Sub-industry enrichment - special combinations
        try:
            if 'sub_industry' in x.columns:
                # Fill NaN with 'Unknown'
                x['sub_industry'] = x['sub_industry'].fillna('Unknown')
                
                # Special high-L2S combinations
                x['sub_industry_full_service_restaurants'] = (x['sub_industry'] == 'Full-Service Restaurants').astype(int)
                x['sub_industry_residential_construction'] = (x['sub_industry'] == 'Residential Building Construction').astype(int)
                x['sub_industry_hvac_plumbing'] = (x['sub_industry'] == 'HVACPlumbing and Electrician').astype(int)
                
                # Construction split
                x['sub_industry_construction'] = x['sub_industry'].str.contains('Construction', na=False).astype(int)
                
                # Create 'Other' category for remaining sub-industries
                x['sub_industry_Other'] = (~x['sub_industry'].isin([
                    'Full-Service Restaurants', 'Residential Building Construction', 
                    'HVACPlumbing and Electrician'
                ])).astype(int)
                
                # Drop original column
                x = x.drop(columns=['sub_industry'])
        except Exception as e:
            print(f"    ⚠️ Error processing sub_industry: {e}")
        
        # Users probability of sale enrichment - smart grouping
        try:
            if 'users_prob_sale' in x.columns:
                # Fill NaN with 'Unknown'
                
                # Smart grouping based on your analysis
                x['users_prob_very_low'] = x['users_prob_sale'].str.contains('Very Low', na=False).astype(int)
                x['users_prob_low'] = (x['users_prob_sale'].str.contains('Low', na=False) & 
                                      ~x['users_prob_sale'].str.contains('Very Low', na=False)).astype(int)
                x['users_prob_medium'] = x['users_prob_sale'].str.contains('Medium', na=False).astype(int)
                x['users_prob_high'] = (x['users_prob_sale'].str.contains('High', na=False) & 
                                       ~x['users_prob_sale'].str.contains('Very High', na=False)).astype(int)
                x['users_prob_very_high'] = x['users_prob_sale'].str.contains('Very High', na=False).astype(int)
                
                # Drop original column
                x = x.drop(columns=['users_prob_sale'])
        except Exception as e:
            print(f"    ⚠️ Error processing users_prob_sale: {e}")
        
        return x
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass

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
        
        # Note: p_sale creation moved to Biz2CreditPrep1 for consistency across all models
        
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
            print(f"  No numeric columns found in FinalNumericFilter")
            return pd.DataFrame()  # Return empty DataFrame if no numeric columns
        
        x_filtered = x[numeric_cols]
        
        # Ensure we return a pandas DataFrame, not numpy array
        if not isinstance(x_filtered, pd.DataFrame):
            x_filtered = pd.DataFrame(x_filtered, columns=numeric_cols)
        
        return x_filtered
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass

class RemoveUserRankFeatures(BaseEstimator, TransformerMixin):
    """Remove users_prob_sale encoded features from the pipeline"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        x = X.copy()
        
        # Remove the encoded users_prob_sale features
        cols_to_remove = [
            'users_prob_very_low',
            'users_prob_low', 
            'users_prob_medium',
            'users_prob_high',
            'users_prob_very_high'
        ]
        print(f"Removing columns: {cols_to_remove}")
        # Only remove columns that exist
        existing_cols_to_remove = [col for col in cols_to_remove if col in x.columns]
        if existing_cols_to_remove:
            x = x.drop(columns=existing_cols_to_remove)
        
        print(' all columns remaining: ', x.columns)
        return x

class UserRankOnlyFilter(BaseEstimator, TransformerMixin):
    """Keep only users_prob_sale feature and apply one-hot encoding"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        x = X.copy()
        
        # Keep only users_prob_sale column
        if 'users_prob_sale' in x.columns:
            x = x[['users_prob_sale']]
            
            # Apply the same one-hot encoding as in Biz2CreditPrep3
            try:
                # Smart grouping based on your analysis
                x['users_prob_very_low'] = x['users_prob_sale'].str.contains('Very Low', na=False).astype(int)
                x['users_prob_low'] = (x['users_prob_sale'].str.contains('Low', na=False) & 
                                      ~x['users_prob_sale'].str.contains('Very Low', na=False)).astype(int)
                x['users_prob_medium'] = x['users_prob_sale'].str.contains('Medium', na=False).astype(int)
                x['users_prob_high'] = (x['users_prob_sale'].str.contains('High', na=False) & 
                                       ~x['users_prob_sale'].str.contains('Very High', na=False)).astype(int)
                x['users_prob_very_high'] = x['users_prob_sale'].str.contains('Very High', na=False).astype(int)
                
                # Drop original column
                x = x.drop(columns=['users_prob_sale'])
            except Exception as e:
                print(f"    ⚠️ Error processing users_prob_sale: {e}")
        else:
            # If no users_prob_sale column, create dummy features
            x = pd.DataFrame({
                'users_prob_very_low': [0] * len(X),
                'users_prob_low': [0] * len(X),
                'users_prob_medium': [0] * len(X),
                'users_prob_high': [0] * len(X),
                'users_prob_very_high': [0] * len(X)
            })
        
        return x

class AvocadoModel:
    """Custom avocado model that creates rules based on age and revenue conditions"""
    
    def __init__(self):
        self.vals_ = None
        self.model_name = "Unknown"
        # Rule groups will be populated with actual data-driven probabilities during training
        self.rule_groups = []
        self.rule_conditions = [
            {"desc": "Age < 12 or Invalid Legal"},
            {"desc": "Age 12-17, Revenue < 120K"},
            {"desc": "Age ≥ 18, Revenue < 120K"},
            {"desc": "Age 12-17, Revenue ≥ 120K"},
            {"desc": "Age ≥ 18, Revenue ≥ 120K"}
        ]
    
    def fit(self, X, y=None, sample_weight=None, **fit_params): #-> "Estimator"
        """Fit the model by calculating average target values for each rule group"""
        x = X.copy()
        
        # Ensure age is numeric (in months)
        if 'age_of_business_months' in x.columns:
            x['age_of_business_months'] = pd.to_numeric(x['age_of_business_months'], errors='coerce')
        else:
            # If no age column, create a dummy
            x['age_of_business_months'] = 0
        
        # Ensure revenue is numeric
        if 'application_annual_revenue' in x.columns:
            x['application_annual_revenue'] = pd.to_numeric(x['application_annual_revenue'], errors='coerce')
        else:
            # If no revenue column, create a dummy
            x['application_annual_revenue'] = 0
        
        # Calculate average target values for each rule group
        vals = []
        for i, mask in enumerate(self._get_masks(x)):
            if mask.sum() > 0:  # Calculate mean if data exists
                vals.append(y[mask].mean())
            else:
                vals.append(0.0)  # Default value if no data
        
        self.vals_ = vals
        
        # Update rule_groups with actual data-driven probabilities
        for i, (rule, val) in enumerate(zip(self.rule_conditions, vals)):
            self.rule_groups.append({"prob": val, "desc": rule["desc"]})
        return self
    
    def predict(self, X):
        """Predict binary classification"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probability based on rule groups"""
        x = X.copy()
        
        # Ensure age is numeric (in months)
        if 'age_of_business_months' in x.columns:
            x['age_of_business_months'] = pd.to_numeric(x['age_of_business_months'], errors='coerce')
        else:
            x['age_of_business_months'] = 0
        
        # Ensure revenue is numeric
        if 'application_annual_revenue' in x.columns:
            x['application_annual_revenue'] = pd.to_numeric(x['application_annual_revenue'], errors='coerce')
        else:
            x['application_annual_revenue'] = 0
        
        # Apply rules and assign probabilities
        y_pos = np.zeros(len(x))
        for i, mask in enumerate(self._get_masks(x)):
            # Use calculated probabilities from training data
            if hasattr(self, 'vals_') and self.vals_ is not None and len(self.vals_) > i:
                y_pos[mask] = self.vals_[i]
            else:
                # Fallback to rule group probabilities (should not happen after training)
                y_pos[mask] = 0.0
        
        # Return 2D array like sklearn models: [[prob_class_0, prob_class_1], ...]
        y_neg = 1 - y_pos
        return np.column_stack([y_neg, y_pos])
    
    def _get_masks(self, x):
        """Define rule groups based on exact business specifications"""
        # Check if business legal structure is valid (Corporation or LLC)
        # Convert to numpy array to avoid pandas Series comparison issues
        valid_legal_structure = x['business_legal_structure'].str.lower().isin(['corporation', 'corp', 'limited liability company', 'llc']).values
        
        # Convert pandas Series to numpy arrays for boolean operations
        age = x['age_of_business_months'].values
        revenue = x['application_annual_revenue'].values
        
        return [
            # Group 1: Age < 12 months OR invalid legal structure - automatically assigned 0.0
            np.logical_or(age < 12, ~valid_legal_structure),
            
            # Group 2: Age 12-17 months AND Revenue >= 120,000 USD AND valid legal structure
            np.logical_and.reduce([age >= 12, age <= 17, revenue >= 120000, valid_legal_structure]),
            
            # Group 3: Age 12-17 months AND Revenue < 120,000 USD AND valid legal structure
            np.logical_and.reduce([age >= 12, age <= 17, revenue < 120000, valid_legal_structure]),
            
            # Group 4: Age >= 18 months AND Revenue >= 120,000 USD AND valid legal structure
            np.logical_and.reduce([age >= 18, revenue >= 120000, valid_legal_structure]),
            
            # Group 5: Age >= 18 months AND Revenue < 120,000 USD AND valid legal structure
            np.logical_and.reduce([age >= 18, revenue < 120000, valid_legal_structure])
        ]
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_categories(self, categories):
        """Set pre-calculated categories for consistent features across folds"""
        pass
        
    def get_rule_groups(self):
        """Returns the rule groups ordered by L2S value"""
        return self.rule_groups
    
    def calculate_ece(self, X, y_true):
        """Calculate ECE using Avocado's rule boundaries"""
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Define bin edges based on rule probabilities (sorted ascending)
        bin_edges = [0.0, 0.01, 0.032, 0.28, 0.492, 1.0]
        
        prob_true = []
        prob_pred = []
        bin_metrics = []
        
        # Calculate metrics for each bin
        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            
            # Get samples in this bin
            if i == 0:  # First bin is exactly 0.0
                bin_mask = (y_pred_proba == 0.0)
            elif i == len(bin_edges) - 2:  # Last bin includes the edge
                bin_mask = (y_pred_proba >= bin_start)
            else:  # Middle bins
                bin_mask = (y_pred_proba >= bin_start) & (y_pred_proba < bin_end)
            
            if np.any(bin_mask):
                bin_samples = np.sum(bin_mask)
                bin_sales = np.sum(y_true[bin_mask])
                bin_pred_prob = np.mean(y_pred_proba[bin_mask])
                bin_actual_l2s = bin_sales / bin_samples
                
                prob_true.append(bin_actual_l2s)
                prob_pred.append(bin_pred_prob)
                
                bin_metrics.append({
                    'bin_idx': i,
                    'bin_samples': int(bin_samples),
                    'bin_sales': int(bin_sales),
                    'bin_pred_prob': float(bin_pred_prob),
                    'bin_actual_l2s': float(bin_actual_l2s),
                    'l2s_comparison': float(abs(bin_pred_prob - bin_actual_l2s)),
                    'bin_range': f"{bin_start:.3f}-{bin_end:.3f}",
                    'rule_desc': self.rule_groups[i]['desc']
                })
        
        # Calculate ECE
        if prob_true and prob_pred:
            ece = np.mean(np.abs(np.array(prob_true) - np.array(prob_pred)))
        else:
            ece = 0.0
            
        return {
            'ece': ece,
            'prob_true': np.array(prob_true),
            'prob_pred': np.array(prob_pred),
            'bin_metrics': bin_metrics,
            'n_bins': len(bin_metrics)
        }
        


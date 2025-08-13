"""
Custom Transformers for Biz2Credit using ds_modeling framework
"""

import numpy as np
import pandas as pd
from ds_modeling.ml_framework.base import Transformer


class Biz2CreditPrep1(Transformer):
    """
    First preprocessing transformer for Biz2Credit
    Handles: p_sale creation, log transforms, null indicators
    """
    
    def _fit(self, x, y=None, sample_weight=None, **fit_params):
        """Learn preprocessing parameters"""
        self.impute_vals = {}
        
        # Store sample weights if provided (for downstream use)
        if sample_weight is not None:
            self.sample_weight = sample_weight
            print(f"Received sample weights: min={sample_weight.min()}, max={sample_weight.max()}, mean={sample_weight.mean():.2f}")
        
        # Learn imputation values for numeric features
        if 'age_of_business_months' in x.columns:
            self.impute_vals['age'] = self._impute_age(x['age_of_business_months'])
        
        if 'application_annual_revenue' in x.columns:
            self.impute_vals['revenue'] = self._impute_revenue(x['application_annual_revenue'])
            
        return self
    
    def _transform(self, x):
        """Apply preprocessing transformations"""
        x = x.copy()
        

        
        # Store target variable if it exists
        target_col = None
        if 'sales_count' in x.columns:
            target_col = x['sales_count'].copy()
        
        # 1. Create p_sale feature with simple fallback logic
        if 'normalized_p_cr_sale' in x.columns and 'normalized_p_cr_lead' in x.columns:
            # Case when: if normalized_p_cr_sale = 0.0 then p_cr_sale else normalized_p_cr_sale
            pcr_sale = np.where(x['normalized_p_cr_sale'] == 0.0, x['p_cr_sale'], x['normalized_p_cr_sale'])
            pcr_lead = np.where(x['normalized_p_cr_lead'] == 0.0, x['p_cr_lead'], x['normalized_p_cr_lead'])
            
            # Now calculate the ratio and clip
            x['p_sale'] = (pcr_sale / pcr_lead).clip(0, 1)
            x['p_sale'] = x['p_sale'].fillna(0.0001)
        
        # 2. Convert 0.0 to NaN for enrichment features (business logic)
        enrichment_features = ['age_of_business_months'] #  'application_annual_revenue'
        for feature in enrichment_features:
            if feature in x.columns:
                x[feature] = x[feature].replace(0.0, np.nan)
        
        # 3. Create log annual revenue
        if 'application_annual_revenue' in x.columns:
            x['log_annual_revenue'] = np.log1p(x['application_annual_revenue'])
            # Handle infinite values
            x['log_annual_revenue'] = x['log_annual_revenue'].replace([np.inf, -np.inf], np.nan)
        
        # 4. Create null indicators
        x['isnull_age'] = x['age_of_business_months'].isnull().astype(int)
        x['isnull_revenue'] = x['application_annual_revenue'].isnull().astype(int)
        x['isnull_legal'] = x['business_legal_structure'].isnull().astype(int)
        
        # 5. Create revenue tiers
        if 'log_annual_revenue' in x.columns:
            x['revenue_tier'] = self._create_revenue_tier(x['log_annual_revenue'])
        
        # 6. Create age groups
        if 'age_of_business_months' in x.columns:
            x['age_group'] = self._create_age_group(x['age_of_business_months'])
        
        # 7. Create interaction features
        if 'p_sale' in x.columns:
            x['p_sale_x_isnull_age'] = x['p_sale'] * x['isnull_age']
            x['p_sale_x_isnull_revenue'] = x['p_sale'] * x['isnull_revenue']
        
        # Restore target variable if it was present
        if target_col is not None:
            x['sales_count'] = target_col
        
        return x
    
    def _impute_age(self, age_series):
        """Impute age values using business logic"""
        # Convert months to years and impute
        age_years = age_series / 12
        return age_years.median()
    
    def _impute_revenue(self, revenue_series):
        """Impute revenue values using business logic"""
        return revenue_series.median()
    
    def _create_revenue_tier(self, log_revenue):
        """Create revenue tiers with business logic"""
        def tier_function(val):
            if pd.isna(val):
                return 'Low'  # Null revenue -> Low tier (low_L2S group)
            elif val < 11.5:  # ~$100k
                return 'Low'
            elif val < 12.5:  # ~$250k
                return 'Medium'
            elif val < 13.5:  # ~$750k
                return 'High'
            else:
                return 'Premium'
        
        return log_revenue.apply(tier_function)
    
    def _create_age_group(self, age_months):
        """Create age groups with business logic"""
        def age_function(val):
            if pd.isna(val):
                return 'Young'  # Null age -> Young group (1-2 bins as you mentioned)
            elif val < 12:  # < 1 year
                return 'New'
            elif val < 36:  # 1-3 years
                return 'Young'
            elif val < 60:  # 3-5 years
                return 'Expected'
            else:
                return 'Mature'
        
        return age_months.apply(age_function)


class Biz2CreditPrep2(Transformer):
    """
    Second preprocessing transformer for Biz2Credit
    Handles: null handling, feature selection, final cleanup
    """
    
    def _fit(self, x, y=None, sample_weight=None, **fit_params):
        """Learn preprocessing parameters"""
        # Define features to keep (both original and engineered)
        self.features_to_keep = [
            'p_sale', 'age_of_business_months', 'log_annual_revenue',
            'business_legal_structure', 'revenue_tier', 'age_group',
            'isnull_age', 'isnull_revenue', 'isnull_legal',
            'p_sale_x_isnull_age', 'p_sale_x_isnull_revenue'
        ]
        
        # Only keep features that exist in the data
        self.features_to_keep = [f for f in self.features_to_keep if f in x.columns]
        
        # Only keep essential features - don't add all numeric/categorical automatically
        # This prevents processing irrelevant fields
        essential_features = [
            'p_sale', 'age_of_business_months', 'log_annual_revenue',
            'business_legal_structure', 'revenue_tier', 'age_group',
            'isnull_age', 'isnull_revenue', 'isnull_legal',
            'p_sale_x_isnull_age', 'p_sale_x_isnull_revenue'
        ]
        
        # Only add features that exist in the data
        self.features_to_keep = [f for f in essential_features if f in x.columns]
        
        # Features to keep are defined in self.features_to_keep
        
        return self
    
    def _transform(self, x):
        """Apply final preprocessing"""
        x = x.copy()
        
        # Store target variable if it exists
        target_col = None
        if 'sales_count' in x.columns:
            target_col = x['sales_count'].copy()
        
        # Keep only the features we need (don't remove rows here)
        available_features = [f for f in self.features_to_keep if f in x.columns]
        x = x[available_features]
        
        # Restore target variable if it was present
        if target_col is not None:
            x['sales_count'] = target_col
        
        return x


class Biz2CreditImputer(Transformer):
    """
    Smart imputer for Biz2Credit with business logic
    """
    
    def _fit(self, x, y=None, sample_weight=None, **fit_params):
        """Learn imputation strategies with business logic"""
        self.impute_strategies = {}
        
        # Business logic for specific features based on your analysis
        if 'age_of_business_months' in x.columns:
            # Age: First bin into groups, then add null ages to 1-2 bins
            # Use median of young businesses (1-3 years = 12-36 months)
            young_age_mask = (x['age_of_business_months'] >= 12) & (x['age_of_business_months'] <= 36)
            if young_age_mask.any():
                age_median = x.loc[young_age_mask, 'age_of_business_months'].median()
            else:
                age_median = 24  # Default to 2 years if no young businesses
            self.impute_strategies['age_of_business_months'] = age_median
        
        if 'application_annual_revenue' in x.columns:
            # Revenue: First bin into groups, then impute nulls with low_L2S values
            # Use median of low revenue businesses (< $100K = log_revenue < 11.5)
            low_revenue_mask = x['application_annual_revenue'] < 100000
            if low_revenue_mask.any():
                revenue_median = x.loc[low_revenue_mask, 'application_annual_revenue'].median()
            else:
                revenue_median = 50000  # Default to $50K if no low revenue businesses
            self.impute_strategies['application_annual_revenue'] = revenue_median
        
        if 'business_legal_structure' in x.columns:
            # Legal structure: Use business logic (closer to low_L2S)
            # Map nulls to 'Limited Liability Company' (most common, business-friendly)
            self.impute_strategies['business_legal_structure'] = 'Limited Liability Company'
        
        # For other numeric features, use median
        numeric_features = x.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature not in self.impute_strategies and feature in x.columns:
                self.impute_strategies[feature] = x[feature].median()
        
        # For other categorical features, use 'Unknown'
        categorical_features = x.select_dtypes(include=['object', 'category']).columns
        for feature in categorical_features:
            if feature not in self.impute_strategies and feature in x.columns:
                self.impute_strategies[feature] = 'Unknown'
        
        return self
    
    def _transform(self, x):
        """Apply imputation"""
        x = x.copy()
        
        # Store target variable if it exists
        target_col = None
        if 'sales_count' in x.columns:
            target_col = x['sales_count'].copy()
        
        for feature, strategy in self.impute_strategies.items():
            if feature in x.columns:
                if isinstance(strategy, str):
                    x[feature] = x[feature].fillna(strategy)
                else:
                    x[feature] = x[feature].fillna(strategy)
        
        # Restore target variable if it was present
        if target_col is not None:
            x['sales_count'] = target_col
        
        return x


class Biz2CreditCategoricalEncoder(Transformer):
    """
    Encodes categorical features for machine learning
    """
    
    def _fit(self, x, y=None, sample_weight=None, **fit_params):
        """Learn encoding strategies"""
        self.categorical_features = x.select_dtypes(include=['object', 'category']).columns.tolist()
        self.encoding_maps = {}
        
        # Create simple label encoding for each categorical feature
        for feature in self.categorical_features:
            if feature in x.columns:
                unique_values = x[feature].dropna().unique()
                # Create a mapping from category to integer
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                # Add 'Unknown' if it's not in the mapping
                if 'Unknown' not in encoding_map:
                    encoding_map['Unknown'] = len(encoding_map)
                self.encoding_maps[feature] = encoding_map
        
        return self
    
    def _transform(self, x):
        """Apply categorical encoding"""
        x = x.copy()
        
        # Store target variable if it exists
        target_col = None
        if 'sales_count' in x.columns:
            target_col = x['sales_count'].copy()
        
        for feature, encoding_map in self.encoding_maps.items():
            if feature in x.columns:
                # Replace categories with integers
                x[feature] = x[feature].map(encoding_map)
                # Fill any remaining NaN values with the 'Unknown' code
                x[feature] = x[feature].fillna(encoding_map.get('Unknown', -1))
                # Convert to numeric type
                x[feature] = pd.to_numeric(x[feature], errors='coerce')
        
        # Restore target variable if it was present
        if target_col is not None:
            x['sales_count'] = target_col
        
        return x


class BaselinePredictor:
    """
    Baseline predictor that just returns p_sale values as predictions
    This serves as a baseline comparison for ML models
    """
    
    def fit(self, X, y=None):
        """Fit method - no training needed for baseline"""
        return self
    
    def predict(self, X):
        """Return p_sale values as predictions"""
        if 'p_sale' in X.columns:
            return X['p_sale'].values
        else:
            # Fallback to zeros if p_sale not available
            return np.zeros(len(X))
    
    def predict_proba(self, X):
        """Return p_sale as probability for class 1"""
        if 'p_sale' in X.columns:
            p_sale = X['p_sale'].values
            # Return [1-p_sale, p_sale] for binary classification
            return np.column_stack([1 - p_sale, p_sale])
        else:
            # Fallback to 50/50 probabilities
            n_samples = len(X)
            return np.column_stack([np.ones(n_samples) * 0.5, np.ones(n_samples) * 0.5])


# Factory function to create Biz2Credit preprocessing pipeline
def create_biz2credit_preprocessing():
    """Create the complete Biz2Credit preprocessing pipeline"""
    return Biz2CreditPrep1(), Biz2CreditPrep2(), Biz2CreditImputer(), Biz2CreditCategoricalEncoder()


from ds_modeling.transformers.categorization_transformers import ExtremeOccurrenceGrouper
from ds_modeling.transformers.imputers_transformers import SimpleImputer , AddNaNIndicator
from ds_modeling.ml_framework.base import Transformer
from ds_modeling.ml_framework.pipeline import make_pipeline
from ds_modeling.ml_framework.prdictors_wrappers.classifiers.linear import SkLogisticRegression


class FeaturesAdder(Transformer):
    def _fit(self, x, y=None, weights=None):
        ...

    def _transform(self, x):
        x = x.copy()
        x['age_of_business_years'] = x['age_of_business_months'] // 12
        x['application_annual_revenue_100'] = x.application_annual_revenue // 100000
        x['is_age_0'] = (x['age_of_business_years'] == 0).astype(int)
        x['is_business_Corporation'] = (x['business_legal_structure'] == 'Corporation').astype(int)
        x['is_business_sole_Ppoprietorship'] = (x['business_legal_structure'] == 'Sole Proprietorship').astype(int)
        x['p_l2s'] = ((x['normalized_p_cr_sale'] / x['normalized_p_cr_lead']))
        max_pl2s = x['p_l2s'][np.isfinite(x['p_l2s'])].max()
        min_pl2s = x['p_l2s'][np.isfinite(x['p_l2s'])].min()
        x['p_l2s'].clip(lower=min_pl2s, upper=max_pl2s, inplace=True)
        x['p_l2s'].fillna(min_pl2s, inplace=True)
        return x[['normalized_p_cr_lead', 'normalized_p_cr_sale', 'p_l2s', 'age_of_business_years',
                  'application_annual_revenue_100',
                  'business_legal_structure_is_na', 'age_of_business_months_is_na',
                  'application_annual_revenue_is_na', 'is_age_0',
                  'is_business_Corporation', 'is_business_sole_Ppoprietorship'
                ]]
        


def build_pip1e():
    nans = AddNaNIndicator(cols=['business_legal_structure'])
    imputer = SimpleImputer(strategy='median', cols=['age_of_business_months', 'application_annual_revenue'], add_indicator=True)
    

    return make_pipeline(nans, imputer, FeaturesAdder(), SkLogisticRegression(C=100))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, log_loss, r2_score, mean_squared_error
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')


def calculate_ece_by_leads(
    y_true,
    y_pred_prob,
    leads_count,
    n_bins: int = 10,
    print_details: bool = False,
    model_name: str = "Model"
) -> float:
    """
    Expected‑calibration‑error (ECE) weighted by *total* leads in each bin.

    • y_true : array‑like of ints
        Observed sales per row (not a rate)
    • y_pred_prob   : array‑like of floats
        Predicted probability of sale (0‑1)
    • leads_count   : array‑like of ints
        Number of leads represented by each row
    """
    df = pd.DataFrame(
        {"sales": y_true, "pred": y_pred_prob, "leads": leads_count}
    ).dropna(subset=["leads"])
    print('dbug leads in ece: ', df.leads.value_counts())

    # Guard against zero‑lead rows
    df = df[df["leads"] > 0].copy()
    if df.empty:
        print('empty df in ece by leads error!!@!@#@$@$')
        return np.nan

    # Actual rate
    df["actual"] = df["sales"] / df["leads"]

    # -------- binning: sort by predicted probability, then bin by equal total leads -------- #
    df = df.sort_values("y_pred_prob" if "y_pred_prob" in df.columns else "pred", ascending=True).reset_index(drop=True)
    df["cum_leads"] = df["leads"].cumsum()
    total_leads = df["leads"].sum()
    leads_per_bin = total_leads / n_bins

    # Assign bin numbers so that each bin has (approximately) equal total leads
    bin_edges = []
    current_leads = 0
    current_bin = 0
    bin_assignments = np.zeros(len(df), dtype=int)
    for idx, leads in enumerate(df["leads"]):
        if current_leads >= (current_bin + 1) * leads_per_bin and current_bin < n_bins - 1:
            current_bin += 1
        bin_assignments[idx] = current_bin
        current_leads += leads
    df["bin"] = bin_assignments

    if print_details:
        print(f"\n=== ECE ANALYSIS BY LEADS: {model_name} ===")
        print(
            f"{'Bin':<3} {'Rows':<6} {'Leads':<10} "
            f"{'Mean_Pred':<10} {'Mean_Act':<10} {'Abs_Diff':<9} {'Weight':<8}"
        )
        print("-" * 70)

    ece = 0.0

    for b in range(n_bins):
        bin_df = df[df["bin"] == b]
        if bin_df.empty:
            continue

        weight = bin_df["leads"].sum() / total_leads
        diff = abs(bin_df["pred"].mean() - bin_df["actual"].mean())
        ece += weight * diff

        if print_details:
            print(
                f"{b:<3} {len(bin_df):<6} "
                f"{bin_df['leads'].sum():<10.0f} "
                f"{bin_df['pred'].mean():<10.4f} "
                f"{bin_df['actual'].mean():<10.4f} "
                f"{diff:<9.4f} {weight:<8.4f}"
            )

    if print_details:
        print(f"\nTotal ECE: {ece:.4f}")

    return ece

def create_chart_df(models_dict):
    """
    Create a DataFrame for plotting model metrics.
    """
    df = pd.DataFrame()
    for name, metrics in models_dict.items():
        df.loc[name, 'AUC'] = metrics.get('AUC', np.nan)
        df.loc[name, 'R2'] = metrics.get('R2', np.nan)
        df.loc[name, 'RMSE'] = metrics.get('RMSE', np.nan)
        df.loc[name, 'LogLoss'] = metrics.get('LogLoss', np.nan)
        df.loc[name, 'ECE'] = metrics.get('ECE', np.nan)
    return df

def display_model_charts(results_dict):
    """
    Display comparison charts for base and tuned models.
    Expects a dictionary with keys 'base' and/or 'tuned', each mapping to a dict of model results.
    """
    import matplotlib.pyplot as plt

    base_models = results_dict.get('base', None)
    tuned_models = results_dict.get('tuned', None)

    # Plot base models
    if base_models:
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        fig.suptitle('BASE MODELS COMPARISON', fontsize=16, fontweight='bold')
        
        base_df = create_chart_df(base_models)
        
        # AUC (higher is better)
        base_df['AUC'].plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_title('ROC AUC (Higher is Better)')
        axes[0].set_ylabel('AUC')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R2 (higher is better)
        base_df['R2'].plot(kind='bar', ax=axes[1], color='seagreen', alpha=0.7)
        axes[1].set_title('R2 (Higher is Better)')
        axes[1].set_ylabel('R2')
        axes[1].tick_params(axis='x', rotation=45)
        
        # RMSE (lower is better)
        base_df['RMSE'].plot(kind='bar', ax=axes[2], color='purple', alpha=0.7)
        axes[2].set_title('RMSE (Lower is Better)')
        axes[2].set_ylabel('RMSE')
        axes[2].tick_params(axis='x', rotation=45)
        
        # LogLoss (lower is better)
        base_df['LogLoss'].plot(kind='bar', ax=axes[3], color='coral', alpha=0.7)
        axes[3].set_title('LogLoss (Lower is Better)')
        axes[3].set_ylabel('LogLoss')
        axes[3].tick_params(axis='x', rotation=45)
        
        # ECE (lower is better)
        base_df['ECE'].plot(kind='bar', ax=axes[4], color='gold', alpha=0.7)
        axes[4].set_title('ECE (Lower is Better)')
        axes[4].set_ylabel('ECE')
        axes[4].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    # Plot tuned models
    if tuned_models:
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        fig.suptitle('TUNED MODELS COMPARISON', fontsize=16, fontweight='bold')
        
        tuned_df = create_chart_df(tuned_models)
        
        # AUC (higher is better)
        tuned_df['AUC'].plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_title('ROC AUC (Higher is Better)')
        axes[0].set_ylabel('AUC')
        axes[0].tick_params(axis='x', rotation=45)
        
        # R2 (higher is better)
        tuned_df['R2'].plot(kind='bar', ax=axes[1], color='seagreen', alpha=0.7)
        axes[1].set_title('R2 (Higher is Better)')
        axes[1].set_ylabel('R2')
        axes[1].tick_params(axis='x', rotation=45)
        
        # RMSE (lower is better)
        tuned_df['RMSE'].plot(kind='bar', ax=axes[2], color='purple', alpha=0.7)
        axes[2].set_title('RMSE (Lower is Better)')
        axes[2].set_ylabel('RMSE')
        axes[2].tick_params(axis='x', rotation=45)
        
        # LogLoss (lower is better)
        tuned_df['LogLoss'].plot(kind='bar', ax=axes[3], color='coral', alpha=0.7)
        axes[3].set_title('LogLoss (Lower is Better)')
        axes[3].set_ylabel('LogLoss')
        axes[3].tick_params(axis='x', rotation=45)
        
        # ECE (lower is better)
        tuned_df['ECE'].plot(kind='bar', ax=axes[4], color='gold', alpha=0.7)
        axes[4].set_title('ECE (Lower is Better)')
        axes[4].set_ylabel('ECE')
        axes[4].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

class GenericMLPipeline:
    """
    Generic ML Pipeline for different partner datasets (Classification version)
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_pipeline = None
        self.preprocessor = None

        # Define standard classification models
        self.models = {
            'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_state),
            'RidgeClassifier': RidgeClassifier(random_state=random_state),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=random_state, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, random_state=random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=random_state, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'
            )
        }

        # Define hyperparameter grids for classification
        self.param_grids = {
            'RandomForest': {
                'classifier__n_estimators': randint(100, 500),
                'classifier__max_depth': randint(3, 20),
                'classifier__min_samples_split': randint(2, 20),
                'classifier__min_samples_leaf': randint(1, 10),
                'classifier__max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'classifier__n_estimators': randint(100, 400),
                'classifier__max_depth': randint(3, 10),
                'classifier__learning_rate': uniform(0.01, 0.2),
                'classifier__subsample': uniform(0.6, 0.4),
                'classifier__colsample_bytree': uniform(0.6, 0.4)
            },
            'GradientBoosting': {
                'classifier__n_estimators': randint(100, 400),
                'classifier__max_depth': randint(3, 10),
                'classifier__learning_rate': uniform(0.01, 0.2),
                'classifier__subsample': uniform(0.6, 0.4)
            }
        }

    def print_data_stats(self, df, target_column, enrichment_features=None, stage=""):
        """Print data statistics"""
        print(f"\n=== DATA STATISTICS {stage} ===")
        print(f"Total rows: {len(df):,}")

        if target_column in df.columns:
            print(f"{target_column} distribution:")
            print(df[target_column].value_counts())

        # Count leads (assuming each row is a lead)
        print(f"Total leads: {len(df):,}")

        # Show enrichment feature availability
        if enrichment_features:
            print(f"Enrichment features availability:")
            for feature in enrichment_features:
                if feature in df.columns:
                    non_null = df[feature].notna().sum()
                    print(f"  {feature}: {non_null:,}/{len(df):,} ({non_null/len(df)*100:.1f}%)")

    def handle_enrichment_nulls(self, df, enrichment_features, target_column):
        """
        Remove rows where ALL enrichment features are null.
        """
        print(f"\n=== ENRICHMENT NULL HANDLING ===")
        if not enrichment_features or len(enrichment_features) == 0:
            print("No enrichment features specified; removing all rows.")
            return df.iloc[0:0].copy()
        enrichment_cols = [col for col in enrichment_features if col in df.columns]
        all_enrichment_null = df[enrichment_cols].isnull().all(axis=1)
        rows_to_drop = all_enrichment_null.sum()
        print(f"Rows where ALL enrichment features are null: {rows_to_drop:,}")
        print(f"Rows where at least one enrichment feature is available: {len(df) - rows_to_drop:,}")
        df_clean = df[~all_enrichment_null].copy()
        print(f"Shape after removing rows with all enrichment features null: {df_clean.shape}")
        return df_clean

    def handle_nulls(self, df, feature_columns, null_handling_config):
        """
        Handle nulls based on configuration.
        """
        df_clean = df.copy()
        print(f"\n=== FEATURE-SPECIFIC NULL HANDLING ===")
        for column, strategy in null_handling_config.items():
            if column not in df_clean.columns:
                print(f"Column {column} not found in DataFrame - continuing...")
                continue
            null_count_before = df_clean[column].isnull().sum()
            if null_count_before == 0:
                continue
            print(f"{column}: {null_count_before:,} nulls -> ", end="")
            if strategy == 'median':
                df_clean[column].fillna(df_clean[column].median(), inplace=True)
            elif strategy == 'mean':
                df_clean[column].fillna(df_clean[column].mean(), inplace=True)
            elif strategy == 'mode':
                df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
            elif strategy == 'unknown':
                df_clean[column].fillna('Unknown', inplace=True)
            elif strategy == 'drop_rows':
                df_clean = df_clean.dropna(subset=[column])
            elif strategy == 'drop_columns':
                df_clean = df_clean.drop(columns=[column])
                if column in feature_columns:
                    feature_columns.remove(column)
            null_count_after = df_clean[column].isnull().sum() if column in df_clean.columns else 0
            print(f"{null_count_after:,} nulls (strategy: {strategy})")
        return df_clean, feature_columns

    def create_preprocessor(self, X, preprocessing_config=None, nulls_strategy='median'):
        """
        Create preprocessing pipeline with configurable strategies.
        """
        if preprocessing_config is None:
            preprocessing_config = {}
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        transformers = []
        if numeric_features:
            transformers.append((
                'num', Pipeline([
                    ('imputer', SimpleImputer(strategy=nulls_strategy)),
                    ('scaler', RobustScaler())
                ]), numeric_features
            ))
        for cat_feature in categorical_features:
            strategy = preprocessing_config.get(cat_feature, 'onehot')
            if strategy == 'onehot':
                transformers.append((
                    f'cat_{cat_feature}', Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), [cat_feature]
                ))
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        return self.preprocessor

    def calculate_extended_metrics(self,
                                  y_true,
                                  y_pred,
                                  leads_count,
                                  y_pred_proba=None,
                                  **kwargs):
        """
        y_true      = sales_count (binary outcome (0/1))
        y_pred      = predicted class (0/1)
        leads_count = arraylike
        y_pred_proba = predicted probability for class 1
        """

        # Probability-based metrics
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            auc = np.nan
        try:
            ll = log_loss(y_true, y_pred_proba)
        except Exception:
            ll = np.nan
        try:
            r2 = r2_score(y_true, y_pred_proba)
        except Exception:
            r2 = np.nan
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_proba))
        except Exception:
            rmse = np.nan

        # ECE: use predicted probability for class 1
        ece = calculate_ece_by_leads(
            y_true=y_true,
            y_pred_prob=y_pred_proba,
            leads_count=leads_count,
            n_bins=10,
            print_details=False,
        )

        return {
            "AUC": auc,
            "R2": r2,
            "RMSE": rmse,
            "LogLoss": ll,
            "ECE": ece
        }

    def train_models(self, X_train, X_test, y_train, y_test, leads_test=None):
        """Train all classification models and evaluate with extended metrics"""
        print("\n=== TRAINING CLASSIFICATION MODELS ===")
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                # Some classifiers (e.g., RidgeClassifier) do not have predict_proba
                # Use decision_function and apply sigmoid
                if hasattr(pipeline.named_steps['classifier'], "decision_function"):
                    from scipy.special import expit
                    y_pred_proba = expit(pipeline.decision_function(X_test))
                else:
                    y_pred_proba = y_pred  # fallback, not ideal
            metrics = self.calculate_extended_metrics(
                y_test, y_pred, leads_count=leads_test, y_pred_proba=y_pred_proba
            )
            metrics['pipeline'] = pipeline
            self.results[name] = metrics
            print(f"{name} - AUC: {metrics['AUC']:.4f}, R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, LogLoss: {metrics['LogLoss']:.4f}")
            if 'ECE' in metrics and not np.isnan(metrics['ECE']):
                print(f"         ECE: {metrics['ECE']:.4f}")
        return self.results

    def hyperparameter_tuning(self, X_train, y_train, models_to_tune=None, n_iter=20, cv=3):
        """Perform hyperparameter tuning for specified classification models"""
        if models_to_tune is None:
            models_to_tune = ['RandomForest', 'XGBoost', 'GradientBoosting']
        print("\n=== HYPERPARAMETER TUNING ===")
        tuned_results = {}
        for model_name in models_to_tune:
            if model_name not in self.param_grids:
                print(f"No parameter grid defined for {model_name}, skipping...")
                continue
            print(f"\n--- Tuning {model_name} ---")
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', self.models[model_name])
            ])
            search = RandomizedSearchCV(
                pipeline,
                self.param_grids[model_name],
                n_iter=n_iter,
                scoring='roc_auc',
                cv=cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            search.fit(X_train, y_train)
            tuned_results[f'{model_name}_Tuned'] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'pipeline': search.best_estimator_
            }
            print(f"Best {model_name} params: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        return tuned_results

    def evaluate_tuned_models(self, tuned_results, X_test, y_test, leads_test=None):
        """Evaluate tuned models on test set with extended metrics"""
        print("\n=== EVALUATING TUNED MODELS ===")
        for name, result in tuned_results.items():
            pipeline = result['pipeline']
            y_pred = pipeline.predict(X_test)
            if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            else:
                if hasattr(pipeline.named_steps['classifier'], "decision_function"):
                    from scipy.special import expit
                    y_pred_proba = expit(pipeline.decision_function(X_test))
                else:
                    y_pred_proba = y_pred
            metrics = self.calculate_extended_metrics(
                y_test, y_pred, leads_count=leads_test, y_pred_proba=y_pred_proba
            )
            metrics['pipeline'] = pipeline
            self.results[name] = metrics
            print(f"{name} - AUC: {metrics['AUC']:.4f}, R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, LogLoss: {metrics['LogLoss']:.4f}")
            if 'ECE' in metrics and not np.isnan(metrics['ECE']):
                print(f"         ECE: {metrics['ECE']:.4f}")

    def get_best_model(self, metric='AUC', ascending=False):
        """Get the best model based on specified metric (default: AUC)"""
        if not self.results:
            print("No models trained yet!")
            return None
        comparison_df = pd.DataFrame()
        for name, metrics in self.results.items():
            comparison_df.loc[name, 'AUC'] = metrics.get('AUC', np.nan)
            comparison_df.loc[name, 'R2'] = metrics.get('R2', np.nan)
            comparison_df.loc[name, 'RMSE'] = metrics.get('RMSE', np.nan)
            if 'ECE' in metrics:
                comparison_df.loc[name, 'ECE'] = metrics['ECE']
            if 'LogLoss' in metrics:
                comparison_df.loc[name, 'LogLoss'] = metrics['LogLoss']
        comparison_df = comparison_df.sort_values(metric, ascending=ascending)
        print(f"\n=== MODEL COMPARISON (sorted by {metric}) ===")
        print(comparison_df.round(4))
        best_model_name = comparison_df.index[0]
        self.best_model = best_model_name
        self.best_pipeline = self.results[best_model_name]['pipeline']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Best {metric}: {comparison_df.loc[best_model_name, metric]:.4f}")
        display_model_charts(self.results)
        return self.best_pipeline, best_model_name

    def split_data(self, df, feature_columns, target_column, split_method='by_ratio', split_val=0.2, date_column=None):
        """
        Split the data into train and test sets based on the split_method.
        Returns: X_train, X_test, y_train, y_test, leads_test
        """
        features_no_leads = [c for c in feature_columns if c != 'leads_count']
        if split_method == 'by_date':
            if date_column is None:
                raise ValueError("date_column must be provided when split_method='by_date'")
            print(f"\n=== DATE-BASED SPLIT (cutoff: {split_val}) ===")
            train_df = df[df[date_column] < split_val]
            test_df = df[df[date_column] >= split_val]
            X_train = train_df[features_no_leads].copy()
            y_train = train_df[target_column]
            X_test = test_df[features_no_leads].copy()
            y_test = test_df[target_column]
            leads_train = train_df['leads_count'] if 'leads_count' in train_df.columns else None
            leads_test = test_df['leads_count'] if 'leads_count' in test_df.columns else None
        else:
            X = df[features_no_leads]
            Y = df[target_column]
            if 'leads_count' in df.columns:
                leads = df['leads_count']
                X_train, X_test, y_train, y_test, _, leads_test = train_test_split(
                    X, Y, leads, test_size=split_val, random_state=self.random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=split_val, random_state=self.random_state
                )
                leads_test = None
        return X_train, X_test, y_train, y_test, leads_test

    def run_full_pipeline(self, df, feature_columns, target_column, 
                         enrichment_features=None, null_handling_config=None, 
                         preprocessing_config=None, 
                         split_method='by_ratio', split_val=0.2, date_column=None, 
                         metric_for_selection='AUC'):
        """
        Run the complete ML pipeline with all improvements.
        ECE is calculated using predicted probability vs actual, binned by leads_count (if present in features).
        """
        print("=== STARTING GENERIC ML PIPELINE (CLASSIFICATION) ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {feature_columns}")
        print(f"Target: {target_column}")
        self.print_data_stats(df, target_column, enrichment_features, "BEFORE PROCESSING")
        if enrichment_features:
            df_clean = self.handle_enrichment_nulls(df, enrichment_features, target_column)
        else:
            df_clean = df.copy()
        if null_handling_config:
            df_clean, feature_columns = self.handle_nulls(df_clean, feature_columns, null_handling_config)
        self.print_data_stats(df_clean, target_column, enrichment_features, "AFTER NULL HANDLING")
        X_train, X_test, y_train, y_test, leads_test = self.split_data(
            df_clean, feature_columns, target_column, 
            split_method=split_method, split_val=split_val, date_column=date_column
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        self.create_preprocessor(X_train, preprocessing_config, nulls_strategy='median')
        self.train_models(X_train, X_test, y_train, y_test, leads_test)
        tuned_results = self.hyperparameter_tuning(X_train, y_train)
        if tuned_results:
            self.evaluate_tuned_models(tuned_results, X_test, y_test, leads_test)
        best_pipeline, best_name = self.get_best_model(metric=metric_for_selection)
        return {
            'best_pipeline': best_pipeline,
            'best_model_name': best_name,
            'all_results': self.results,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

def run_partner_analysis(partner_name, df, feature_columns, target_column, 
                        enrichment_features=None, null_handling_config=None,
                        preprocessing_config=None, 
                        split_method='by_ratio', split_val=0.2, date_column=None):
    """
    Wrapper function to run analysis for any partner.
    ECE is automatically calculated if 'leads_count' is present in features.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSIS FOR {partner_name.upper()}")
    print(f"{'='*60}")
    pipeline = GenericMLPipeline()
    results = pipeline.run_full_pipeline(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        enrichment_features=enrichment_features,
        null_handling_config=null_handling_config,
        preprocessing_config=preprocessing_config,
        split_method=split_method,
        split_val=split_val,
        date_column=date_column
    )
    return results, pipeline
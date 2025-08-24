from xml.sax.handler import all_features
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib

# Add path for local imports
sys.path.append(os.path.abspath('../'))

# Import custom modules
from src.data_loader import DataLoader
from src.data_analysis import InitialDataAnalysis

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .churn-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
        color: #721c24;
    }
    .low-risk {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.25rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .feature-input {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def make_subplots():
    raise NotImplementedError

class ChurnDashboard:
    def __init__(self):
        self.data_loader = None
        self.analyzer = None
        self.datasets = None
        self.merged_df = None
        self.org_features = None
        self.ml_models = None
        self.pca_model = None
        self.scaler = None
        self.feature_importance = None
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'datasets' not in st.session_state:
            st.session_state.datasets = None
        if 'merged_df' not in st.session_state:
            st.session_state.merged_df = None
        if 'org_features' not in st.session_state:
            st.session_state.org_features = None
        if 'ml_models' not in st.session_state:
            st.session_state.ml_models = None
        if 'preprocessed_data' not in st.session_state:
            st.session_state.preprocessed_data = None
        if 'pca_results' not in st.session_state:
            st.session_state.pca_results = None
        if 'feature_selection_results' not in st.session_state:
            st.session_state.feature_selection_results = None

    def load_data(self, data_directory="resources"):
        """Load data using DataLoader"""
        try:
            with st.spinner("Loading datasets..."):
                self.data_loader = DataLoader(data_dir=data_directory)
                datasets = self.data_loader.load_all_datasets()
                
                # Store in session state
                st.session_state.datasets = datasets
                st.session_state.data_loaded = True
                
                return datasets
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def run_analysis(self):
        """Run initial data analysis"""
        try:
            with st.spinner("Running churn analysis..."):
                self.analyzer = InitialDataAnalysis(
                    datasets=st.session_state.datasets,
                    output_dir="streamlit_analysis_output"
                )
                
                # Create datasets
                merged_df, org_features = self.analyzer.create_datasets()
                
                # Store in session state
                st.session_state.merged_df = merged_df
                st.session_state.org_features = org_features
                st.session_state.analysis_complete = True
                
                return merged_df, org_features
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            return None, None

    def handle_missing_data(self, df, numeric_strategy='mean', categorical_strategy='mode', 
                           datetime_strategy='drop', verbose=True):
        """
        Fill or drop missing values in a DataFrame by column type.
        """
        df = df.copy()
        
        if verbose:
            missing_before = df.isnull().sum()
            missing_before = missing_before[missing_before > 0]
            if len(missing_before) > 0:
                st.write("**Missing values before preprocessing:**")
                st.dataframe(missing_before.to_frame('Missing Count'))
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif numeric_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif numeric_strategy == 'zero':
                    df[col] = df[col].fillna(0)
                elif numeric_strategy == 'drop':
                    df = df.dropna(subset=[col])
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_strategy == 'mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                elif categorical_strategy == 'constant':
                    df[col] = df[col].fillna('missing')
                elif categorical_strategy == 'drop':
                    df = df.dropna(subset=[col])
        
        # Datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        for col in datetime_cols:
            if df[col].isnull().any():
                if datetime_strategy == 'constant':
                    df[col] = df[col].fillna(pd.Timestamp('1970-01-01'))
                elif datetime_strategy == 'drop':
                    df = df.dropna(subset=[col])
        
        if verbose:
            missing_after = df.isnull().sum()
            missing_after = missing_after[missing_after > 0]
            if len(missing_after) > 0:
                st.write("**Missing values after preprocessing:**")
                st.dataframe(missing_after.to_frame('Missing Count'))
            else:
                st.success("✅ No missing values remaining!")
        
        return df

    def preprocess_for_ml(self):
        """Preprocess data for machine learning"""
        if not st.session_state.analysis_complete:
            st.error("Please complete the churn analysis first.")
            return None
        
        try:
            # Start with merged data
            df_ml = st.session_state.merged_df.copy()
            
            # Get churned organizations
            churned_org = set(st.session_state.datasets['churn']['masked_organisation_id'])
            
            # Handle missing data
            df_ml = self.handle_missing_data(df_ml,
                                           numeric_strategy='mean',
                                           categorical_strategy='mode',
                                           datetime_strategy='drop')
            
            # Create churn label
            df_ml['is_churned'] = df_ml['masked_organisation_id'].isin(churned_org).astype(int)
            
            # Process datetime columns
            ref_date = df_ml['report_date'].max()
            datetime_cols = [
                'report_date', 'create_date', 'current_period_end_datetime', 
                'current_period_start_datetime', 'renewal_date', 'trial_date',
                'chargify_trial_start_datetime', 'chargify_trial_end_datetime',
                'chargify_trial_start_date', 'chargify_trial_end_date', 
                'paid_subscription_start_date', 'next_assessment_at', 'activated_at'
            ]
            
            # Convert to datetime and create duration features
            for col in datetime_cols:
                if col in df_ml.columns:
                    df_ml[col] = pd.to_datetime(df_ml[col], errors='coerce')
            
            # Calculate duration features
            if 'chargify_trial_start_datetime' in df_ml.columns and 'chargify_trial_end_datetime' in df_ml.columns:
                df_ml['trial_duration_days'] = (df_ml['chargify_trial_end_datetime'] - df_ml['chargify_trial_start_datetime']).dt.days
            
            if 'current_period_start_datetime' in df_ml.columns and 'current_period_end_datetime' in df_ml.columns:
                df_ml['subscription_period_days'] = (df_ml['current_period_end_datetime'] - df_ml['current_period_start_datetime']).dt.days
            
            # Create "days since" features
            date_cols = [
                'create_date', 'renewal_date', 'trial_date',
                'chargify_trial_end_datetime', 'paid_subscription_start_date',
                'activated_at', 'next_assessment_at'
            ]
            
            for col in date_cols:
                if col in df_ml.columns:
                    df_ml[f'{col}_days_since'] = (ref_date - df_ml[col]).dt.days
            
            # Aggregate by organization
            group_col = 'masked_organisation_id'
            labels = df_ml.groupby(group_col)['is_churned'].max()
            
            # Separate column types
            numeric_cols = df_ml.select_dtypes(include='number').drop(columns=['is_churned'], errors='ignore').columns
            categorical_cols = df_ml.select_dtypes(include=['object', 'category']).columns.tolist()
            bool_cols = df_ml.select_dtypes(include='bool').columns
            
            # Remove group column from categorical
            if 'masked_organisation_id' in categorical_cols:
                categorical_cols.remove('masked_organisation_id')
            
            # Aggregation
            agg_numeric = df_ml.groupby(group_col)[numeric_cols].agg(['mean', 'std', 'min', 'max', 'last'])
            agg_numeric.columns = ['_'.join(col) for col in agg_numeric.columns]
            
            # Categorical aggregation with encoding
            if categorical_cols:
                agg_categorical = df_ml.sort_values('report_date').groupby(group_col)[categorical_cols].last()
                agg_categorical_encoded = pd.get_dummies(agg_categorical, drop_first=True)
            else:
                agg_categorical_encoded = pd.DataFrame(index=agg_numeric.index)
            
            # Boolean aggregation
            if len(bool_cols) > 0:
                agg_bool = df_ml.sort_values('report_date').groupby(group_col)[bool_cols].last()
            else:
                agg_bool = pd.DataFrame(index=agg_numeric.index)
            
            # Combine all features
            df_flat = pd.concat([agg_numeric, agg_categorical_encoded, agg_bool], axis=1)
            
            # Add churn label
            df_flat = df_flat.merge(labels.rename("is_churned"), left_index=True, right_index=True)
            df_flat = df_flat.reset_index()
            
            preprocessed_data = {
                'df_flat': df_flat,
                'numeric_cols': agg_numeric.columns.tolist(),
                'categorical_encoded_cols': agg_categorical_encoded.columns.tolist() if not agg_categorical_encoded.empty else [],
                'bool_cols': bool_cols.tolist(),
                'org_ids': df_flat['masked_organisation_id'],
                'target': df_flat['is_churned']
            }
            
            st.session_state.preprocessed_data = preprocessed_data
            return preprocessed_data
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return None

    def apply_pca_analysis(self, variance_threshold=0.95):
        """Apply PCA analysis - Step 1 of the ML pipeline"""
        if st.session_state.preprocessed_data is None:
            st.error("Please preprocess data first.")
            return None
        
        try:
            data = st.session_state.preprocessed_data
            df_flat = data['df_flat']
            
            # Prepare data for PCA
            X_all = df_flat.drop(columns=['is_churned', 'masked_organisation_id'])
            
            # Identify numeric columns for PCA
            numeric_cols = X_all.select_dtypes(include='number').columns
            non_pca_cols = X_all.drop(columns=numeric_cols)
            
            st.write(f"Applying PCA to {len(numeric_cols)} numeric features")
            
            # Scale numeric columns
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all[numeric_cols])
            
            # Apply PCA
            pca = PCA(n_components=variance_threshold)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA features DataFrame
            pca_features = pd.DataFrame(
                X_pca,
                columns=[f'pca_{i+1}' for i in range(X_pca.shape[1])],
                index=X_all.index
            )
            
            # Create PCA components DataFrame for analysis
            pca_components_df = pd.DataFrame(
                pca.components_,
                columns=numeric_cols,
                index=[f'pca_{i+1}' for i in range(pca.n_components_)]
            )
            
            # Get top contributing features for each component
            top_features_per_pca = {}
            for component in pca_components_df.index:
                top_features = pca_components_df.loc[component].abs().sort_values(ascending=False).head(10)
                top_features_per_pca[component] = top_features
            
            # Combine PCA features with non-numeric features
            df_pca_combined = pd.concat([
                non_pca_cols.reset_index(drop=True), 
                pca_features.reset_index(drop=True)
            ], axis=1)
            
            # Add back identifiers and target
            df_pca_combined['masked_organisation_id'] = df_flat['masked_organisation_id'].reset_index(drop=True)
            df_pca_combined['is_churned'] = df_flat['is_churned'].reset_index(drop=True)
            
            # Store PCA results
            pca_results = {
                'df_pca_combined': df_pca_combined,
                'pca_features': pca_features,
                'pca_model': pca,
                'scaler': scaler,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components_df': pca_components_df,
                'top_features_per_component': top_features_per_pca,
                'numeric_cols_original': numeric_cols.tolist(),
                'non_pca_cols': non_pca_cols.columns.tolist()
            }
            
            st.session_state.pca_results = pca_results
            
            # Display PCA summary
            st.success(f"✅ PCA Applied! Reduced {len(numeric_cols)} features to {X_pca.shape[1]} components")
            st.write(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")
            
            return pca_results
            
        except Exception as e:
            st.error(f"Error applying PCA: {str(e)}")
            return None

    def perform_feature_selection(self, top_n=20):
        """Perform feature selection using Random Forest - Step 2 after PCA"""
        if st.session_state.pca_results is None:
            st.error("Please apply PCA first.")
            return None
        
        try:
            pca_results = st.session_state.pca_results
            df_pca_combined = pca_results['df_pca_combined']
            
            # Prepare features and target
            X = df_pca_combined.drop(columns=['is_churned', 'masked_organisation_id'])
            y = df_pca_combined['is_churned']
            
            st.write(f"Performing feature selection on {X.shape[1]} features")
            
            # Train/test split for feature selection
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            
            # Train Random Forest for feature importance
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            feature_names = X.columns
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            
            # Select top N features
            top_features = feature_importance_df.head(top_n)['feature'].tolist()
            
            # Create final dataset with top features
            df_final = df_pca_combined[['masked_organisation_id', 'is_churned'] + top_features].copy()
            
            # Store feature selection results
            feature_selection_results = {
                'df_final': df_final,
                'feature_importance_df': feature_importance_df,
                'top_features': top_features,
                'rf_model': rf_model,
                'top_n': top_n,
                'X_train': X_train[top_features],
                'X_test': X_test[top_features],
                'y_train': y_train,
                'y_test': y_test
            }
            
            st.session_state.feature_selection_results = feature_selection_results
            
            st.success(f"✅ Feature Selection Complete! Selected top {top_n} features")
            
            return feature_selection_results
            
        except Exception as e:
            st.error(f"Error in feature selection: {str(e)}")
            return None

    def train_ml_models_on_selected_features(self):
        """Train ML models on selected features - Step 3 after feature selection"""
        if st.session_state.feature_selection_results is None:
            st.error("Please perform feature selection first.")
            return None
        
        try:
            fs_results = st.session_state.feature_selection_results
            X_train = fs_results['X_train']
            X_test = fs_results['X_test']
            y_train = fs_results['y_train']
            y_test = fs_results['y_test']
            
            st.write(f"Training models on {X_train.shape[1]} selected features")
            
            # Define models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'Naive Bayes': GaussianNB(),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            }
            
            results = {}
            
            # Train and evaluate models
            for name, model in models.items():
                try:
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'auc_score': auc_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'confusion_matrix': confusion_matrix(y_test, y_pred)
                    }
                    
                    st.write(f"✅ {name} trained successfully - Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
                    continue
            
            st.session_state.ml_models = results
            return results
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None

    def display_live_prediction(self):
        """Display live prediction interface"""
        st.header("🔮 Live Churn Prediction")
        
        if st.session_state.ml_models is None:
            st.warning("Please train models first to enable live predictions.")
            return
        
        st.markdown("""
        **Use this interface to predict churn probability for new organizations.** 
        Input the feature values below and select a trained model to get real-time predictions.
        """)
        
        models = st.session_state.ml_models
        fs_results = st.session_state.feature_selection_results
        top_features = fs_results['top_features']
        
        # Model selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Model Selection")
            
            selected_model_name = st.selectbox(
                "Choose Prediction Model",
                list(models.keys()),
                help="Select which trained model to use for prediction"
            )
            
            # Show model performance
            if selected_model_name:
                model_info = models[selected_model_name]
                st.metric("Model Accuracy", f"{model_info['accuracy']:.3f}")
                if model_info.get('auc_score'):
                    st.metric("AUC Score", f"{model_info['auc_score']:.3f}")
        
        with col2:
            st.subheader("📊 Feature Input")
            
            # Create input interface for top features
            input_data = {}
            
            # Get statistics from training data for default values
            fs_results = st.session_state.feature_selection_results
            X_train = fs_results['X_train']
            
            # Create input fields for each feature
            num_cols = 3
            feature_chunks = [top_features[i:i + num_cols] for i in range(0, len(top_features), num_cols)]
            
            for chunk in feature_chunks:
                cols = st.columns(len(chunk))
                for i, feature in enumerate(chunk):
                    with cols[i]:
                        # Get feature statistics for better input ranges
                        if feature in X_train.columns:
                            feature_data = X_train[feature]
                            min_val = float(feature_data.min())
                            max_val = float(feature_data.max())
                            mean_val = float(feature_data.mean())
                            
                            # Determine if feature is likely boolean/binary
                            unique_vals = feature_data.nunique()
                            if unique_vals == 2 and set(feature_data.unique()).issubset({0, 1, True, False}):
                                # Binary feature
                                input_data[feature] = st.selectbox(
                                    feature.replace('_', ' ').title(),
                                    [0, 1],
                                    index=int(mean_val > 0.5),
                                    key=f"input_{feature}"
                                )
                            else:
                                # Continuous feature
                                input_data[feature] = st.number_input(
                                    feature.replace('_', ' ').title(),
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    step=(max_val - min_val) / 100,
                                    key=f"input_{feature}",
                                    format="%.4f"
                                )
        
        # Prediction section
        st.subheader("🚀 Make Prediction")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                selected_model = models[selected_model_name]['model']
                
                try:
                    # Predict
                    prediction = selected_model.predict(input_df)[0]
                    prediction_proba = selected_model.predict_proba(input_df)[0] if hasattr(selected_model, 'predict_proba') else None
                    
                    # Store prediction results
                    st.session_state.latest_prediction = {
                        'model_name': selected_model_name,
                        'prediction': prediction,
                        'prediction_proba': prediction_proba,
                        'input_data': input_data
                    }
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        # Handle presets
        if hasattr(st.session_state, 'preset_selected'):
            X_train = fs_results['X_train']
            
            if st.session_state.preset_selected == "high_risk":
                # Use 90th percentile values for features that correlate with churn
                preset_data = {feature: float(X_train[feature].quantile(0.9)) for feature in top_features}
            elif st.session_state.preset_selected == "low_risk":
                # Use 10th percentile values
                preset_data = {feature: float(X_train[feature].quantile(0.1)) for feature in top_features}
            else:  # average
                # Use median values
                preset_data = {feature: float(X_train[feature].median()) for feature in top_features}
            
            # Apply preset and rerun to update inputs
            for feature, value in preset_data.items():
                st.session_state[f"input_{feature}"] = value
            
            # Clear preset selection
            delattr(st.session_state, 'preset_selected')
            st.rerun()
        
        # Display prediction results
        if hasattr(st.session_state, 'latest_prediction'):
            pred_results = st.session_state.latest_prediction
            
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction result card
                prediction = pred_results['prediction']
                prediction_proba = pred_results['prediction_proba']
                
                if prediction == 1:
                    churn_prob = prediction_proba[1] if prediction_proba is not None else 1.0
                    st.markdown(f"""
                    <div class="high-risk">
                        <h3>⚠️ HIGH CHURN RISK</h3>
                        <p><strong>Prediction:</strong> Customer likely to churn</p>
                        <p><strong>Confidence:</strong> {churn_prob:.1%}</p>
                        <p><strong>Model Used:</strong> {pred_results['model_name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    retain_prob = prediction_proba[0] if prediction_proba is not None else 1.0
                    st.markdown(f"""
                    <div class="low-risk">
                        <h3>✅ LOW CHURN RISK</h3>
                        <p><strong>Prediction:</strong> Customer likely to stay</p>
                        <p><strong>Confidence:</strong> {retain_prob:.1%}</p>
                        <p><strong>Model Used:</strong> {pred_results['model_name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability visualization
                if prediction_proba is not None:
                    prob_df = pd.DataFrame({
                        'Status': ['Stay', 'Churn'],
                        'Probability': prediction_proba,
                        'Color': ['#28a745', '#dc3545']
                    })
                    
                    fig_prob = px.bar(
                        prob_df,
                        x='Status',
                        y='Probability',
                        color='Color',
                        color_discrete_map={'#28a745': '#28a745', '#dc3545': '#dc3545'},
                        title="Churn Probability",
                        labels={'Probability': 'Probability'}
                    )
                    fig_prob.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                    fig_prob.update_layout(showlegend=False, yaxis_tickformat='.0%')
                    st.plotly_chart(fig_prob, use_container_width=True)
            
            with col2:
                # Feature contribution analysis
                st.subheader("📊 Feature Data")
                
                # Show input values vs training data distribution
                input_data = pred_results['input_data']
                X_train = fs_results['X_train']
                
                # Calculate percentiles for each input feature
                feature_analysis = []
                for feature, value in input_data.items():
                    if feature in X_train.columns:
                        percentile = (X_train[feature] < value).mean() * 100
                        feature_analysis.append({
                            'Feature': feature.replace('_', ' ').title(),
                            'Input Value': value,
                            'Percentile': percentile,
                            'Risk Level': 'High' if percentile > 75 else 'Medium' if percentile > 25 else 'Low'
                        })
                
                analysis_df = pd.DataFrame(feature_analysis)
                analysis_df = analysis_df.sort_values('Percentile', ascending=False)
                
                # Color code by risk level
                def color_risk(val):
                    if val == 'High':
                        return 'background-color: #ffebee'
                    elif val == 'Medium':
                        return 'background-color: #fff3e0'
                    else:
                        return 'background-color: #e8f5e8'
                
                styled_df = analysis_df.style.applymap(color_risk, subset=['Risk Level'])
                st.dataframe(styled_df, use_container_width=True)
        
        # Batch prediction section
        st.subheader("📋 Batch Prediction")
        
        st.markdown("""
        **Upload a CSV file with the same features to predict churn for multiple organizations.**
        """)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file for batch prediction",
            type=['csv'],
            help="CSV should contain columns matching the selected features"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded file with {len(batch_df)} rows and {len(batch_df.columns)} columns")
                
                # Check if required features are present
                missing_features = set(top_features) - set(batch_df.columns)
                
                if missing_features:
                    st.error(f"Missing required features: {', '.join(missing_features)}")
                else:
                    if st.button("🚀 Run Batch Prediction"):
                        with st.spinner("Running batch predictions..."):
                            # Prepare data
                            batch_X = batch_df[top_features]
                            
                            # Make predictions
                            selected_model = models[selected_model_name]['model']
                            batch_predictions = selected_model.predict(batch_X)
                            
                            if hasattr(selected_model, 'predict_proba'):
                                batch_proba = selected_model.predict_proba(batch_X)
                                churn_proba = batch_proba[:, 1]
                            else:
                                churn_proba = batch_predictions.astype(float)
                            
                            # Add predictions to dataframe
                            batch_results = batch_df.copy()
                            batch_results['churn_prediction'] = batch_predictions
                            batch_results['churn_probability'] = churn_proba
                            batch_results['risk_level'] = pd.cut(
                                churn_proba,
                                bins=[0, 0.3, 0.7, 1.0],
                                labels=['Low', 'Medium', 'High']
                            )
                            
                            # Display results summary
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                high_risk_count = (batch_results['risk_level'] == 'High').sum()
                                st.metric("High Risk Customers", high_risk_count)
                            
                            with col2:
                                medium_risk_count = (batch_results['risk_level'] == 'Medium').sum()
                                st.metric("Medium Risk Customers", medium_risk_count)
                            
                            with col3:
                                low_risk_count = (batch_results['risk_level'] == 'Low').sum()
                                st.metric("Low Risk Customers", low_risk_count)
                            
                            # Display results table
                            st.subheader("Batch Prediction Results")
                            
                            # Add color coding for risk levels
                            def highlight_risk(row):
                                if row['risk_level'] == 'High':
                                    return ['background-color: #ffebee'] * len(row)
                                elif row['risk_level'] == 'Medium':
                                    return ['background-color: #fff3e0'] * len(row)
                                else:
                                    return ['background-color: #e8f5e8'] * len(row)
                            
                            display_cols = ['churn_prediction', 'churn_probability', 'risk_level'] + top_features[:5]
                            available_cols = [col for col in display_cols if col in batch_results.columns]
                            
                            styled_batch = batch_results[available_cols].style.apply(highlight_risk, axis=1)
                            st.dataframe(styled_batch, use_container_width=True)
                            
                            # Download results
                            results_csv = batch_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Batch Results",
                                data=results_csv,
                                file_name=f"batch_churn_predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                            # Risk distribution chart
                            risk_counts = batch_results['risk_level'].value_counts()
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color_discrete_map={
                                    'Low': '#28a745',
                                    'Medium': '#ffc107',
                                    'High': '#dc3545'
                                }
                            )
                            fig_risk.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_risk, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

    def display_data_overview(self):
        """Display data overview and summary statistics"""
        st.header("📋 Data Overview")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first using the sidebar.")
            return
        
        datasets = st.session_state.datasets
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Commercial Records",
                f"{len(datasets['commercial']):,}",
                help="Total commercial data records"
            )
        
        with col2:
            st.metric(
                "Product Records", 
                f"{len(datasets['product']):,}",
                help="Total product data records"
            )
        
        with col3:
            unique_orgs = datasets['commercial']['masked_organisation_id'].nunique()
            st.metric(
                "Unique Organizations",
                f"{unique_orgs:,}",
                help="Number of unique organizations in dataset"
            )
        
        # Data quality summary
        st.subheader("Data Quality Summary")
        
        summary_data = []
        for name, df in datasets.items():
            summary_data.append({
                'Dataset': name.title(),
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Unique Organizations': df['masked_organisation_id'].nunique(),
                'Missing Values': f"{df.isnull().sum().sum():,}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Show sample data
        with st.expander("View Sample Data"):
            dataset_choice = st.selectbox("Select dataset to preview:", list(datasets.keys()))
            st.dataframe(datasets[dataset_choice].head(10), use_container_width=True)

    def display_preprocessing_pipeline(self):
        """Display the complete preprocessing pipeline"""
        st.header("🔧 ML Preprocessing Pipeline")
        
        if not st.session_state.analysis_complete:
            st.warning("Please complete the churn analysis first.")
            return
        
        st.markdown("""
            ### 🔄 Pipeline Steps:

            1. **Data Merging & Sorting**  
            - Merge commercial and product datasets on `masked_organisation_id` and `report_date`  
            - Sort by `report_date` for time-aware feature engineering  

            2. **Data Preprocessing**  
            - Handle missing values (numeric, categorical, datetime)  
            - Encode categorical variables  
            - Create derived features (durations, days since events)

            3. **Data Flattening**  
            - Aggregate time-series data into one row per organization  
            - Apply statistical summaries (mean, std, min, max, last) for numeric features  
            - Capture last known state for categorical and boolean features
                    
            4. **PCA Analysis**  
            - Apply dimensionality reduction to scaled numeric features  
            - Retain 95% variance using Principal Component Analysis

            5. **Feature Selection**  
            - Identify top N most important features using a trained Random Forest model

            6. **Model Training & Evaluation**  
            - Train multiple classifiers (Random Forest, SVM, Gradient Boosting, etc.)  
            - Evaluate using accuracy, precision, recall, and F1-score
            """)

        
        # Step 1: Data Preprocessing
        st.subheader("Step 1: Data Preprocessing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            numeric_strategy = st.selectbox(
                "Numeric Strategy",
                ['mean', 'median', 'zero', 'drop'],
                index=0,
                help="Strategy for handling missing numeric values"
            )
        
        with col2:
            categorical_strategy = st.selectbox(
                "Categorical Strategy",
                ['mode', 'constant', 'drop'],
                index=0,
                help="Strategy for handling missing categorical values"
            )
        
        with col3:
            datetime_strategy = st.selectbox(
                "Datetime Strategy",
                ['drop', 'constant'],
                index=0,
                help="Strategy for handling missing datetime values"
            )
        
        if st.button("🔄 Run Preprocessing", type="primary"):
            with st.spinner("Preprocessing data..."):
                preprocessed_data = self.preprocess_for_ml()
                if preprocessed_data:
                    st.success("✅ Data preprocessing completed!")
                    
                    df_flat = preprocessed_data['df_flat']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Organizations", len(df_flat))
                    with col2:
                        st.metric("Total Features", len(df_flat.columns) - 2)  # Exclude ID and target
                    with col3:
                        churn_rate = df_flat['is_churned'].mean()
                        st.metric("Churn Rate", f"{churn_rate:.1%}")

    def display_pca_analysis(self):
        """Display PCA analysis - Step 2"""
        st.header("📊 Step 2: PCA Analysis")
        
        if st.session_state.preprocessed_data is None:
            st.warning("Please complete preprocessing first.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            variance_threshold = st.slider(
                "Variance Threshold",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Percentage of variance to retain"
            )
            
            if st.button("🔍 Apply PCA", type="primary"):
                with st.spinner("Applying PCA..."):
                    pca_results = self.apply_pca_analysis(variance_threshold)
                    if pca_results:
                        st.success("✅ PCA Analysis completed!")
        
        with col1:
            if st.session_state.pca_results is not None:
                pca_results = st.session_state.pca_results
                
                # Explained variance plot
                fig_variance = go.Figure()
                fig_variance.add_trace(go.Scatter(
                    x=list(range(1, len(pca_results['explained_variance_ratio']) + 1)),
                    y=pca_results['explained_variance_ratio'],
                    mode='lines+markers',
                    name='Individual',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Cumulative variance
                cumulative_var = np.cumsum(pca_results['explained_variance_ratio'])
                fig_variance.add_trace(go.Scatter(
                    x=list(range(1, len(cumulative_var) + 1)),
                    y=cumulative_var,
                    mode='lines+markers',
                    name='Cumulative',
                    line=dict(color='#ff7f0e', width=2)
                ))
                
                fig_variance.update_layout(
                    title="PCA Explained Variance",
                    xaxis_title="Principal Component",
                    yaxis_title="Variance Explained",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_variance, use_container_width=True)
                
                # Show PCA components summary
                st.subheader("PCA Components Summary")
                
                component_summary = pd.DataFrame({
                    'Component': [f"PC{i+1}" for i in range(len(pca_results['explained_variance_ratio']))],
                    'Explained_Variance': pca_results['explained_variance_ratio'],
                    'Cumulative_Variance': np.cumsum(pca_results['explained_variance_ratio'])
                })
                
                st.dataframe(component_summary.round(4), use_container_width=True)

    def display_feature_selection(self):
        """Display feature selection - Step 3"""
        st.header("🎯 Step 3: Feature Selection")
        
        if st.session_state.pca_results is None:
            st.warning("Please apply PCA first.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Selection Configuration")
            
            top_n = st.slider(
                "Number of Top Features",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Select top N most important features"
            )
            
            if st.button("🎯 Perform Feature Selection", type="primary"):
                with st.spinner("Selecting features..."):
                    fs_results = self.perform_feature_selection(top_n)
                    if fs_results:
                        st.success(f"✅ Selected top {top_n} features!")
        
        with col2:
            if st.session_state.feature_selection_results is not None:
                fs_results = st.session_state.feature_selection_results
                feature_importance_df = fs_results['feature_importance_df']
                
                # Feature importance plot
                top_features_display = feature_importance_df.head(15)
                
                fig_importance = px.bar(
                    top_features_display,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Feature Importances",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_importance.update_traces(texttemplate='%{x:.4f}', textposition='outside')
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Rankings")
                st.dataframe(feature_importance_df.head(20).round(4), use_container_width=True)

    def display_model_training(self):
        """Display model training - Step 4"""
        st.header("🤖 Step 4: Model Training")
        
        if st.session_state.feature_selection_results is None:
            st.warning("Please perform feature selection first.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Training Configuration")
            
            fs_results = st.session_state.feature_selection_results
            st.write(f"Training on {len(fs_results['top_features'])} selected features")
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training ML models..."):
                    models = self.train_ml_models_on_selected_features()
                    if models:
                        st.success("✅ Models trained successfully!")
        
        with col2:
            if st.session_state.ml_models is not None:
                st.subheader("Model Performance Comparison")
                
                models = st.session_state.ml_models
                
                # Create performance comparison
                performance_data = []
                for name, result in models.items():
                    performance_data.append({
                        'Model': name,
                        'Accuracy': result['accuracy'],
                        'AUC Score': result.get('auc_score', 'N/A'),
                        'CV Mean': result['cv_mean'],
                        'CV Std': result['cv_std']
                    })
                
                performance_df = pd.DataFrame(performance_data)
                performance_df = performance_df.sort_values('Accuracy', ascending=False)
                
                # Performance bar chart
                fig_perf = px.bar(
                    performance_df,
                    x='Accuracy',
                    y='Model',
                    orientation='h',
                    title="Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale='Viridis'
                )
                fig_perf.update_traces(texttemplate='%{x:.3f}', textposition='outside')
                fig_perf.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Performance table
                st.dataframe(performance_df.round(4), use_container_width=True)

    def display_churn_analysis(self):
        """Display churn analysis results"""
        st.header("🎯 Churn Analysis Results")
        
        if not st.session_state.analysis_complete:
            st.warning("Please run the analysis first using the sidebar.")
            return
        
        merged_df = st.session_state.merged_df
        org_features = st.session_state.org_features
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        churn_rate = org_features['is_churned'].mean()
        total_orgs = len(org_features)
        churned_orgs = org_features['is_churned'].sum()
        active_orgs = total_orgs - churned_orgs
        
        with col1:
            st.metric(
                "Overall Churn Rate",
                f"{churn_rate:.1%}",
                help="Percentage of organizations that have churned"
            )
        
        with col2:
            st.metric(
                "Total Organizations",
                f"{total_orgs:,}",
                help="Total number of organizations analyzed"
            )
        
        with col3:
            st.metric(
                "Churned Organizations",
                f"{churned_orgs:,}",
                delta=f"-{churned_orgs:,}",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Active Organizations",
                f"{active_orgs:,}",
                delta=f"+{active_orgs:,}",
                delta_color="normal"
            )
        
        # Churn distribution pie chart
        st.subheader("Churn Distribution")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_pie = px.pie(
                values=[active_orgs, churned_orgs],
                names=['Active', 'Churned'],
                title="Customer Status Distribution",
                color_discrete_map={'Active': '#2ecc71', 'Churned': '#e74c3c'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Churn rate by time period
            if 'report_date' in merged_df.columns:
                monthly_churn = self.calculate_monthly_churn(merged_df)
                
                if not monthly_churn.empty:
                    fig_time = px.line(
                        monthly_churn.reset_index(),
                        x='report_date',
                        y='churn_rate',
                        title="Churn Rate Over Time",
                        labels={'churn_rate': 'Churn Rate (%)', 'report_date': 'Month'}
                    )
                    fig_time.update_traces(line_color='#e74c3c', line_width=3)
                    fig_time.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig_time, use_container_width=True)

    def calculate_monthly_churn(self, merged_df):
        """Calculate monthly churn rates"""
        try:
            monthly_stats = (merged_df
                            .groupby([pd.Grouper(key='report_date', freq='M'), 'is_churned'])
                            .size()
                            .unstack(fill_value=0))
            
            if True in monthly_stats.columns and False in monthly_stats.columns:
                monthly_stats['total'] = monthly_stats[True] + monthly_stats[False]
                monthly_stats['churn_rate'] = monthly_stats[True] / monthly_stats['total']
                return monthly_stats
            
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    def display_model_evaluation(self):
        """Display detailed model evaluation"""
        st.header("📈 Model Evaluation & Results")
        
        if st.session_state.ml_models is None:
            st.warning("Please train models first.")
            return
        
        models = st.session_state.ml_models
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            list(models.keys()),
            help="Choose a model to see detailed evaluation metrics"
        )
        
        if selected_model:
            model_results = models[selected_model]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Metrics
                st.subheader(f"{selected_model} - Performance Metrics")
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Accuracy", f"{model_results['accuracy']:.4f}")
                    if model_results.get('auc_score'):
                        st.metric("AUC Score", f"{model_results['auc_score']:.4f}")
                
                with metrics_col2:
                    st.metric("CV Mean", f"{model_results['cv_mean']:.4f}")
                    st.metric("CV Std", f"±{model_results['cv_std']:.4f}")
                
                # Classification report
                if 'y_test' in model_results and 'y_pred' in model_results:
                    st.subheader("Classification Report")
                    report = classification_report(
                        model_results['y_test'], 
                        model_results['y_pred'],
                        output_dict=True
                    )
                    
                    # Convert to DataFrame for better display
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
            
            with col2:
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                cm = model_results['confusion_matrix']
                
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Active', 'Churned'],
                    y=['Active', 'Churned'],
                    color_continuous_scale='Blues',
                    title=f"Confusion Matrix - {selected_model}"
                )
                
                # Add text annotations
                for i in range(len(cm)):
                    for j in range(len(cm[0])):
                        fig_cm.add_annotation(
                            x=j, y=i,
                            text=str(cm[i][j]),
                            showarrow=False,
                            font=dict(color="white" if cm[i][j] > cm.max()/2 else "black", size=16)
                        )
                
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # ROC Curve
            if model_results.get('y_proba') is not None:
                st.subheader("ROC Curve")
                
                fpr, tpr, _ = roc_curve(model_results['y_test'], model_results['y_proba'])
                auc_score = roc_auc_score(model_results['y_test'], model_results['y_proba'])
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{selected_model} (AUC = {auc_score:.3f})',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_roc.update_layout(
                    title=f"ROC Curve - {selected_model}",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    width=600, height=400
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)

    def display_industry_analysis(self):
        """Display industry and country analysis"""
        st.header("🏭 Industry & Geographic Analysis")
        
        if not st.session_state.analysis_complete:
            st.warning("Please run the analysis first using the sidebar.")
            return
        
        org_features = st.session_state.org_features
        
        # Find industry and country columns
        industry_col = None
        country_col = None
        
        for col in org_features.columns:
            if 'industry' in col.lower():
                industry_col = col
                break
        
        for col in org_features.columns:
            if 'country' in col.lower():
                country_col = col
                break
        
        col1, col2 = st.columns(2)
        
        # Industry Analysis
        with col1:
            if industry_col:
                st.subheader("Industry Churn Analysis")
                industry_stats = self.analyze_categorical_churn(org_features, industry_col, 'Industry')
                
                # Top industries by churn rate
                top_industries = industry_stats.head(10)
                
                fig_industry = px.bar(
                    top_industries,
                    x='Churn_Rate',
                    y='Industry',
                    orientation='h',
                    title="Top 6 Industries by Churn Rate",
                    labels={'Churn_Rate': 'Churn Rate (%)', 'Industry': 'Industry'},
                    color='Churn_Rate',
                    color_continuous_scale='Reds'
                )
                fig_industry.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_industry.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                st.plotly_chart(fig_industry, use_container_width=True)
                
                # Industry summary table
                st.dataframe(
                    top_industries.head(10).round(3),
                    use_container_width=True
                )
            else:
                st.info("No industry column found in the dataset")
        
        # Country Analysis
        with col2:
            if country_col:
                st.subheader("Country Churn Analysis")
                country_stats = self.analyze_categorical_churn(org_features, country_col, 'Country')
                
                # Top countries by churn rate
                top_countries = country_stats.head(10)
                
                fig_country = px.bar(
                    top_countries,
                    x='Churn_Rate',
                    y='Country',
                    orientation='h',
                    title="Top 6 Countries by Churn Rate",
                    labels={'Churn_Rate': 'Churn Rate (%)', 'Country': 'Country'},
                    color='Churn_Rate',
                    color_continuous_scale='Oranges'
                )
                fig_country.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_country.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                st.plotly_chart(fig_country, use_container_width=True)
                
                # Country summary table
                st.dataframe(
                    top_countries.head(10).round(3),
                    use_container_width=True
                )
            else:
                st.info("No country column found in the dataset")

    def analyze_categorical_churn(self, df, column, category_name):
        """Analyze churn by categorical variable"""
        stats = df.groupby(column)['is_churned'].agg(['count', 'sum', 'mean']).reset_index()
        stats.columns = [category_name, 'Total_Customers', 'Churned_Count', 'Churn_Rate']
        stats = stats[stats['Total_Customers'] >= 3]  # Filter small categories
        stats = stats.sort_values('Churn_Rate', ascending=False)
        
        return stats

    def display_insights_recommendations(self):
        """Display key insights and recommendations"""
        st.header("💡 Key Insights & Methodology")
        
        if not st.session_state.analysis_complete:
            st.warning("Please run the analysis first using the sidebar.")
            return
        
        org_features = st.session_state.org_features
        
        # Key insights
        churn_rate = org_features['is_churned'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Key Insights")
            
            insights = [
                f"Overall churn rate is {churn_rate:.1%}",
                f"Analysis covers {len(org_features):,} organizations",
                f"Dataset spans {len(st.session_state.merged_df):,} records",
            ]
            
            # Add feature selection insights if available
            if st.session_state.feature_selection_results is not None:
                fs_results = st.session_state.feature_selection_results
                top_feature = fs_results['feature_importance_df'].iloc[0]
                insights.append(f"Most important feature: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
            
            # Add PCA insights if available
            if st.session_state.pca_results is not None:
                pca_results = st.session_state.pca_results
                total_variance = sum(pca_results['explained_variance_ratio'])
                insights.append(f"PCA retained {total_variance:.1%} of variance with {len(pca_results['explained_variance_ratio'])} components")
            
            for insight in insights:
                st.markdown(f"• {insight}")
        
        with col2:
            st.subheader("📋 Methodology")
            
            recommendations = [
                "**PCA-Driven Approach**: Use dimensionality reduction to focus on key patterns",
                "**Feature-Based Strategy**: Prioritize top features identified through Random Forest",
                "**Model Ensemble**: Combine predictions from multiple algorithms for better accuracy",
                "**Regular Retraining**: Update models monthly with new data",
                "**Proactive Monitoring**: Track top feature changes in real-time"
            ]
            
            for rec in recommendations:
                st.markdown(f"• {rec}")

    def sidebar_controls(self):
        """Create sidebar controls"""
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Data loading section
        st.sidebar.header("📁 Data Loading")
        
        data_directory = st.sidebar.text_input(
            "Data Directory Path",
            value="resources",
            help="Path to the directory containing your CSV files"
        )
        
        if st.sidebar.button("🔄 Load Data", type="primary"):
            with st.sidebar:
                datasets = self.load_data(data_directory)
                if datasets:
                    st.success("✅ Data loaded successfully!")
                    # Display basic info
                    st.write(f"**Datasets loaded:**")
                    for name, df in datasets.items():
                        st.write(f"• {name}: {len(df):,} rows")
        
        # Analysis section
        st.sidebar.header("🔬 Analysis")
        
        if st.session_state.data_loaded:
            if st.sidebar.button("▶️ Run Churn Analysis", type="primary"):
                with st.sidebar:
                    merged_df, org_features = self.run_analysis()
                    if merged_df is not None and org_features is not None:
                        st.success("✅ Analysis completed!")
                        st.write(f"**Results:**")
                        st.write(f"• {len(org_features):,} organizations analyzed")
                        st.write(f"• {org_features['is_churned'].mean():.1%} churn rate")
        else:
            st.sidebar.info("Load data first to enable analysis")
        
        # Pipeline status
        st.sidebar.header("🚀 ML Pipeline Status")
        
        # Show pipeline progress
        progress_items = [
            ("Data Loaded", st.session_state.data_loaded),
            ("Analysis Complete", st.session_state.analysis_complete),
            ("Data Preprocessed", st.session_state.preprocessed_data is not None),
            ("PCA Applied", st.session_state.pca_results is not None),
            ("Features Selected", st.session_state.feature_selection_results is not None),
            ("Models Trained", st.session_state.ml_models is not None)
        ]
        
        for item, status in progress_items:
            if status:
                st.sidebar.success(f"✅ {item}")
            else:
                st.sidebar.info(f"⏳ {item}")
        
        # Download section
        st.sidebar.header("📥 Export Results")
        
        if st.session_state.feature_selection_results is not None:
            fs_results = st.session_state.feature_selection_results
            df_final = fs_results['df_final']
            
            final_csv = df_final.to_csv(index=False)
            
            st.sidebar.download_button(
                label="📊 Download Final ML Dataset",
                data=final_csv,
                file_name="final_ml_dataset.csv",
                mime="text/csv"
            )
            
            # Feature importance CSV
            importance_csv = fs_results['feature_importance_df'].to_csv(index=False)
            st.sidebar.download_button(
                label="🎯 Download Feature Importance",
                data=importance_csv,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.info("Complete ML pipeline to enable downloads")

    def main(self):
        """Main dashboard function"""
        self.initialize_session_state()
        
        # Title and description
        st.markdown('<h1 style="font-size: 2.5em; color: #4B8BBE;">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

        st.markdown("""
        <div style='font-size: 1.1em; line-height: 1.6'>
        Welcome to the <b>Customer Churn Prediction Dashboard.</b>  
        This tool walks you through a complete machine learning pipeline for churn analysis, including:
                    
        <strong>-> Data Analysis</strong>  
        <strong>-> Data Preprocessing</strong>  
        <strong>-> PCA for Dimensionality Reduction</strong>  
        <strong>-> Feature Selection</strong>  
        <strong>-> Model Training & Evaluation</strong>  
        <strong>-> Live Prediction Interface</strong>  

        Use the navigation tabs above to explore each stage of the pipeline.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar controls
        self.sidebar_controls()
        
        # Main content tabs - Updated to include Live Prediction
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📋 Data Overview", 
            "🎯 Churn Analysis", 
            "🏭 Industry Analysis", 
            "🔧 Step 1: Preprocessing",
            "📊 Step 2: PCA Analysis",
            "🎯 Step 3: Feature Selection",
            "🤖 Step 4: Model Training",
            "📈 Model Evaluation",
            "🔮 Live Prediction",
            "🌟 Future Steps"
        ])
        
        with tab1:
            self.display_data_overview()
        
        with tab2:
            self.display_churn_analysis()
        
        with tab3:
            self.display_industry_analysis()
        
        with tab4:
            self.display_preprocessing_pipeline()
        
        with tab5:
            self.display_pca_analysis()
        
        with tab6:
            self.display_feature_selection()
        
        with tab7:
            self.display_model_training()
        
        with tab8:
            self.display_model_evaluation()
        
        with tab9:
            self.display_live_prediction()
            
        with tab10:
            self.display_insights_recommendations()
            
            st.markdown("""
                ### 🔮 **Future Steps (Coming Soon)**

                - 🧪 **Hyperparameter Tuning**  
                Optimize model performance using grid search, randomized search, or Bayesian optimization for better accuracy and generalization.

                - 🧠 **Advanced Modeling Techniques**  
                Explore and incorporate more sophisticated models such as **XGBoost**, **LightGBM**, and **deep learning architectures** (e.g., feedforward neural networks or temporal models like LSTMs) to improve churn prediction accuracy. These models can better capture complex patterns, feature interactions, and non-linear relationships in the data — especially beneficial for larger and more diverse datasets.

                - 📈 **Learning Curves**  
                Plot training and validation scores across dataset sizes to diagnose overfitting or underfitting.

                - 🧬 **Ensemble Stacking**  
                Combine multiple base models using stacking or blending to improve robustness and predictive power.

                - 🧾 **AutoML Integration**  
                Automate model selection and feature engineering using tools like Auto-sklearn, TPOT, or H2O AutoML.

                - ☁️ **Model Deployment to Cloud**  
                Deploy trained models to cloud platforms such as **AWS SageMaker**, **Google Cloud AI Platform**, or **Azure ML** for scalable, reliable access. Enable RESTful API endpoints for real-time predictions, integrate with CI/CD pipelines for automated updates, and monitor usage, latency, and model drift in production environments.

                - 🔁 **Scalable Processing for Larger Datasets**  
                Adapt the pipeline to efficiently handle larger datasets using batch processing, optimized data pipelines, and memory-aware transformations for preprocessing, feature extraction, and model training.

                - 🤖 **AI-Powered Output Explanation (RAG)**  
                Integrate Large Language Model with Retrieval-Augmented Generation (RAG) pipelines to generate natural language explanations of model predictions, supported by contextual data and relevant documentation, enabling users to **ask why a customer churned** and receive transparent, AI-generated answers.

                - 🎯 **Model Interpretability & SHAP Analysis**  
                Add SHAP (SHapley Additive exPlanations) values to explain individual predictions and understand feature contributions for each customer prediction.

                - 📱 **Real-time Monitoring Dashboard**  
                Create a live monitoring system that tracks model performance, data drift, and prediction accuracy over time with automated alerts.

                - 🔄 **Automated Retraining Pipeline**  
                Implement automated model retraining when new data becomes available or when model performance degrades.

                ---

                ⚠️ **Limitations of the Current Approach**  
                - **Designed for in-memory processing**: May not scale well with datasets larger than a few thousand records.  
                - **Simple feature selection**: Only Random Forest importance is used; may miss subtle interactions captured by other methods.  
                - **Basic temporal handling**: Sorting by `report_date` is done, but no time-series modeling or lag feature engineering is applied.  
                - **No hyperparameter optimization**: All models use default settings; performance may be suboptimal.  
                - **Limited real-time capabilities**: Current live prediction requires manual input; no automated data pipeline integration.
            """)

        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
        Customer Churn Prediction Dashboard | PCA → Feature Selection → ML Pipeline → Live Prediction
        </div>
        """, unsafe_allow_html=True)

# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = ChurnDashboard()
    dashboard.main()
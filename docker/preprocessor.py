import os
import joblib
import numpy as np
import pandas as pd
import logging
import re

# --- Note: In a real project, you would package your 'src' directory ---
# --- and install it. For simplicity here, we assume the preprocessor  ---
# --- files are available in the container.                           ---
from src.data_processing.preprocessor.cleaning import clean_data
from src.data_processing.preprocessor.type_conversion import convert_column_types
from src.data_processing.preprocessor.missing_imputation import MissingValueImputer
from src.data_processing.preprocessor.outlier_handler import OutlierTransformer
from src.data_processing.preprocessor.feature_engineering import FeatureEngineeringTransformer
from src.data_processing.preprocessor.binning import BinningTransformer
from src.data_processing.preprocessor.rare_label_encoder import RareLabelEncoder
from src.data_processing.preprocessor.encoding import OneHotEncodingTransformer
from src.data_processing.preprocessor.vif_filter import RFECVTransformer as VIFTransformer
from src.data_processing.preprocessor.target_encoder_wrapper import TargetEncoderWrapper
from sklearn.preprocessing import StandardScaler

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Schema Definition ---
# This is the expected schema for the raw data coming into the endpoint.
EXPECTED_RAW_SCHEMA = {
    'Prospect ID': 'prospect_id', 'Lead Number': 'lead_number', 'Lead Origin': 'lead_origin',
    'Lead Source': 'lead_source', 'Do Not Email': 'do_not_email', 'Do Not Call': 'do_not_call',
    'TotalVisits': 'total_visits',
    'Total Time Spent on Website': 'total_time_spent_on_website',
    'Page Views Per Visit': 'page_views_per_visit', 'Last Activity': 'last_activity',
    'Country': 'country', 'Specialization': 'specialization',
    'How did you hear about X Education': 'how_did_you_hear_about_x_education',
    'What is your current occupation': 'what_is_your_current_occupation',
    'What matters most to you in choosing a course': 'what_matters_most_to_you_in_choosing_a_course',
    'Search': 'search', 'Magazine': 'magazine', 'Newspaper Article': 'newspaper_article',
    'X Education Forums': 'x_education_forums', 'Newspaper': 'newspaper',
    'Digital Advertisement': 'digital_advertisement', 'Through Recommendations': 'through_recommendations',
    'Receive More Updates About Our Courses': 'receive_more_updates_about_our_courses',
    'Tags': 'tags', 'Lead Quality': 'lead_quality',
    'Update me on Supply Chain Content': 'update_me_on_supply_chain_content',
    'Get updates on DM Content': 'get_updates_on_dm_content', 'Lead Profile': 'lead_profile',
    'City': 'city', 'Asymmetrique Activity Index': 'asymmetrique_activity_index',
    'Asymmetrique Profile Index': 'asymmetrique_profile_index',
    'Asymmetrique Activity Score': 'asymmetrique_activity_score',
    'Asymmetrique Profile Score': 'asymmetrique_profile_score',
    'I agree to pay the amount through cheque': 'i_agree_to_pay_the_amount_through_cheque',
    'A free copy of Mastering The Interview': 'a_free_copy_of_mastering_the_interview',
    'Last Notable Activity': 'last_notable_activity'
}

class PreprocessingPipeline:
    """
    A class to encapsulate the entire multi-stage preprocessing pipeline.
    It loads all necessary artifacts during initialization and provides a single
    `transform` method to process raw incoming data for prediction.
    """
    def __init__(self, artifact_path='/opt/ml/model/artifacts'):
        self.artifact_path = artifact_path
        self.artifacts = self._load_all_artifacts()

    def _load_all_artifacts(self):
        logger.info(f"Loading all preprocessing artifacts from: {self.artifact_path}")
        artifacts = {}
        artifact_map = {
            "imputer": "imputation/imputer_values.joblib",
            "outlier_config": "outliers/outlier_config.joblib",
            "vif_selected_features": "vif/vif_selected_features.joblib",
            "encoder": "encoding/full_encoding_transformer.joblib",
            "initial_scaler": "scaling/feature_scaler.joblib",
            "final_target_encoder": "final_target_encoder.joblib",
            "final_scaler": "feature_scaler.pkl"
        }
        for name, rel_path in artifact_map.items():
            full_path = os.path.join(self.artifact_path, rel_path)
            try:
                artifacts[name] = joblib.load(full_path)
                logger.info(f"✅ Successfully loaded artifact: {name}")
            except Exception as e:
                logger.error(f"❌ Failed to load artifact: {name} from path: {full_path}. Error: {e}")
                artifacts[name] = None
        return artifacts

    def _standardize_input_columns(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # ... (This logic is the same as in your prediction_service.py)
        def standardize(name): return re.sub(r'[^0-9a-zA-Z]+', '_', str(name)).lower().strip('_')
        raw_to_snake = {standardize(k): v for k, v in EXPECTED_RAW_SCHEMA.items()}
        df_renamed = raw_df.copy()
        df_renamed.columns = [standardize(col) for col in df_renamed.columns]
        final_df = pd.DataFrame()
        expected_snake_cols = list(EXPECTED_RAW_SCHEMA.values())
        for standardized_raw_name, final_snake_name in raw_to_snake.items():
            if standardized_raw_name in df_renamed.columns: final_df[final_snake_name] = df_renamed[standardized_raw_name]
        for col in expected_snake_cols:
            if col not in final_df.columns: final_df[col] = np.nan
        final_df = final_df[expected_snake_cols]
        return final_df

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting preprocessing for input data with shape {raw_df.shape}")
        
        df = self._standardize_input_columns(raw_df)

        # The sequence of transformations must exactly match the training pipeline
        df = clean_data(df, verbose=False)
        df = convert_column_types(df)
        
        imputer = MissingValueImputer(); imputer.imputer_values_ = self.artifacts["imputer"]
        df = imputer.transform(df)
        
        outlier_transformer = OutlierTransformer(); outlier_transformer.clip_bounds_ = self.artifacts["outlier_config"]
        df = outlier_transformer.transform(df)
        
        fe_transformer = FeatureEngineeringTransformer(); fe_transformer.fit(df); df = fe_transformer.transform(df)
        
        binning_transformer = BinningTransformer(); df = binning_transformer.transform(df)
        
        rare_label_encoder = RareLabelEncoder(); df = rare_label_encoder.transform(df)
        
        vif_features_to_keep = self.artifacts["vif_selected_features"]
        if vif_features_to_keep:
            numeric_cols = df.select_dtypes(include=np.number)
            vif_candidates = numeric_cols.loc[:, numeric_cols.nunique() > 2].columns
            cols_to_drop = [col for col in vif_candidates if col not in vif_features_to_keep]
            df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        encoder = self.artifacts["encoder"]
        if encoder:
            encoded_data = encoder.transform(df)
            feature_names = encoder.get_feature_names_out()
            df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

        initial_scaler_artifact = self.artifacts["initial_scaler"]
        if initial_scaler_artifact:
            scaler_transformer = initial_scaler_artifact['transformer']
            scaler_cols = initial_scaler_artifact['columns']
            for col in scaler_cols:
                if col not in df.columns: df[col] = 0
            df_to_scale = df[scaler_cols]
            scaled_data = scaler_transformer.transform(df_to_scale)
            df_scaled = pd.DataFrame(scaled_data, columns=scaler_cols, index=df.index)
            df_unscaled = df.drop(columns=scaler_cols, errors='ignore')
            df = pd.concat([df_unscaled, df_scaled], axis=1)

        # We skip SHAP and Clustering as they are not needed for inference
        
        final_target_encoder = self.artifacts["final_target_encoder"]
        if final_target_encoder:
            df = final_target_encoder.transform(df)

        final_scaler = self.artifacts["final_scaler"]
        if final_scaler:
            df_final = pd.DataFrame(final_scaler.transform(df), columns=df.columns, index=df.index)
        else:
            df_final = df

        logger.info(f"✅ Preprocessing fully complete. Final model-ready shape: {df_final.shape}")
        return df_final

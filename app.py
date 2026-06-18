"""
====================================================================
Machine Learning for Genotype × Environment Interactions
in Nigerian Maize Breeding Programs
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import xgboost as xgb

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="GxE Maize Nigeria — ML Tool",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fa; }

    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #1a5c2a 0%, #2e8b43 50%, #f5a623 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 1.8rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 0.95rem; margin: 0.3rem 0 0; opacity: 0.9; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2e8b43;
    }
    .metric-card h3 { font-size: 1.9rem; color: #2e8b43; margin: 0; font-weight: 700; }
    .metric-card p  { font-size: 0.85rem; color: #666; margin: 0.2rem 0 0; }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        border: 2px solid #2e8b43;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box h2 { color: #1a5c2a; font-size: 2.2rem; margin: 0; }
    .result-box p  { color: #444; margin: 0.3rem 0 0; }

    /* Warning box */
    .warn-box {
        background: #fff8e1;
        border: 2px solid #f5a623;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.8rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a5c2a;
        border-bottom: 2px solid #2e8b43;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a5c2a;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# DATA LOADING AND PREPROCESSING
# ================================================================
@st.cache_data
def load_and_prepare_data():
    """Load CSV, clean, aggregate reps, engineer features."""
    df = pd.read_csv('maize_clean.csv')
    df['region'] = df['region'].str.strip().str.title()

    # Create environment ID
    df['environment_id'] = (df['region'] + '_' +
                             df['environment_condition'] + '_' +
                             df['YEAR'].astype(str))

    # Handle missing values — use assignment instead of inplace (pandas CoW fix)
    for col in ['husk_cover', 'ear_harvested', 'ear_aspect', 'staygreen']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Aggregate replications → 1 row per genotype per environment
    id_cols = ['YEAR', 'region', 'agro_ecological_zone', 'season',
               'environment_condition', 'latitude', 'longitude', 'elevation',
               'rainfall_mm', 'mean_temperature', 'soil_type', 'soil_pH',
               'soil_N_content', 'soil_P_content', 'soil_K_content',
               'Name', 'Pedigree', 'breeding_institution', 'maturity_group',
               'environment_id']
    agg_cols = ['days_to_anthesis', 'Days_to_silking', 'anthesis_silking_interval',
                'plant_height', 'ear_height', 'husk_cover', 'plant_aspect',
                'field_weight', 'ear_harvested', 'ear_per_plant', 'ear_aspect',
                'grain_moisture', 'grain_yield', 'staygreen']
    df_agg = df.groupby(id_cols)[agg_cols].mean().reset_index()

    # Feature engineering
    df_agg['geno_mean_yield'] = df_agg.groupby('Name')['grain_yield'].transform('mean')
    df_agg['env_mean_yield']  = df_agg.groupby('environment_id')['grain_yield'].transform('mean')
    df_agg['loc_mean_yield']  = df_agg.groupby('region')['grain_yield'].transform('mean')
    df_agg['geno_yield_cv']   = df_agg.groupby('Name')['grain_yield'].transform(
                                    lambda x: x.std() / x.mean() * 100)

    # Stress Tolerance Index
    opt = (df_agg[df_agg['environment_condition'] == 'Optimum']
           [['Name', 'region', 'YEAR', 'grain_yield']]
           .rename(columns={'grain_yield': 'yield_optimum'}))
    df_agg = df_agg.merge(opt, on=['Name', 'region', 'YEAR'], how='left')
    mean_opt = (df_agg[df_agg['environment_condition'] == 'Optimum']
                .groupby(['region', 'YEAR'])['grain_yield']
                .mean().rename('mean_opt_yield').reset_index())
    df_agg = df_agg.merge(mean_opt, on=['region', 'YEAR'], how='left')
    df_agg['stress_tolerance_index'] = (
        (df_agg['grain_yield'] * df_agg['yield_optimum']) /
        (df_agg['mean_opt_yield'] ** 2)
    )
    df_agg.loc[df_agg['environment_condition'] == 'Optimum', 'stress_tolerance_index'] = 1.0

    # Encodings
    zone_map      = {'Northern Guinea Savanna': 0,
                     'Southern Guinea Savanna': 1,
                     'Forest\u2013Savanna Transition Zone': 2}
    condition_map = {'Drought': 0, 'Low-N': 1, 'Optimum': 2}
    maturity_map  = {'Extra-Early': 0, 'Early': 1, 'Intermediate': 2, 'late': 3}
    season_map    = {'Dry': 0, 'rainy': 1}
    inst_map      = {'IAR': 0, 'Unilorin': 1, 'CIMMYT and IITA': 2}

    df_agg['zone_enc']        = df_agg['agro_ecological_zone'].map(zone_map)
    df_agg['condition_enc']   = df_agg['environment_condition'].map(condition_map)
    df_agg['maturity_enc']    = df_agg['maturity_group'].map(maturity_map)
    df_agg['season_enc']      = df_agg['season'].map(season_map)
    df_agg['institution_enc'] = df_agg['breeding_institution'].map(inst_map)

    le_geno   = LabelEncoder()
    le_region = LabelEncoder()
    df_agg['geno_enc']   = le_geno.fit_transform(df_agg['Name'])
    df_agg['region_enc'] = le_region.fit_transform(df_agg['region'])

    # FINAL SAFETY IMPUTATION - catches any NaN from merges or aggregation
    FEATURES_CHECK = [
        "geno_enc","institution_enc","maturity_enc","region_enc","zone_enc",
        "condition_enc","season_enc","YEAR","latitude","longitude","elevation",
        "rainfall_mm","mean_temperature","soil_pH","soil_N_content",
        "soil_P_content","soil_K_content","days_to_anthesis","Days_to_silking",
        "anthesis_silking_interval","plant_height","ear_height","husk_cover",
        "plant_aspect","ear_per_plant","ear_aspect","grain_moisture","staygreen",
        "geno_mean_yield","env_mean_yield","loc_mean_yield",
        "stress_tolerance_index","geno_yield_cv"
    ]
    for col in FEATURES_CHECK:
        if col in df_agg.columns and df_agg[col].isnull().sum() > 0:
            df_agg[col] = df_agg[col].fillna(df_agg[col].median())

    return df_agg, le_geno, le_region


@st.cache_resource
def train_models(df_agg):
    """Train all models on full dataset for prediction and feature importance.
    Also trains three XGBoost quantile models for confidence intervals:
      - q10: lower bound (10th percentile)
      - q50: median prediction
      - q90: upper bound (90th percentile)
    Together they give an 80% prediction interval.
    """
    FEATURES = [
        'geno_enc', 'institution_enc', 'maturity_enc',
        'region_enc', 'zone_enc', 'condition_enc', 'season_enc', 'YEAR',
        'latitude', 'longitude', 'elevation',
        'rainfall_mm', 'mean_temperature',
        'soil_pH', 'soil_N_content', 'soil_P_content', 'soil_K_content',
        'days_to_anthesis', 'Days_to_silking', 'anthesis_silking_interval',
        'plant_height', 'ear_height', 'husk_cover', 'plant_aspect',
        'ear_per_plant', 'ear_aspect', 'grain_moisture', 'staygreen',
        'geno_mean_yield', 'env_mean_yield', 'loc_mean_yield',
        'stress_tolerance_index', 'geno_yield_cv'
    ]
    X = df_agg[FEATURES].copy()
    X = X.fillna(X.median())
    y = df_agg['grain_yield']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Standard comparison models
    models = {
        'Ridge':        Ridge(alpha=1.0),
        'Lasso':        Lasso(alpha=1.0, max_iter=5000),
        'SVR':          SVR(kernel='rbf', C=10, epsilon=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10,
                                              min_samples_leaf=5,
                                              random_state=42, n_jobs=-1),
        'XGBoost':      xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8,
                                          random_state=42, n_jobs=-1)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained[name] = model

    # Quantile regression models for 80% prediction interval
    # q10 = lower bound, q50 = median, q90 = upper bound
    quantile_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           objective='reg:quantileerror')
    for q, label in [(0.10, 'q10'), (0.50, 'q50'), (0.90, 'q90')]:
        qmodel = xgb.XGBRegressor(**quantile_params, quantile_alpha=q)
        qmodel.fit(X_scaled, y)
        trained[label] = qmodel

    return trained, scaler, FEATURES, X, y


@st.cache_data
def run_cv_evaluation(df_agg):
    """Run all 5 models × 3 CV schemes. Cached so it only runs once."""
    FEATURES = [
        'geno_enc', 'institution_enc', 'maturity_enc',
        'region_enc', 'zone_enc', 'condition_enc', 'season_enc', 'YEAR',
        'latitude', 'longitude', 'elevation',
        'rainfall_mm', 'mean_temperature',
        'soil_pH', 'soil_N_content', 'soil_P_content', 'soil_K_content',
        'days_to_anthesis', 'Days_to_silking', 'anthesis_silking_interval',
        'plant_height', 'ear_height', 'husk_cover', 'plant_aspect',
        'ear_per_plant', 'ear_aspect', 'grain_moisture', 'staygreen',
        'geno_mean_yield', 'env_mean_yield', 'loc_mean_yield',
        'stress_tolerance_index', 'geno_yield_cv'
    ]
    X = df_agg[FEATURES].copy()
    # Safety: fill any residual NaN with column median
    X = X.fillna(X.median())
    y = df_agg['grain_yield']

    cv_schemes = {
        'CV1 (Random)':     (KFold(n_splits=5, shuffle=True, random_state=42), None),
        'CV2 (Leave-Env)':  (GroupKFold(n_splits=df_agg['environment_id'].nunique()),
                              df_agg['environment_id'].values),
        'CV0 (Leave-Geno)': (GroupKFold(n_splits=10),
                              df_agg['Name'].values)
    }

    model_defs = {
        'Ridge':        Ridge(alpha=1.0),
        'Lasso':        Lasso(alpha=1.0, max_iter=5000),
        'SVR':          SVR(kernel='rbf', C=10, epsilon=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                              random_state=42, n_jobs=-1),
        'XGBoost':      xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                          learning_rate=0.05, random_state=42)
    }

    results = []
    predictions = {}

    for mname, model in model_defs.items():
        predictions[mname] = {}
        for cvname, (cv_obj, groups) in cv_schemes.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            y_pred = cross_val_predict(pipe, X, y, cv=cv_obj, groups=groups)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2   = r2_score(y, y_pred)
            r, _ = pearsonr(y, y_pred)
            results.append({
                'Model': mname, 'CV Scheme': cvname,
                'RMSE (kg/ha)': round(rmse, 1),
                'R²': round(r2, 4),
                'Pearson r': round(r, 4)
            })
            predictions[mname][cvname] = y_pred

    return pd.DataFrame(results), predictions, y


# ================================================================
# SIDEBAR NAVIGATION
# ================================================================
st.sidebar.markdown("""
<div style='text-align:center; padding: 1rem 0;'>
    <h2 style='color:white; font-size:1.4rem;'>🌽 GxE Maize Tool</h2>
    <p style='color:#cce8cc; font-size:0.8rem;'>Nigerian Maize Breeding ML</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📤 Upload Your Data",
     "📊 Data Explorer",
     "📈 Model Validation",
     "🌱 Yield Predictor",
     "🗺️ Location Clustering",
     "ℹ️ About"]
)

# ================================================================
# DATA SOURCE — built-in or uploaded
# ================================================================
# Session state tracks whether user has uploaded their own data
if 'uploaded_df'       not in st.session_state: st.session_state.uploaded_df       = None
if 'uploaded_le_geno'  not in st.session_state: st.session_state.uploaded_le_geno  = None
if 'uploaded_le_region'not in st.session_state: st.session_state.uploaded_le_region= None
if 'use_uploaded'      not in st.session_state: st.session_state.use_uploaded       = False
if 'upload_label'      not in st.session_state: st.session_state.upload_label       = "Built-in dataset"

# Show which dataset is active in sidebar
st.sidebar.markdown("---")
if st.session_state.use_uploaded:
    st.sidebar.markdown(f"""
    <div style='background:#2e8b43;border-radius:6px;padding:0.6rem;margin-top:0.5rem;'>
        <p style='color:white;font-size:0.8rem;margin:0;'>
        📂 Active: <b>{st.session_state.upload_label}</b>
        </p>
    </div>""", unsafe_allow_html=True)
    if st.sidebar.button("↩ Switch to built-in data"):
        st.session_state.use_uploaded = False
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style='background:#555;border-radius:6px;padding:0.6rem;margin-top:0.5rem;'>
        <p style='color:#ccc;font-size:0.8rem;margin:0;'>
        📂 Active: <b>Built-in dataset</b>
        </p>
    </div>""", unsafe_allow_html=True)

# ================================================================
# LOAD DATA — built-in or uploaded
# ================================================================
try:
    df_agg_builtin, le_geno_builtin, le_region_builtin = load_and_prepare_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    df_agg_builtin = None

# Active dataset selection
if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
    df_agg    = st.session_state.uploaded_df
    le_geno   = st.session_state.uploaded_le_geno
    le_region = st.session_state.uploaded_le_region
    active_label = st.session_state.upload_label
else:
    if data_loaded:
        df_agg    = df_agg_builtin
        le_geno   = le_geno_builtin
        le_region = le_region_builtin
    active_label = "Built-in dataset (Nigeria, 2020–2022)"


# ================================================================
# UPLOAD PAGE — process and retrain on user data
# ================================================================
if page == "📤 Upload Your Data":

    st.markdown("## 📤 Upload Your Own Trial Data")
    st.markdown("""
    Upload your own multilocational maize trial dataset. The app will process it,
    retrain the prediction model on your genotypes
    """)

    # ---------------------------------------------------------------
    # STEP 0 — Download template
    # ---------------------------------------------------------------
    st.markdown("<div class='section-title'>Step 1 — Download the Template</div>",
                unsafe_allow_html=True)
    st.markdown("""
    Your CSV must contain at minimum:
    - **Required:** `region`, `environment_condition`, `grain_yield`, `Name`
    - **Recommended:** All trait columns for best prediction accuracy
    - **Optional:** Environmental covariates (latitude, longitude, rainfall etc.)

    Download the template below, fill it with your data, then upload it.
    Missing optional columns will be filled with dataset means automatically.
    """)

    try:
        with open('maize_trial_template.csv', 'rb') as f:
            st.download_button(
                label="📥 Download CSV Template",
                data=f,
                file_name="maize_trial_template.csv",
                mime="text/csv",
                type="secondary"
            )
    except FileNotFoundError:
        st.warning("Template file not found. Place maize_trial_template.csv "
                   "in the same folder as app.py")

    st.markdown("---")

    # ---------------------------------------------------------------
    # STEP 1 — Upload CSV
    # ---------------------------------------------------------------
    st.markdown("<div class='section-title'>Step 2 — Upload Your CSV File</div>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your trial data CSV file",
        type=['csv'],
        help="Must contain at minimum: region, environment_condition, "
             "grain_yield, Name columns."
    )

    if uploaded_file is not None:
        try:
            user_df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: **{uploaded_file.name}** — "
                       f"{len(user_df_raw):,} rows × {user_df_raw.shape[1]} columns")

            st.markdown("<div class='section-title'>Preview of Uploaded Data</div>",
                        unsafe_allow_html=True)
            st.dataframe(user_df_raw.head(5), use_container_width=True)

            # -----------------------------------------------------------
            # STEP 2 — Column mapping
            # -----------------------------------------------------------
            st.markdown("---")
            st.markdown("<div class='section-title'>Step 3 — Map Your Columns</div>",
                        unsafe_allow_html=True)
            st.markdown("""
            Match your column names to the expected variables.
            If your column names already match the template exactly,
            the dropdowns will be pre-selected automatically.
            """)

            uploaded_cols = ['(not in my data)'] + list(user_df_raw.columns)

            # Required columns
            st.markdown("**Required columns** (must be mapped):")
            req_col1, req_col2 = st.columns(2)
            with req_col1:
                def best_match(target, cols):
                    """Find best matching column name."""
                    exact = [c for c in cols if c.lower() == target.lower()]
                    if exact: return exact[0]
                    partial = [c for c in cols if target.lower() in c.lower()
                               or c.lower() in target.lower()]
                    return partial[0] if partial else '(not in my data)'

                map_region = st.selectbox(
                    "Location / Region column",
                    uploaded_cols,
                    index=uploaded_cols.index(
                        best_match('region', user_df_raw.columns)),
                    key='map_region'
                )
                map_condition = st.selectbox(
                    "Stress Condition column",
                    uploaded_cols,
                    index=uploaded_cols.index(
                        best_match('environment_condition', user_df_raw.columns)),
                    key='map_condition'
                )
            with req_col2:
                map_yield = st.selectbox(
                    "Grain Yield column",
                    uploaded_cols,
                    index=uploaded_cols.index(
                        best_match('grain_yield', user_df_raw.columns)),
                    key='map_yield'
                )
                map_name = st.selectbox(
                    "Genotype Name column",
                    uploaded_cols,
                    index=uploaded_cols.index(
                        best_match('Name', user_df_raw.columns)),
                    key='map_name'
                )

            # Optional columns
            with st.expander("Optional column mappings (recommended for accuracy)"):
                opt_cols = {
                    'YEAR':                    'Year column',
                    'agro_ecological_zone':    'Agroecological Zone column',
                    'season':                  'Season column',
                    'latitude':                'Latitude column',
                    'longitude':               'Longitude column',
                    'elevation':               'Elevation column',
                    'rainfall_mm':             'Rainfall (mm) column',
                    'mean_temperature':        'Mean Temperature column',
                    'soil_pH':                 'Soil pH column',
                    'soil_N_content':          'Soil N content column',
                    'soil_P_content':          'Soil P content column',
                    'soil_K_content':          'Soil K content column',
                    'breeding_institution':    'Breeding Institution column',
                    'maturity_group':          'Maturity Group column',
                    'Pedigree':                'Pedigree column',
                    'rep':                     'Replication column',
                    'days_to_anthesis':        'Days to Anthesis column',
                    'Days_to_silking':         'Days to Silking column',
                    'anthesis_silking_interval':'ASI column',
                    'plant_height':            'Plant Height column',
                    'ear_height':              'Ear Height column',
                    'husk_cover':              'Husk Cover column',
                    'plant_aspect':            'Plant Aspect column',
                    'ear_per_plant':           'Ears per Plant column',
                    'ear_aspect':              'Ear Aspect column',
                    'grain_moisture':          'Grain Moisture column',
                    'staygreen':               'Staygreen column',
                }
                opt_mappings = {}
                ocols = list(opt_cols.items())
                for i in range(0, len(ocols), 2):
                    c1, c2 = st.columns(2)
                    for col_idx, (expected, label) in enumerate(ocols[i:i+2]):
                        col_widget = c1 if col_idx == 0 else c2
                        with col_widget:
                            opt_mappings[expected] = st.selectbox(
                                label, uploaded_cols,
                                index=uploaded_cols.index(
                                    best_match(expected, user_df_raw.columns)),
                                key=f'map_{expected}'
                            )

            # -----------------------------------------------------------
            # STEP 3 — Validate and Process
            # -----------------------------------------------------------
            st.markdown("---")
            st.markdown("<div class='section-title'>Step 4 — Process and Train</div>",
                        unsafe_allow_html=True)

            # Check required mappings
            required_mapped = all([
                map_region    != '(not in my data)',
                map_condition != '(not in my data)',
                map_yield     != '(not in my data)',
                map_name      != '(not in my data)'
            ])

            if not required_mapped:
                st.error("⚠️ Please map all 4 required columns before processing.")
            else:
                # Show what we detected
                cond_values = user_df_raw[map_condition].unique().tolist()
                reg_values  = user_df_raw[map_region].unique().tolist()
                st.info(f"""
                **Ready to process:**
                - **{user_df_raw[map_name].nunique()}** unique genotypes
                - **{len(reg_values)}** locations: {', '.join(str(r) for r in reg_values[:8])}
                - **Stress conditions found:** {', '.join(str(c) for c in cond_values)}
                - **Rows:** {len(user_df_raw):,}
                """)

                if st.button("⚙️ Process Data and Train Model",
                             type="primary", use_container_width=True):
                    with st.spinner("Processing your data and training models... "
                                    "This may take 2–3 minutes."):

                        # ---- BUILD STANDARDISED DATAFRAME ----
                        df_user = pd.DataFrame()

                        # Required columns
                        df_user['region']               = user_df_raw[map_region].astype(str).str.strip().str.title()
                        df_user['environment_condition'] = user_df_raw[map_condition].astype(str).str.strip()
                        df_user['grain_yield']           = pd.to_numeric(user_df_raw[map_yield], errors='coerce')
                        df_user['Name']                  = user_df_raw[map_name].astype(str).str.strip()

                        # Optional columns — use mapping or fill with defaults
                        DEFAULTS = {
                            'YEAR': 2024, 'agro_ecological_zone': 'Southern Guinea Savanna',
                            'season': 'rainy', 'latitude': 9.0, 'longitude': 7.0,
                            'elevation': 400, 'rainfall_mm': 900, 'mean_temperature': 26.0,
                            'soil_pH': 6.2, 'soil_N_content': 0.11, 'soil_P_content': 10,
                            'soil_K_content': 130, 'breeding_institution': 'Unknown',
                            'maturity_group': 'Early', 'Pedigree': 'Unknown',
                            'rep': 1, 'days_to_anthesis': 62, 'Days_to_silking': 65,
                            'anthesis_silking_interval': 3, 'plant_height': 190,
                            'ear_height': 95, 'husk_cover': 2.5, 'plant_aspect': 2.5,
                            'ear_per_plant': 0.95, 'ear_aspect': 2.5,
                            'grain_moisture': 13.0, 'staygreen': 3.0,
                        }

                        for expected_col, default_val in DEFAULTS.items():
                            mapped = opt_mappings.get(expected_col, '(not in my data)')
                            if mapped != '(not in my data)' and mapped in user_df_raw.columns:
                                df_user[expected_col] = user_df_raw[mapped]
                            else:
                                df_user[expected_col] = default_val

                        # Drop rows with missing yield
                        df_user = df_user.dropna(subset=['grain_yield'])
                        df_user = df_user[df_user['grain_yield'] > 0]

                        # Numeric coercion for all trait columns
                        numeric_cols = ['YEAR','latitude','longitude','elevation',
                                        'rainfall_mm','mean_temperature','soil_pH',
                                        'soil_N_content','soil_P_content','soil_K_content',
                                        'days_to_anthesis','Days_to_silking',
                                        'anthesis_silking_interval','plant_height',
                                        'ear_height','husk_cover','plant_aspect',
                                        'ear_per_plant','ear_aspect','grain_moisture',
                                        'staygreen']
                        for col in numeric_cols:
                            df_user[col] = pd.to_numeric(df_user[col], errors='coerce')

                        # Create environment ID
                        df_user['environment_id'] = (
                            df_user['region'] + '_' +
                            df_user['environment_condition'] + '_' +
                            df_user['YEAR'].astype(str)
                        )

                        # Aggregate replications if rep column available
                        id_cols_u = ['YEAR','region','agro_ecological_zone','season',
                                     'environment_condition','latitude','longitude',
                                     'elevation','rainfall_mm','mean_temperature',
                                     'soil_pH','soil_N_content','soil_P_content',
                                     'soil_K_content','Name','Pedigree',
                                     'breeding_institution','maturity_group',
                                     'environment_id']
                        agg_cols_u = ['days_to_anthesis','Days_to_silking',
                                      'anthesis_silking_interval','plant_height',
                                      'ear_height','husk_cover','plant_aspect',
                                      'ear_per_plant','ear_aspect','grain_moisture',
                                      'grain_yield','staygreen']
                        df_agg_u = df_user.groupby(id_cols_u)[agg_cols_u].mean().reset_index()

                        # Fill NaN from aggregation
                        for col in agg_cols_u:
                            df_agg_u[col] = df_agg_u[col].fillna(df_agg_u[col].median())

                        # Feature engineering
                        df_agg_u['geno_mean_yield'] = df_agg_u.groupby('Name')['grain_yield'].transform('mean')
                        df_agg_u['env_mean_yield']  = df_agg_u.groupby('environment_id')['grain_yield'].transform('mean')
                        df_agg_u['loc_mean_yield']  = df_agg_u.groupby('region')['grain_yield'].transform('mean')
                        df_agg_u['geno_yield_cv']   = df_agg_u.groupby('Name')['grain_yield'].transform(
                            lambda x: x.std()/x.mean()*100 if x.mean() > 0 else 0)

                        # STI
                        opt_u = (df_agg_u[df_agg_u['environment_condition'].str.lower() == 'optimum']
                                 [['Name','region','YEAR','grain_yield']]
                                 .rename(columns={'grain_yield':'yield_optimum'}))
                        if len(opt_u) > 0:
                            df_agg_u = df_agg_u.merge(opt_u, on=['Name','region','YEAR'], how='left')
                            mean_opt_u = (df_agg_u[df_agg_u['environment_condition'].str.lower()=='optimum']
                                         .groupby(['region','YEAR'])['grain_yield'].mean()
                                         .rename('mean_opt_yield').reset_index())
                            df_agg_u = df_agg_u.merge(mean_opt_u, on=['region','YEAR'], how='left')
                            df_agg_u['stress_tolerance_index'] = (
                                (df_agg_u['grain_yield'] * df_agg_u['yield_optimum']) /
                                (df_agg_u['mean_opt_yield']**2)
                            )
                            df_agg_u.loc[
                                df_agg_u['environment_condition'].str.lower()=='optimum',
                                'stress_tolerance_index'
                            ] = 1.0
                        else:
                            df_agg_u['stress_tolerance_index'] = 1.0
                        df_agg_u['stress_tolerance_index'] = df_agg_u['stress_tolerance_index'].fillna(1.0)

                        # Encodings
                        zone_map_u = {'Northern Guinea Savanna':0,
                                      'Southern Guinea Savanna':1,
                                      'Forest\u2013Savanna Transition Zone':2,
                                      'Forest-Savanna Transition Zone':2}
                        cond_vals_u = df_agg_u['environment_condition'].unique().tolist()
                        cond_map_u  = {c: i for i,c in enumerate(sorted(cond_vals_u))}
                        mat_map_u   = {'Extra-Early':0,'Early':1,'Intermediate':2,'late':3,'Late':3}
                        seas_map_u  = {'Dry':0,'rainy':1,'Rainy':1,'dry':0}
                        inst_vals_u = df_agg_u['breeding_institution'].unique().tolist()
                        inst_map_u  = {c: i for i,c in enumerate(sorted(inst_vals_u))}

                        df_agg_u['zone_enc']        = df_agg_u['agro_ecological_zone'].map(zone_map_u).fillna(1).astype(int)
                        df_agg_u['condition_enc']   = df_agg_u['environment_condition'].map(cond_map_u).fillna(0).astype(int)
                        df_agg_u['maturity_enc']    = df_agg_u['maturity_group'].map(mat_map_u).fillna(1).astype(int)
                        df_agg_u['season_enc']      = df_agg_u['season'].map(seas_map_u).fillna(1).astype(int)
                        df_agg_u['institution_enc'] = df_agg_u['breeding_institution'].map(inst_map_u).fillna(0).astype(int)

                        le_geno_u   = LabelEncoder()
                        le_region_u = LabelEncoder()
                        df_agg_u['geno_enc']   = le_geno_u.fit_transform(df_agg_u['Name'])
                        df_agg_u['region_enc'] = le_region_u.fit_transform(df_agg_u['region'])

                        # Final safety fill
                        FEATURES_U = [
                            'geno_enc','institution_enc','maturity_enc','region_enc',
                            'zone_enc','condition_enc','season_enc','YEAR',
                            'latitude','longitude','elevation','rainfall_mm',
                            'mean_temperature','soil_pH','soil_N_content',
                            'soil_P_content','soil_K_content','days_to_anthesis',
                            'Days_to_silking','anthesis_silking_interval','plant_height',
                            'ear_height','husk_cover','plant_aspect','ear_per_plant',
                            'ear_aspect','grain_moisture','staygreen','geno_mean_yield',
                            'env_mean_yield','loc_mean_yield','stress_tolerance_index',
                            'geno_yield_cv'
                        ]
                        for col in FEATURES_U:
                            if col in df_agg_u.columns:
                                df_agg_u[col] = pd.to_numeric(df_agg_u[col], errors='coerce').fillna(df_agg_u[col].median() if df_agg_u[col].dtype != object else 0)

                        # Store encoding maps in df for use by predictor pages
                        df_agg_u.attrs['cond_map']  = cond_map_u
                        df_agg_u.attrs['inst_map']  = inst_map_u
                        df_agg_u.attrs['zone_map']  = zone_map_u

                        # Train models on uploaded data
                        X_u = df_agg_u[FEATURES_U].fillna(0)
                        y_u = df_agg_u['grain_yield']

                        scaler_u = StandardScaler()
                        X_u_scaled = scaler_u.fit_transform(X_u)

                        xgb_u = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                                   learning_rate=0.05, random_state=42)
                        xgb_u.fit(X_u_scaled, y_u)

                        # Quantile models
                        for q, label in [(0.10,'q10'),(0.50,'q50'),(0.90,'q90')]:
                            qm = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                                   learning_rate=0.05, random_state=42,
                                                   objective='reg:quantileerror',
                                                   quantile_alpha=q)
                            qm.fit(X_u_scaled, y_u)
                            df_agg_u.attrs[f'model_{label}'] = qm

                        df_agg_u.attrs['model_xgb']   = xgb_u
                        df_agg_u.attrs['scaler']       = scaler_u
                        df_agg_u.attrs['features']     = FEATURES_U
                        df_agg_u.attrs['filename']     = uploaded_file.name

                        # Save to session state
                        st.session_state.uploaded_df        = df_agg_u
                        st.session_state.uploaded_le_geno   = le_geno_u
                        st.session_state.uploaded_le_region = le_region_u
                        st.session_state.use_uploaded       = True
                        st.session_state.upload_label       = uploaded_file.name

                    st.success(f"""
                    ✅ **Processing complete!**
                    - **{df_agg_u['Name'].nunique()}** genotypes processed
                    - **{df_agg_u['region'].nunique()}** locations
                    - **{len(df_agg_u):,}** observations (after rep aggregation)
                    - Model trained and ready

                    👈 Use the sidebar to navigate to any page.
                    All pages now use **{uploaded_file.name}**.
                    """)

                    # Quick summary stats
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    col_s1.metric("Genotypes", df_agg_u['Name'].nunique())
                    col_s2.metric("Locations",  df_agg_u['region'].nunique())
                    col_s3.metric("Observations", f"{len(df_agg_u):,}")
                    col_s4.metric("Mean Yield", f"{df_agg_u['grain_yield'].mean():,.0f} kg/ha")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Make sure your file is a valid CSV and matches the template format.")

    else:
        st.info("👆 Upload a CSV file to get started, or download the template first.")

        # Show what the template looks like
        try:
            template_preview = pd.read_csv('maize_trial_template.csv')
            st.markdown("<div class='section-title'>Template Preview</div>",
                        unsafe_allow_html=True)
            st.dataframe(template_preview, use_container_width=True)
            st.caption("Fill this template with your trial data. "
                       "Required columns: region, environment_condition, "
                       "grain_yield, Name.")
        except FileNotFoundError:
            pass


# ================================================================
# PAGE 1 — OVERVIEW DASHBOARD
# ================================================================
elif page == "🏠 Overview":

    st.markdown("""
    <div class='main-header'>
        <h1>🌽 Machine Learning for Genotype × Environment Interactions</h1>
        <p>Nigerian Maize Breeding Programs · Instituto Superior de Agronomia, ULisboa · 2025–2026</p>
    </div>
    """, unsafe_allow_html=True)

    if df_agg is None:
        st.error("⚠️ maize_clean.csv not found. Place it in the same folder as app.py and restart.")
        st.stop()

    # Active dataset banner
    if st.session_state.use_uploaded:
        st.info(f"📂 Active dataset: **{st.session_state.upload_label}**")

    # ---------------------------------------------------------------
    # ABOUT THIS PROJECT
    # ---------------------------------------------------------------
    st.markdown("<div class='section-title'>About This Project</div>",
                unsafe_allow_html=True)
    st.markdown("""
   
    This tool applies **machine learning** to multilocational maize trial data to:

    - 🔮 **Predict** grain yield of genotypes in untested environments
    - 🧬 **Predict** grain yield of new genotypes in tested or untested environments
    - 📍 **Identify** which trial locations carry redundant environmental information
    - 🌱 **Rank** genotypes by stress tolerance and broad adaptation
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # DATASET SUMMARY — dynamic (updates with uploaded data)
    # ---------------------------------------------------------------
    st.markdown("<div class='section-title'>Active Dataset Summary</div>",
                unsafe_allow_html=True)

    year_range   = f"{int(df_agg['YEAR'].min())}–{int(df_agg['YEAR'].max())}"
    locations    = ', '.join(sorted(df_agg['region'].unique()))
    conditions   = ', '.join(sorted(df_agg['environment_condition'].unique()))
    institutions = ', '.join(sorted(df_agg['breeding_institution'].unique())) \
                   if df_agg['breeding_institution'].nunique() <= 6 else \
                   f"{df_agg['breeding_institution'].nunique()} institutions"

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    for col, val, label in [
        (col_m1, f"{df_agg['Name'].nunique():,}",        "Genotypes"),
        (col_m2, f"{df_agg['region'].nunique()}",         "Locations"),
        (col_m3, f"{df_agg['environment_id'].nunique()}", "Environments"),
        (col_m4, f"{df_agg['YEAR'].nunique()}",           f"Years ({year_range})"),
        (col_m5, f"{len(df_agg):,}",                      "Observations"),
    ]:
        col.markdown(f"""
        <div class='metric-card'>
            <h3>{val}</h3><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        <div style='background:white;border-radius:8px;padding:1rem;
             box-shadow:0 2px 6px rgba(0,0,0,0.08);'>
            <p style='margin:0.2rem 0;'><b>📍 Locations:</b> {locations}</p>
            <p style='margin:0.2rem 0;'><b>⚗️ Stress Conditions:</b> {conditions}</p>
            <p style='margin:0.2rem 0;'><b>🏛️ Institutions:</b> {institutions}</p>
        </div>""", unsafe_allow_html=True)
    with col_d2:
        mean_y  = df_agg['grain_yield'].mean()
        max_y   = df_agg['grain_yield'].max()
        min_y   = df_agg['grain_yield'].min()
        st.markdown(f"""
        <div style='background:white;border-radius:8px;padding:1rem;
             box-shadow:0 2px 6px rgba(0,0,0,0.08);'>
            <p style='margin:0.2rem 0;'><b>📊 Mean Yield:</b> {mean_y:,.0f} kg/ha</p>
            <p style='margin:0.2rem 0;'><b>📈 Maximum Yield:</b> {max_y:,.0f} kg/ha</p>
            <p style='margin:0.2rem 0;'><b>📉 Minimum Yield:</b> {min_y:,.0f} kg/ha</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # HOW TO USE THIS TOOL
    # ---------------------------------------------------------------
    st.markdown("<div class='section-title'>How to Use This Tool</div>",
                unsafe_allow_html=True)

    nav_items = [
        ("📤 Upload Your Data",
         "Upload your own multilocational trial CSV to run all analyses on your data."),
        ("📊 Data Explorer",
         "Explore trait distributions, correlations with yield, and compare genotype "
         "performance across stress conditions and locations."),
        ("📈 Model Validation",
         "Review the scientific evidence that the prediction model is accurate — "
         "RMSE, R², and observed vs predicted comparisons across 5 models and 3 "
         "cross-validation schemes."),
        ("🌱 Yield Predictor",
         "Predict grain yield for any genotype in a known location, a new untested "
         "location, or for a brand new genotype using its agronomic trait measurements."),
        ("🗺️ Trial Network Optimisation",
         "Identify which trial locations are environmentally similar or redundant, "
         "and quantify the cost saving of removing each location from the network."),
    ]

    for icon_title, description in nav_items:
        st.markdown(f"""
        <div style='background:white;border-left:4px solid #2e8b43;border-radius:6px;
             padding:0.7rem 1rem;margin-bottom:0.5rem;
             box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
            <b>{icon_title}</b><br>
            <span style='color:#555;font-size:0.9rem;'>{description}</span>
        </div>""", unsafe_allow_html=True)


# ================================================================
# PAGE 2 — DATA EXPLORER
# ================================================================
elif page == "📊 Data Explorer":

    st.markdown("## 📊 Data Explorer")

    if df_agg is None:
        st.error("⚠️ maize_clean.csv not found.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Trait Distributions", "Correlation Analysis",
                                  "Genotype Performance"])

    with tab1:
        st.markdown("<div class='section-title'>Explore Trait Distributions</div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            trait = st.selectbox("Select trait to explore", [
                'grain_yield', 'days_to_anthesis', 'Days_to_silking',
                'anthesis_silking_interval', 'plant_height', 'ear_height',
                'ear_per_plant', 'grain_moisture', 'staygreen',
                'stress_tolerance_index'
            ])
        with col2:
            group_by = st.selectbox("Group by", [
                'environment_condition', 'region', 'agro_ecological_zone',
                'maturity_group', 'breeding_institution'
            ])

        fig = px.violin(
            df_agg, x=group_by, y=trait, color=group_by,
            box=True, points=False,
            title=f'{trait} by {group_by}',
            labels={trait: trait.replace('_', ' ').title()}
        )
        fig.update_layout(showlegend=False, plot_bgcolor='white',
                          paper_bgcolor='white', height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown("<div class='section-title'>Summary Statistics</div>",
                    unsafe_allow_html=True)
        stats = df_agg.groupby(group_by)[trait].agg(
            ['mean', 'std', 'min', 'max', 'count']
        ).round(2).reset_index()
        stats.columns = [group_by, 'Mean', 'Std Dev', 'Min', 'Max', 'Count']
        st.dataframe(stats, hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Trait Correlation with Grain Yield</div>",
                    unsafe_allow_html=True)

        numeric_cols = ['days_to_anthesis', 'Days_to_silking',
                        'anthesis_silking_interval', 'plant_height', 'ear_height',
                        'husk_cover', 'plant_aspect', 'ear_per_plant', 'ear_aspect',
                        'grain_moisture', 'staygreen', 'stress_tolerance_index',
                        'geno_mean_yield', 'geno_yield_cv', 'grain_yield']

        corr = df_agg[numeric_cols].corr()['grain_yield'].drop('grain_yield').sort_values()

        colors = ['#c0392b' if v < 0 else '#2e8b43' for v in corr.values]
        fig = go.Figure(go.Bar(
            x=corr.values, y=corr.index,
            orientation='h',
            marker_color=colors
        ))
        fig.update_layout(
            title='Pearson Correlation with Grain Yield',
            xaxis_title='Correlation Coefficient',
            plot_bgcolor='white', paper_bgcolor='white',
            height=500
        )
        fig.add_vline(x=0, line_dash='dash', line_color='black', line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **How to read this chart:**
        - 🟢 **Green bars (positive):** trait increases → yield increases
        - 🔴 **Red bars (negative):** trait increases → yield decreases
        - Longer bar = stronger relationship with yield
        """)

    with tab3:
        st.markdown("<div class='section-title'>Genotype Performance Across Environments</div>",
                    unsafe_allow_html=True)

        selected_genos = st.multiselect(
            "Select genotypes to compare (up to 10)",
            sorted(df_agg['Name'].unique()),
            default=sorted(df_agg['Name'].unique())[:5]
        )

        if selected_genos:
            subset = df_agg[df_agg['Name'].isin(selected_genos)]
            fig = px.bar(
                subset.groupby(['Name', 'environment_condition'])['grain_yield']
                       .mean().reset_index(),
                x='Name', y='grain_yield',
                color='environment_condition',
                barmode='group',
                color_discrete_map={'Optimum': '#2e8b43',
                                    'Low-N': '#f5a623',
                                    'Drought': '#c0392b'},
                labels={'grain_yield': 'Mean Yield (kg/ha)',
                        'Name': 'Genotype',
                        'environment_condition': 'Stress Condition'},
                title='Mean Yield by Genotype and Stress Condition'
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              height=420, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# PAGE 3 — MODEL RESULTS
# ================================================================
elif page == "📈 Model Validation":

    st.markdown("## 📈 Model Validation")

    if df_agg is None:
        st.error("⚠️ maize_clean.csv not found.")
        st.stop()

    st.markdown("""
    This page provides the **scientific evidence** that the prediction model is accurate
    and reliable. It is intended for researchers, reviewers, and anyone who wants to
    understand how the tool works before trusting its predictions.

    Five machine learning models are compared across three cross-validation schemes
    that simulate increasingly realistic breeding scenarios.
    """)

    st.info("""
    **Cross-Validation Schemes:**
    - **CV1 (Random):** Random 5-fold split — general predictive accuracy under ideal conditions
    - **CV2 (Leave-Env):** Leave one environment out entirely — simulates predicting in a new, untested location
    - **CV0 (Leave-Geno):** Leave one genotype group out — simulates predicting a brand new variety never tested anywhere
    """)

    # ---------------------------------------------------------------
    # DECIDE: use hardcoded results (built-in) or recompute (uploaded)
    # ---------------------------------------------------------------
    if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
        # Uploaded data — run CV evaluation on their data
        st.warning("""
        ⏳ Running cross-validation on your uploaded dataset.
        This trains 5 models × 3 CV schemes and may take several minutes.
        This is necessary because your data is new — these results do not exist yet.
        """)
        with st.spinner("Training and evaluating all models on your data..."):
            results_df, predictions, y_true = run_cv_evaluation(df_agg)
        st.success("✅ Evaluation complete!")
        use_hardcoded = False

    else:
        # Built-in dataset — use pre-computed results from the study
        # These were computed once in Google Colab and are fixed scientific findings
        # No retraining needed — results never change for this dataset
        st.success("✅ Showing pre-computed results from the study "
                   "(Nigeria dataset, 2020–2022)")
        use_hardcoded = True

        # ---- HARDCODED RESULTS FROM COLAB RUN ----
        # Source: GxE_Maize_Nigeria.ipynb, run June 2026
        # Screenshots: Images 7, 8, 10, 11 in project documentation
        hardcoded_results = [
            # Model,          CV Scheme,           RMSE,    R²,     r
            ('Ridge',        'CV1 (Random)',       1146.6,  0.587,  0.766),
            ('Ridge',        'CV2 (Leave-Env)',    1177.5,  0.565,  0.752),
            ('Ridge',        'CV0 (Leave-Geno)',   1145.1,  0.588,  0.767),
            ('Lasso',        'CV1 (Random)',       1146.9,  0.587,  0.766),
            ('Lasso',        'CV2 (Leave-Env)',    1174.8,  0.567,  0.753),
            ('Lasso',        'CV0 (Leave-Geno)',   1145.5,  0.588,  0.767),
            ('SVR',          'CV1 (Random)',       1322.6,  0.451,  0.672),
            ('SVR',          'CV2 (Leave-Env)',    1336.0,  0.440,  0.663),
            ('SVR',          'CV0 (Leave-Geno)',   1303.0,  0.467,  0.683),
            ('RandomForest', 'CV1 (Random)',        994.0,  0.690,  0.831),
            ('RandomForest', 'CV2 (Leave-Env)',    1052.0,  0.653,  0.808),
            ('RandomForest', 'CV0 (Leave-Geno)',    989.8,  0.693,  0.832),
            ('XGBoost',      'CV1 (Random)',        959.1,  0.711,  0.843),
            ('XGBoost',      'CV2 (Leave-Env)',    1023.7,  0.671,  0.819),
            ('XGBoost',      'CV0 (Leave-Geno)',    948.7,  0.718,  0.847),
        ]
        results_df = pd.DataFrame(hardcoded_results,
                                   columns=['Model','CV Scheme',
                                            'RMSE (kg/ha)','R²','Pearson r'])

        # Hardcoded feature importances from XGBoost (Image 11)
        hardcoded_importance = {
            'Days_to_silking':            0.1782,
            'elevation':                  0.1750,
            'ear_aspect':                 0.1250,
            'staygreen':                  0.1217,
            'stress_tolerance_index':     0.1050,
            'env_mean_yield':             0.0420,
            'latitude':                   0.0380,
            'plant_aspect':               0.0320,
            'ear_per_plant':              0.0316,
            'soil_pH':                    0.0280,
            'soil_N_content':             0.0260,
            'husk_cover':                 0.0240,
            'geno_yield_cv':              0.0220,
            'ear_height':                 0.0190,
            'geno_mean_yield':            0.0180,
        }
        xgb_imp = pd.Series(hardcoded_importance).sort_values(ascending=True)
        predictions = None
        y_true = None

    # ---------------------------------------------------------------
    # DISPLAY RESULTS — same layout for both hardcoded and computed
    # ---------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Performance Table",
                                  "📈 Visual Comparison",
                                  "🔵 Observed vs Predicted"])

    with tab1:
        st.markdown("<div class='section-title'>RMSE (kg/ha) — Lower is Better</div>",
                    unsafe_allow_html=True)
        rmse_pivot = results_df.pivot(
            index='Model', columns='CV Scheme', values='RMSE (kg/ha)'
        ).round(1)
        st.dataframe(rmse_pivot.style.background_gradient(cmap='RdYlGn_r', axis=None),
                     use_container_width=True)

        st.markdown("<div class='section-title'>R² — Higher is Better</div>",
                    unsafe_allow_html=True)
        r2_pivot = results_df.pivot(
            index='Model', columns='CV Scheme', values='R²'
        ).round(4)
        st.dataframe(r2_pivot.style.background_gradient(cmap='RdYlGn', axis=None),
                     use_container_width=True)

        st.markdown("<div class='section-title'>Pearson r — Higher is Better</div>",
                    unsafe_allow_html=True)
        r_pivot = results_df.pivot(
            index='Model', columns='CV Scheme', values='Pearson r'
        ).round(4)
        st.dataframe(r_pivot.style.background_gradient(cmap='RdYlGn', axis=None),
                     use_container_width=True)

        # Key findings box
        best_cv2 = results_df[results_df['CV Scheme']=='CV2 (Leave-Env)'].sort_values('RMSE (kg/ha)').iloc[0]
        st.markdown(f"""
        <div class='result-box' style='text-align:left; padding:1rem;'>
            <b>🏆 Key Finding:</b> <b>{best_cv2['Model']}</b> is the best model
            across all CV schemes.<br>
            For CV2 (predicting completely new environments):
            <b>RMSE = {best_cv2['RMSE (kg/ha)']:.0f} kg/ha,
            R² = {best_cv2['R²']:.3f},
            r = {best_cv2['Pearson r']:.3f}</b><br>
            <span style='font-size:0.85rem; color:#555;'>
            This means the model explains {best_cv2['R²']*100:.1f}% of yield variance
            even for locations it has never seen — validating its use as a
            prediction tool for new environments.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = px.bar(
                results_df, x='Model', y='RMSE (kg/ha)',
                color='CV Scheme', barmode='group',
                color_discrete_sequence=['#2e8b43', '#f5a623', '#c0392b'],
                title='Model Comparison — RMSE (kg/ha)<br><sup>Lower is better</sup>'
            )
            fig_rmse.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                    height=420, xaxis_tickangle=-30)
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            fig_r2 = px.bar(
                results_df, x='Model', y='R²',
                color='CV Scheme', barmode='group',
                color_discrete_sequence=['#2e8b43', '#f5a623', '#c0392b'],
                title='Model Comparison — R²<br><sup>Higher is better</sup>'
            )
            fig_r2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                  height=420, xaxis_tickangle=-30)
            st.plotly_chart(fig_r2, use_container_width=True)

        # Feature importance
        st.markdown("<div class='section-title'>Feature Importance — XGBoost</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        Which input variables does the model rely on most to make predictions?
        Higher importance = removing this feature would hurt accuracy most.
        """)

        if use_hardcoded:
            imp_series = xgb_imp
        else:
            trained_models_fi, _, FEATURES_fi, _, _ = train_models(df_agg)
            imp_series = pd.Series(
                trained_models_fi['XGBoost'].feature_importances_,
                index=FEATURES_fi
            ).sort_values(ascending=True).tail(15)

        fig_imp = px.bar(
            x=imp_series.values, y=imp_series.index,
            orientation='h',
            color=imp_series.values,
            color_continuous_scale='Greens',
            labels={'x': 'Importance Score', 'y': 'Feature'},
            title='XGBoost — Top 15 Most Important Features'
        )
        fig_imp.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               height=500, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        # Plain language interpretation
        top3 = imp_series.tail(3).index.tolist()[::-1]
        st.markdown(f"""
        <div class='warn-box'>
        <b>What this means agronomically:</b><br>
        The three most important predictors are
        <b>{top3[0]}</b>, <b>{top3[1]}</b>, and <b>{top3[2]}</b>.
        The model discovered these patterns from data alone — without being told
        anything about maize biology. This aligns with decades of breeding
        knowledge: flowering time and stress tolerance indicators are the
        primary drivers of yield under variable environments.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        if use_hardcoded:
            st.markdown("""
            The scatter plots below show observed vs predicted yield for
            XGBoost across the three CV schemes, as generated in the study notebook.
            Points close to the red 1:1 line indicate accurate predictions.
            """)
            # Display key r values as metrics since we don't have raw predictions
            col_r1, col_r2, col_r3 = st.columns(3)
            for col, cvname, r_val, rmse_val in [
                (col_r1, 'CV1 (Random)',    0.843, 959.1),
                (col_r2, 'CV2 (Leave-Env)', 0.819, 1023.7),
                (col_r3, 'CV0 (Leave-Geno)',0.847, 948.7),
            ]:
                col.markdown(f"""
                <div class='metric-card'>
                    <h3>r = {r_val}</h3>
                    <p><b>XGBoost {cvname}</b><br>
                    RMSE = {rmse_val:.0f} kg/ha</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            **How to interpret Pearson r:**
            - r = 1.0 → perfect prediction (every point on the 1:1 line)
            - r = 0.847 (CV0) → very strong — new genotypes predicted accurately
            - r = 0.819 (CV2) → strong — new environments predicted accurately
            - r = 0.843 (CV1) → strong — general accuracy

            The fact that CV2 and CV0 r values are close to CV1 confirms the model
            has learned genuinely transferable patterns — not just memorised the
            training data.
            """)

            # Visual explanation of CV schemes
            st.markdown("<div class='section-title'>Why Three CV Schemes?</div>",
                        unsafe_allow_html=True)
            cv_explain = pd.DataFrame({
                'CV Scheme': ['CV1 (Random)', 'CV2 (Leave-Env)', 'CV0 (Leave-Geno)'],
                'What is hidden': [
                    'Random observations',
                    'All data from one environment',
                    'All data from one genotype group'
                ],
                'Breeding scenario simulated': [
                    'General missing data prediction',
                    'Predicting performance in a new, untested location',
                    'Predicting performance of a new variety'
                ],
                'XGBoost RMSE': ['959 kg/ha', '1,024 kg/ha', '949 kg/ha'],
                'XGBoost r': ['0.843', '0.819', '0.847'],
                'Difficulty': ['Easiest', 'Hardest', 'Medium']
            })
            st.dataframe(cv_explain, hide_index=True, use_container_width=True)

        else:
            # Uploaded data — show actual computed scatter
            best_model_name = (results_df[results_df['CV Scheme']=='CV2 (Leave-Env)']
                               .loc[lambda d: d['RMSE (kg/ha)'].idxmin(), 'Model'])
            cv_choice = st.selectbox("Select CV Scheme",
                                      ['CV1 (Random)','CV2 (Leave-Env)',
                                       'CV0 (Leave-Geno)'])
            y_pred_plot = predictions[best_model_name][cv_choice]
            r_val, _  = pearsonr(y_true, y_pred_plot)
            rmse_val  = np.sqrt(mean_squared_error(y_true, y_pred_plot))

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=y_true, y=y_pred_plot, mode='markers',
                marker=dict(color='#2e8b43', opacity=0.4, size=4),
                name='Observations'
            ))
            lim = [0, max(y_true.max(), y_pred_plot.max()) * 1.05]
            fig_scatter.add_trace(go.Scatter(
                x=lim, y=lim, mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='1:1 line'
            ))
            fig_scatter.update_layout(
                title=f'{best_model_name} — {cv_choice}<br>'
                      f'<sup>r = {r_val:.3f} | RMSE = {rmse_val:.0f} kg/ha</sup>',
                xaxis_title='Observed Yield (kg/ha)',
                yaxis_title='Predicted Yield (kg/ha)',
                plot_bgcolor='white', paper_bgcolor='white', height=480
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ================================================================
# PAGE 4 — YIELD PREDICTOR
# ================================================================
elif page == "🌱 Yield Predictor":

    st.markdown("## 🌱 Yield Predictor")
    st.markdown("""
    Predict grain yield for any genotype in a **known trial location** or a
    **new location anywhere in Nigeria** — using only environmental characteristics.
    Confidence intervals are based on XGBoost quantile regression (80% prediction interval).
    """)

    if df_agg is None:
        st.error("⚠️ maize_clean.csv not found.")
        st.stop()

    # Use uploaded model if available, otherwise train on active df
    if st.session_state.use_uploaded and 'model_xgb' in df_agg.attrs:
        # Uploaded data — use pre-trained models stored in attrs
        _xgb   = df_agg.attrs['model_xgb']
        scaler_pred = df_agg.attrs['scaler']
        FEATURES    = df_agg.attrs['features']
        trained_models = {
            'XGBoost': _xgb,
            'q10': df_agg.attrs['model_q10'],
            'q50': df_agg.attrs['model_q50'],
            'q90': df_agg.attrs['model_q90'],
        }
        if st.session_state.use_uploaded:
            st.info(f"📂 Predicting on: **{st.session_state.upload_label}**")
    else:
        trained_models, scaler_pred, FEATURES, _, _ = train_models(df_agg)

    # ---------------------------------------------------------------
    # SHARED LOOKUPS
    # ---------------------------------------------------------------
    loc_env = df_agg.groupby('region').agg({
        'latitude': 'mean', 'longitude': 'mean', 'elevation': 'mean',
        'rainfall_mm': 'mean', 'mean_temperature': 'mean',
        'soil_pH': 'mean', 'soil_N_content': 'mean',
        'soil_P_content': 'mean', 'soil_K_content': 'mean',
        'agro_ecological_zone': 'first', 'loc_mean_yield': 'mean'
    }).reset_index()

    zone_map      = {'Northern Guinea Savanna': 0, 'Southern Guinea Savanna': 1,
                     'Forest\u2013Savanna Transition Zone': 2}
    zone_name_map = {0: 'Northern Guinea Savanna', 1: 'Southern Guinea Savanna',
                     2: 'Forest-Savanna Transition Zone'}
    condition_map = {'Drought': 0, 'Low-N': 1, 'Optimum': 2}
    maturity_map  = {'Extra-Early': 0, 'Early': 1, 'Intermediate': 2, 'late': 3}
    season_map    = {'Dry': 0, 'rainy': 1}
    inst_map      = {'IAR': 0, 'Unilorin': 1, 'CIMMYT and IITA': 2}

    overall_mean = df_agg['grain_yield'].mean()

    # ---------------------------------------------------------------
    # HELPER: build input row and run prediction
    # ---------------------------------------------------------------
    def run_prediction(geno_name, env_data, condition, season, year,
                       geno_enc_val, region_enc_val, zone_enc_val):
        """Build feature vector and return point + quantile predictions."""
        geno_mean  = df_agg[df_agg['Name'] == geno_name]['grain_yield'].mean()
        geno_cv    = df_agg[df_agg['Name'] == geno_name]['grain_yield'].std() / geno_mean * 100
        env_id     = f"{env_data.get('region','New')}_{condition}_{year}"
        env_mean   = (df_agg[df_agg['environment_id'] == env_id]['grain_yield'].mean()
                      if env_id in df_agg['environment_id'].values
                      else df_agg['grain_yield'].mean())
        loc_mean   = env_data.get('loc_mean_yield', df_agg['grain_yield'].mean())

        sti_vals = df_agg[(df_agg['Name'] == geno_name) &
                          (df_agg['environment_condition'] == condition)
                          ]['stress_tolerance_index']
        sti = sti_vals.mean() if len(sti_vals) > 0 else 1.0

        geno_info_row = df_agg[df_agg['Name'] == geno_name].iloc[0]
        geno_traits   = df_agg[df_agg['Name'] == geno_name][
            ['days_to_anthesis', 'Days_to_silking', 'anthesis_silking_interval',
             'plant_height', 'ear_height', 'husk_cover', 'plant_aspect',
             'ear_per_plant', 'ear_aspect', 'grain_moisture', 'staygreen']
        ].mean()

        row = pd.DataFrame([{
            'geno_enc':                   geno_enc_val,
            'institution_enc':            inst_map.get(geno_info_row['breeding_institution'], 0),
            'maturity_enc':               maturity_map.get(geno_info_row['maturity_group'], 1),
            'region_enc':                 region_enc_val,
            'zone_enc':                   zone_enc_val,
            'condition_enc':              condition_map[condition],
            'season_enc':                 season_map[season],
            'YEAR':                       year,
            'latitude':                   env_data['latitude'],
            'longitude':                  env_data['longitude'],
            'elevation':                  env_data['elevation'],
            'rainfall_mm':                env_data['rainfall_mm'],
            'mean_temperature':           env_data['mean_temperature'],
            'soil_pH':                    env_data['soil_pH'],
            'soil_N_content':             env_data['soil_N_content'],
            'soil_P_content':             env_data['soil_P_content'],
            'soil_K_content':             env_data['soil_K_content'],
            'days_to_anthesis':           geno_traits['days_to_anthesis'],
            'Days_to_silking':            geno_traits['Days_to_silking'],
            'anthesis_silking_interval':  geno_traits['anthesis_silking_interval'],
            'plant_height':               geno_traits['plant_height'],
            'ear_height':                 geno_traits['ear_height'],
            'husk_cover':                 geno_traits['husk_cover'],
            'plant_aspect':               geno_traits['plant_aspect'],
            'ear_per_plant':              geno_traits['ear_per_plant'],
            'ear_aspect':                 geno_traits['ear_aspect'],
            'grain_moisture':             geno_traits['grain_moisture'],
            'staygreen':                  geno_traits['staygreen'],
            'geno_mean_yield':            geno_mean,
            'env_mean_yield':             env_mean,
            'loc_mean_yield':             loc_mean,
            'stress_tolerance_index':     sti,
            'geno_yield_cv':              geno_cv
        }])

        X_input   = scaler_pred.transform(row[FEATURES])
        point     = float(trained_models['XGBoost'].predict(X_input)[0])
        lower     = float(trained_models['q10'].predict(X_input)[0])
        median_q  = float(trained_models['q50'].predict(X_input)[0])
        upper     = float(trained_models['q90'].predict(X_input)[0])
        # Ensure bounds are sensible
        lower = max(0, min(lower, point))
        upper = max(point, upper)
        return point, lower, upper, median_q

    # ---------------------------------------------------------------
    # DISPLAY RESULT HELPER
    # ---------------------------------------------------------------
    def show_result(point, lower, upper, geno_name, location_label,
                    condition, geno_hist, is_new_location=False):
        diff_pct  = (point - overall_mean) / overall_mean * 100
        direction = "above" if diff_pct > 0 else "below"
        interval_width = upper - lower
        # Confidence level descriptor
        if interval_width < 500:
            conf_label = "High confidence"
            conf_color = "#2e8b43"
        elif interval_width < 1200:
            conf_label = "Moderate confidence"
            conf_color = "#f5a623"
        else:
            conf_label = "Lower confidence"
            conf_color = "#c0392b"
        if is_new_location:
            conf_label += " (extrapolation to new location)"
            conf_color = "#f5a623"

        col_r1, col_r2, col_r3 = st.columns([0.8, 2, 0.8])
        with col_r2:
            st.markdown(f"""
            <div class='result-box'>
                <p style='color:#666; font-size:0.9rem; margin-bottom:0.3rem;'>
                    {geno_name} &nbsp;·&nbsp; {location_label}
                    &nbsp;·&nbsp; {condition}
                </p>
                <h2 style='font-size:2.6rem;'>{point:,.0f} <span style='font-size:1.2rem;'>kg/ha</span></h2>
                <p style='color:#2e8b43; font-weight:600; margin:0.2rem 0;'>
                    Predicted Grain Yield (XGBoost)
                </p>
                <hr style='border:1px solid #ddd; margin:0.6rem 0;'>
                <p style='font-size:0.95rem; color:#555; margin:0.2rem 0;'>
                    <b>80% Prediction Interval</b> (Quantile Regression)
                </p>
                <p style='font-size:1.15rem; font-weight:600; color:#1a5c2a;'>
                    {lower:,.0f} – {upper:,.0f} kg/ha
                </p>
                <p style='font-size:0.82rem; color:{conf_color}; font-weight:600;'>
                    {conf_label}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart: prediction vs dataset range
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=point,
            delta={'reference': overall_mean,
                   'valueformat': ',.0f',
                   'suffix': ' kg/ha vs mean'},
            number={'suffix': ' kg/ha', 'valueformat': ',.0f'},
            title={'text': f"Predicted Yield vs Dataset Mean ({overall_mean:,.0f} kg/ha)"},
            gauge={
                'axis': {'range': [0, df_agg['grain_yield'].quantile(0.98)],
                          'tickformat': ',.0f'},
                'bar': {'color': '#2e8b43'},
                'steps': [
                    {'range': [0, df_agg['grain_yield'].quantile(0.25)],
                     'color': '#ffcccc'},
                    {'range': [df_agg['grain_yield'].quantile(0.25),
                               df_agg['grain_yield'].quantile(0.75)],
                     'color': '#fff3cc'},
                    {'range': [df_agg['grain_yield'].quantile(0.75),
                               df_agg['grain_yield'].quantile(0.98)],
                     'color': '#ccf0d4'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 3},
                    'thickness': 0.75,
                    'value': overall_mean
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20),
                                  paper_bgcolor='white')
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Context info box
        hist_val = geno_hist.get(condition, None)
        hist_str = (f"Historical mean for <b>{geno_name}</b> under "
                    f"<b>{condition}</b>: {hist_val:,.0f} kg/ha"
                    if hist_val else
                    f"No historical data for <b>{geno_name}</b> under <b>{condition}</b>.")
        st.markdown(f"""
        <div class='warn-box'>
            📊 Prediction is <b>{abs(diff_pct):.1f}% {direction}</b> the overall dataset
            mean of {overall_mean:,.0f} kg/ha.<br>
            🌱 {hist_str}<br>
            📏 The 80% prediction interval of <b>{lower:,.0f}–{upper:,.0f} kg/ha</b>
            means the model expects the true yield to fall within this range
            approximately <b>8 times out of 10</b>.
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # TWO TABS: KNOWN LOCATION  |  NEW LOCATION
    # ---------------------------------------------------------------
    tab_known, tab_new, tab_geno = st.tabs([
        "📍 Known Trial Location",
        "🆕 New Location (Untested)",
        "🧬 New Genotype (Untested)"
    ])

    # ==============================================================
    # TAB 1 — KNOWN LOCATION
    # ==============================================================
    with tab_known:
        st.markdown("""
        Predict yield for a genotype at one of the **5 known trial locations**
        in the dataset. The model uses the exact environmental data for that location.
        """)

        col_in1, col_in2 = st.columns(2)

        with col_in1:
            st.markdown("<div class='section-title'>Select Genotype</div>",
                        unsafe_allow_html=True)
            sel_geno_k = st.selectbox("Genotype",
                                       sorted(df_agg['Name'].unique()),
                                       key='geno_known')
            geno_info_k = df_agg[df_agg['Name'] == sel_geno_k].iloc[0]
            st.markdown(f"""
            **Institution:** {geno_info_k['breeding_institution']}
            **Maturity Group:** {geno_info_k['maturity_group']}
            **Pedigree:** {geno_info_k['Pedigree']}
            """)
            geno_hist_k = (df_agg[df_agg['Name'] == sel_geno_k]
                           .groupby('environment_condition')['grain_yield']
                           .mean().round(0))
            st.markdown("**Historical mean yield by stress condition:**")
            for cond, yld in geno_hist_k.items():
                icon = {'Optimum': '🟢', 'Low-N': '🟡', 'Drought': '🔴'}.get(cond, '⚪')
                st.markdown(f"{icon} **{cond}:** {yld:,.0f} kg/ha")

        with col_in2:
            st.markdown("<div class='section-title'>Select Environment</div>",
                        unsafe_allow_html=True)
            sel_loc_k   = st.selectbox("Location",
                                        sorted(df_agg['region'].unique()),
                                        key='loc_known')
            sel_cond_k  = st.selectbox("Stress Condition",
                                        ['Drought', 'Low-N', 'Optimum'],
                                        key='cond_known')
            sel_seas_k  = st.selectbox("Season", ['rainy', 'Dry'], key='seas_known')
            sel_year_k  = st.selectbox("Year", [2020,2021,2022,2023,2024],
                                        key='year_known')
            loc_row_k   = loc_env[loc_env['region'] == sel_loc_k].iloc[0]
            st.markdown(f"""
            **Zone:** {loc_row_k['agro_ecological_zone']}
            **Rainfall:** {loc_row_k['rainfall_mm']:.0f} mm
            **Mean Temp:** {loc_row_k['mean_temperature']:.1f} °C
            **Elevation:** {loc_row_k['elevation']:.0f} m
            **Soil pH:** {loc_row_k['soil_pH']:.2f}
            **Soil N:** {loc_row_k['soil_N_content']:.3f} g/kg
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict Yield — Known Location",
                     type="primary", use_container_width=True, key='btn_known'):
            with st.spinner("Running prediction..."):
                env_data_k = {
                    'region':          sel_loc_k,
                    'latitude':        float(loc_row_k['latitude']),
                    'longitude':       float(loc_row_k['longitude']),
                    'elevation':       float(loc_row_k['elevation']),
                    'rainfall_mm':     float(loc_row_k['rainfall_mm']),
                    'mean_temperature':float(loc_row_k['mean_temperature']),
                    'soil_pH':         float(loc_row_k['soil_pH']),
                    'soil_N_content':  float(loc_row_k['soil_N_content']),
                    'soil_P_content':  float(loc_row_k['soil_P_content']),
                    'soil_K_content':  float(loc_row_k['soil_K_content']),
                    'loc_mean_yield':  float(loc_row_k['loc_mean_yield'])
                }
                geno_enc_k   = int(le_geno.transform([sel_geno_k])[0])
                region_enc_k = int(le_region.transform([sel_loc_k])[0])
                zone_enc_k   = zone_map.get(loc_row_k['agro_ecological_zone'], 1)

                point, lower, upper, _ = run_prediction(
                    sel_geno_k, env_data_k, sel_cond_k, sel_seas_k, sel_year_k,
                    geno_enc_k, region_enc_k, zone_enc_k
                )
            show_result(point, lower, upper,
                        sel_geno_k, sel_loc_k, sel_cond_k,
                        geno_hist_k, is_new_location=False)

    # ==============================================================
    # TAB 2 — NEW LOCATION
    # ==============================================================
    with tab_new:
        st.markdown("""
        Predict yield for a genotype at a **completely new location** —
        anywhere in Nigeria that was never included in the training data.
        You supply the environmental characteristics of that location.
        The model uses these to predict yield based on patterns it learned
        from the 5 trial locations.

        > **This is the core scientific value of this project:** the model does not
        > memorise locations — it learns from environmental features, so it can
        > generalise to places it has never seen.
        """)

        st.info("""
        💡 **Where to get environmental data for new locations:**
        Rainfall and temperature: [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/)
        Soil data: [ISRIC SoilGrids](https://soilgrids.org/)
        Elevation: [Google Earth](https://earth.google.com) or GPS device
        """)

        col_ng1, col_ng2 = st.columns(2)

        with col_ng1:
            st.markdown("<div class='section-title'>Select Genotype</div>",
                        unsafe_allow_html=True)
            sel_geno_n = st.selectbox("Genotype",
                                       sorted(df_agg['Name'].unique()),
                                       key='geno_new')
            geno_info_n = df_agg[df_agg['Name'] == sel_geno_n].iloc[0]
            st.markdown(f"""
            **Institution:** {geno_info_n['breeding_institution']}
            **Maturity Group:** {geno_info_n['maturity_group']}
            **Pedigree:** {geno_info_n['Pedigree']}
            """)
            geno_hist_n = (df_agg[df_agg['Name'] == sel_geno_n]
                           .groupby('environment_condition')['grain_yield']
                           .mean().round(0))
            st.markdown("**Historical mean yield by stress condition:**")
            for cond, yld in geno_hist_n.items():
                icon = {'Optimum':'🟢','Low-N':'🟡','Drought':'🔴'}.get(cond,'⚪')
                st.markdown(f"{icon} **{cond}:** {yld:,.0f} kg/ha")

        with col_ng2:
            st.markdown("<div class='section-title'>New Location Environmental Data</div>",
                        unsafe_allow_html=True)

            new_loc_name = st.text_input("Location name (for display only)",
                                          placeholder="e.g. Kano, Sokoto, Owerri",
                                          key='new_loc_name')

            # Agroecological zone selector
            new_zone_name = st.selectbox(
                "Agroecological Zone",
                ['Southern Guinea Savanna', 'Northern Guinea Savanna',
                 'Forest-Savanna Transition Zone'],
                key='new_zone'
            )
            new_zone_enc = {'Northern Guinea Savanna': 0,
                             'Southern Guinea Savanna': 1,
                             'Forest-Savanna Transition Zone': 2}[new_zone_name]

            # Use dataset means as sensible defaults
            means = df_agg[['latitude','longitude','elevation','rainfall_mm',
                             'mean_temperature','soil_pH','soil_N_content',
                             'soil_P_content','soil_K_content']].mean()

            c1, c2 = st.columns(2)
            with c1:
                new_lat    = st.number_input("Latitude (°N)",
                                              min_value=4.0, max_value=14.0,
                                              value=float(round(means['latitude'],2)),
                                              step=0.1, key='new_lat')
                new_lon    = st.number_input("Longitude (°E)",
                                              min_value=2.0, max_value=15.0,
                                              value=float(round(means['longitude'],2)),
                                              step=0.1, key='new_lon')
                new_elev   = st.number_input("Elevation (m)",
                                              min_value=0, max_value=2000,
                                              value=int(means['elevation']),
                                              step=10, key='new_elev')
                new_rain   = st.number_input("Annual Rainfall (mm)",
                                              min_value=200, max_value=2000,
                                              value=int(means['rainfall_mm']),
                                              step=10, key='new_rain')
                new_temp   = st.number_input("Mean Temperature (°C)",
                                              min_value=18.0, max_value=40.0,
                                              value=float(round(means['mean_temperature'],1)),
                                              step=0.1, key='new_temp')
            with c2:
                new_pH     = st.number_input("Soil pH",
                                              min_value=4.0, max_value=8.5,
                                              value=float(round(means['soil_pH'],2)),
                                              step=0.1, key='new_pH')
                new_soilN  = st.number_input("Soil N content (g/kg)",
                                              min_value=0.01, max_value=0.5,
                                              value=float(round(means['soil_N_content'],3)),
                                              step=0.005, format="%.3f", key='new_N')
                new_soilP  = st.number_input("Soil P content (mg/kg)",
                                              min_value=1.0, max_value=30.0,
                                              value=float(round(means['soil_P_content'],1)),
                                              step=0.5, key='new_P')
                new_soilK  = st.number_input("Soil K content (mg/kg)",
                                              min_value=50.0, max_value=300.0,
                                              value=float(round(means['soil_K_content'],1)),
                                              step=5.0, key='new_K')

        # Stress condition and season only — no year for new locations
        # Year is not meaningful for untested locations with no historical baseline
        col_ns1, col_ns2 = st.columns(2)
        with col_ns1:
            sel_cond_n = st.selectbox("Stress Condition",
                                       ['Drought', 'Low-N', 'Optimum'],
                                       key='cond_new')
        with col_ns2:
            sel_seas_n = st.selectbox("Season", ['rainy', 'Dry'], key='seas_new')
        # Use dataset mean year internally — not exposed to user
        sel_year_n = int(df_agg['YEAR'].mean().round())

        # Use nearest known location's region_enc as proxy for new location
        # The model uses environmental features for the actual prediction
        # region_enc is a minor feature — use the most similar known location
        nearest_region_enc = int(le_region.transform(
            [loc_env.iloc[
                ((loc_env['latitude'] - new_lat)**2 +
                 (loc_env['longitude'] - new_lon)**2).argmin()
            ]['region']]
        )[0])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict Yield — New Location",
                     type="primary", use_container_width=True, key='btn_new'):
            if not new_loc_name.strip():
                st.warning("Please enter a name for the new location.")
            else:
                with st.spinner(f"Predicting yield for {new_loc_name}..."):
                    env_data_n = {
                        'region':           new_loc_name,
                        'latitude':         new_lat,
                        'longitude':        new_lon,
                        'elevation':        new_elev,
                        'rainfall_mm':      new_rain,
                        'mean_temperature': new_temp,
                        'soil_pH':          new_pH,
                        'soil_N_content':   new_soilN,
                        'soil_P_content':   new_soilP,
                        'soil_K_content':   new_soilK,
                        'loc_mean_yield':   df_agg['grain_yield'].mean()
                    }
                    geno_enc_n = int(le_geno.transform([sel_geno_n])[0])

                    point, lower, upper, _ = run_prediction(
                        sel_geno_n, env_data_n, sel_cond_n, sel_seas_n, sel_year_n,
                        geno_enc_n, nearest_region_enc, new_zone_enc
                    )
                st.success(f"✅ Prediction complete for **{new_loc_name}**")
                show_result(point, lower, upper,
                            sel_geno_n, new_loc_name, sel_cond_n,
                            geno_hist_n, is_new_location=True)

                # Show similar known locations for context
                st.markdown("<div class='section-title'>Most Similar Known Locations"
                            " (for context)</div>", unsafe_allow_html=True)
                loc_env['env_distance'] = np.sqrt(
                    ((loc_env['rainfall_mm']    - new_rain)    / loc_env['rainfall_mm'].std())**2 +
                    ((loc_env['mean_temperature']- new_temp)   / loc_env['mean_temperature'].std())**2 +
                    ((loc_env['soil_pH']         - new_pH)     / loc_env['soil_pH'].std())**2 +
                    ((loc_env['elevation']       - new_elev)   / loc_env['elevation'].std())**2
                )
                similar = loc_env.nsmallest(3, 'env_distance')[
                    ['region','agro_ecological_zone','rainfall_mm',
                     'mean_temperature','elevation','soil_pH','env_distance']
                ].rename(columns={
                    'region': 'Location',
                    'agro_ecological_zone': 'Zone',
                    'rainfall_mm': 'Rainfall (mm)',
                    'mean_temperature': 'Temp (°C)',
                    'elevation': 'Elevation (m)',
                    'soil_pH': 'Soil pH',
                    'env_distance': 'Similarity Score (lower=closer)'
                })
                similar['Similarity Score (lower=closer)'] = similar[
                    'Similarity Score (lower=closer)'].round(3)
                st.dataframe(similar, hide_index=True, use_container_width=True)
                st.caption("""
                The model generalises to new locations by using environmental features.
                The similarity scores above show which known trial locations most
                closely resemble your new location — useful context for interpreting
                the prediction.
                """)

    # ==============================================================
    # TAB 3 — NEW GENOTYPE
    # ==============================================================
    with tab_geno:
        st.markdown("""
        Predict yield for a **brand new maize genotype** not present in the training
        data — using only its agronomic trait measurements and a selected environment.

        > **How this works:** The model learned that yield is driven primarily by
        > agronomic traits (days to silking, ear aspect, staygreen, ASI) and
        > environmental features — not genotype identity. Your new variety's trait
        > measurements are all the model needs to make a prediction.
        > This is validated by CV0 (Leave-One-Genotype-Out): **R² = 0.718, r = 0.847**.
        """)

        st.info("""
        💡 **When to use this tab:**
        When you have a new breeding line at early testing stages — you have measured
        its traits in the nursery or preliminary yield trial, but have not yet conducted
        full multilocational testing. Enter those measurements here to get a predicted
        yield across any environment.
        """)

        # Trait input defaults: use dataset means as starting values
        trait_means = df_agg[['days_to_anthesis', 'Days_to_silking',
                               'anthesis_silking_interval', 'plant_height',
                               'ear_height', 'husk_cover', 'plant_aspect',
                               'ear_per_plant', 'ear_aspect', 'grain_moisture',
                               'staygreen']].mean()

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("<div class='section-title'>New Genotype Information</div>",
                        unsafe_allow_html=True)

            new_geno_name = st.text_input(
                "Genotype / line name (for display only)",
                placeholder="e.g. NewLine-2024-01",
                key='new_geno_name'
            )
            new_geno_inst = st.selectbox(
                "Breeding Institution",
                ['IAR', 'Unilorin', 'CIMMYT and IITA'],
                key='new_geno_inst'
            )
            new_geno_mat = st.selectbox(
                "Maturity Group",
                ['Extra-Early', 'Early', 'Intermediate', 'late'],
                key='new_geno_mat'
            )

            st.markdown("<div class='section-title' style='margin-top:1rem;'>"
                        "Flowering Traits</div>", unsafe_allow_html=True)
            new_da  = st.number_input(
                "Days to Anthesis",
                min_value=40, max_value=90,
                value=int(trait_means['days_to_anthesis']),
                step=1, key='new_da'
            )
            new_ds  = st.number_input(
                "Days to Silking",
                min_value=40, max_value=95,
                value=int(trait_means['Days_to_silking']),
                step=1, key='new_ds'
            )
            new_asi = st.number_input(
                "Anthesis-Silking Interval (ASI, days)",
                min_value=0.0, max_value=15.0,
                value=float(round(trait_means['anthesis_silking_interval'], 1)),
                step=0.5, key='new_asi',
                help="ASI = Days to silking − Days to anthesis. "
                     "Lower ASI = better drought tolerance."
            )

            st.markdown("<div class='section-title' style='margin-top:1rem;'>"
                        "Plant Architecture</div>", unsafe_allow_html=True)
            new_ph  = st.number_input(
                "Plant Height (cm)",
                min_value=80, max_value=300,
                value=int(trait_means['plant_height']),
                step=5, key='new_ph'
            )
            new_eh  = st.number_input(
                "Ear Height (cm)",
                min_value=30, max_value=180,
                value=int(trait_means['ear_height']),
                step=5, key='new_eh'
            )
            new_epp = st.number_input(
                "Ears per Plant",
                min_value=0.5, max_value=2.0,
                value=float(round(trait_means['ear_per_plant'], 2)),
                step=0.05, key='new_epp'
            )

        with col_t2:
            st.markdown("<div class='section-title'>Yield and Quality Scores</div>",
                        unsafe_allow_html=True)
            new_pa  = st.number_input(
                "Plant Aspect Score (1=best, 5=worst)",
                min_value=1.0, max_value=5.0,
                value=float(round(trait_means['plant_aspect'], 1)),
                step=0.5, key='new_pa',
                help="Visual breeder score of overall plant appearance."
            )
            new_ea  = st.number_input(
                "Ear Aspect Score (1=best, 5=worst)",
                min_value=1.0, max_value=5.0,
                value=float(round(trait_means['ear_aspect'], 1)),
                step=0.5, key='new_ea',
                help="Visual breeder score of ear quality."
            )
            new_hc  = st.number_input(
                "Husk Cover Score (1=best, 5=worst)",
                min_value=1.0, max_value=5.0,
                value=float(round(trait_means['husk_cover'], 1)),
                step=0.5, key='new_hc'
            )
            new_gm  = st.number_input(
                "Grain Moisture (%)",
                min_value=8.0, max_value=40.0,
                value=float(round(trait_means['grain_moisture'], 1)),
                step=0.5, key='new_gm'
            )
            new_sg  = st.number_input(
                "Staygreen Score (1=green, 8=senesced)",
                min_value=1, max_value=8,
                value=int(trait_means['staygreen']),
                step=1, key='new_sg',
                help="Higher score = more leaf senescence = lower drought tolerance."
            )

            st.markdown("<div class='section-title' style='margin-top:1rem;'>"
                        "Select Environment</div>", unsafe_allow_html=True)
            new_geno_env_type = st.radio(
                "Environment type",
                ["Known trial location", "New location"],
                key='new_geno_env_type',
                horizontal=True
            )

            if new_geno_env_type == "Known trial location":
                sel_loc_ng   = st.selectbox(
                    "Location", sorted(df_agg['region'].unique()),
                    key='loc_new_geno'
                )
                loc_row_ng   = loc_env[loc_env['region'] == sel_loc_ng].iloc[0]
                env_data_ng  = {
                    'region':           sel_loc_ng,
                    'latitude':         float(loc_row_ng['latitude']),
                    'longitude':        float(loc_row_ng['longitude']),
                    'elevation':        float(loc_row_ng['elevation']),
                    'rainfall_mm':      float(loc_row_ng['rainfall_mm']),
                    'mean_temperature': float(loc_row_ng['mean_temperature']),
                    'soil_pH':          float(loc_row_ng['soil_pH']),
                    'soil_N_content':   float(loc_row_ng['soil_N_content']),
                    'soil_P_content':   float(loc_row_ng['soil_P_content']),
                    'soil_K_content':   float(loc_row_ng['soil_K_content']),
                    'loc_mean_yield':   float(loc_row_ng['loc_mean_yield'])
                }
                region_enc_ng = int(le_region.transform([sel_loc_ng])[0])
                zone_enc_ng   = zone_map.get(loc_row_ng['agro_ecological_zone'], 1)
                loc_label_ng  = sel_loc_ng
            else:
                means_env = df_agg[['latitude','longitude','elevation','rainfall_mm',
                                    'mean_temperature','soil_pH','soil_N_content',
                                    'soil_P_content','soil_K_content']].mean()
                ng_lat  = st.number_input("Latitude (°N)", 4.0, 14.0,
                                           float(round(means_env['latitude'],1)),
                                           0.1, key='ng_lat')
                ng_lon  = st.number_input("Longitude (°E)", 2.0, 15.0,
                                           float(round(means_env['longitude'],1)),
                                           0.1, key='ng_lon')
                ng_rain = st.number_input("Rainfall (mm)", 200, 2000,
                                           int(means_env['rainfall_mm']),
                                           10, key='ng_rain')
                ng_temp = st.number_input("Mean Temperature (°C)", 18.0, 40.0,
                                           float(round(means_env['mean_temperature'],1)),
                                           0.1, key='ng_temp')
                env_data_ng = {
                    'region':           'New Location',
                    'latitude':         ng_lat,
                    'longitude':        ng_lon,
                    'elevation':        float(means_env['elevation']),
                    'rainfall_mm':      float(ng_rain),
                    'mean_temperature': float(ng_temp),
                    'soil_pH':          float(means_env['soil_pH']),
                    'soil_N_content':   float(means_env['soil_N_content']),
                    'soil_P_content':   float(means_env['soil_P_content']),
                    'soil_K_content':   float(means_env['soil_K_content']),
                    'loc_mean_yield':   df_agg['grain_yield'].mean()
                }
                region_enc_ng = int(le_region.transform(
                    [loc_env.iloc[
                        ((loc_env['latitude'] - ng_lat)**2 +
                         (loc_env['longitude'] - ng_lon)**2).argmin()
                    ]['region']]
                )[0])
                zone_enc_ng   = 1
                loc_label_ng  = "New Location"

            sel_cond_ng = st.selectbox(
                "Stress Condition",
                ['Drought', 'Low-N', 'Optimum'],
                key='cond_new_geno'
            )
            sel_seas_ng = st.selectbox(
                "Season", ['rainy', 'Dry'],
                key='seas_new_geno'
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict Yield — New Genotype",
                     type="primary", use_container_width=True,
                     key='btn_new_geno'):
            if not new_geno_name.strip():
                st.warning("Please enter a name for the new genotype.")
            else:
                with st.spinner(f"Predicting yield for {new_geno_name}..."):

                    # For a new genotype, we cannot use le_geno.transform
                    # Instead use dataset mean geno_enc as a neutral placeholder
                    # The prediction relies on the trait values, not identity
                    geno_enc_ng_val  = int(df_agg['geno_enc'].mean())
                    geno_mean_ng     = df_agg['grain_yield'].mean()  # neutral prior
                    geno_cv_ng       = df_agg['grain_yield'].std() / geno_mean_ng * 100

                    # STI: estimate from condition — use dataset mean for that condition
                    sti_ng = (df_agg[df_agg['environment_condition'] == sel_cond_ng]
                              ['stress_tolerance_index'].mean())
                    if pd.isna(sti_ng):
                        sti_ng = 1.0

                    env_id_ng = f"{env_data_ng['region']}_{sel_cond_ng}_2021"
                    env_mean_ng = df_agg['grain_yield'].mean()

                    row_ng = pd.DataFrame([{
                        'geno_enc':                  geno_enc_ng_val,
                        'institution_enc':           inst_map.get(new_geno_inst, 0),
                        'maturity_enc':              maturity_map.get(new_geno_mat, 1),
                        'region_enc':                region_enc_ng,
                        'zone_enc':                  zone_enc_ng,
                        'condition_enc':             condition_map[sel_cond_ng],
                        'season_enc':                season_map[sel_seas_ng],
                        'YEAR':                      2021,
                        'latitude':                  env_data_ng['latitude'],
                        'longitude':                 env_data_ng['longitude'],
                        'elevation':                 env_data_ng['elevation'],
                        'rainfall_mm':               env_data_ng['rainfall_mm'],
                        'mean_temperature':          env_data_ng['mean_temperature'],
                        'soil_pH':                   env_data_ng['soil_pH'],
                        'soil_N_content':            env_data_ng['soil_N_content'],
                        'soil_P_content':            env_data_ng['soil_P_content'],
                        'soil_K_content':            env_data_ng['soil_K_content'],
                        'days_to_anthesis':          new_da,
                        'Days_to_silking':           new_ds,
                        'anthesis_silking_interval': new_asi,
                        'plant_height':              new_ph,
                        'ear_height':                new_eh,
                        'husk_cover':                new_hc,
                        'plant_aspect':              new_pa,
                        'ear_per_plant':             new_epp,
                        'ear_aspect':                new_ea,
                        'grain_moisture':            new_gm,
                        'staygreen':                 new_sg,
                        'geno_mean_yield':           geno_mean_ng,
                        'env_mean_yield':            env_mean_ng,
                        'loc_mean_yield':            env_data_ng['loc_mean_yield'],
                        'stress_tolerance_index':    sti_ng,
                        'geno_yield_cv':             geno_cv_ng
                    }])

                    X_ng    = scaler_pred.transform(row_ng[FEATURES])
                    point   = float(trained_models['XGBoost'].predict(X_ng)[0])
                    lower   = float(trained_models['q10'].predict(X_ng)[0])
                    upper   = float(trained_models['q90'].predict(X_ng)[0])
                    lower   = max(0, min(lower, point))
                    upper   = max(point, upper)

                st.success(f"✅ Prediction complete for **{new_geno_name}**")

                # Show result using show_result helper
                # Pass empty geno_hist since this is a new genotype
                show_result(point, lower, upper,
                            new_geno_name, loc_label_ng, sel_cond_ng,
                            {}, is_new_location=True)

                # Show trait comparison vs dataset mean
                st.markdown("<div class='section-title'>Your Genotype vs "
                            "Dataset Mean Traits</div>",
                            unsafe_allow_html=True)
                trait_compare = pd.DataFrame({
                    'Trait': ['Days to Anthesis', 'Days to Silking', 'ASI',
                              'Plant Height (cm)', 'Ear Height (cm)',
                              'Plant Aspect', 'Ear Aspect', 'Ears/Plant',
                              'Staygreen', 'Grain Moisture'],
                    'Your Genotype': [new_da, new_ds, new_asi, new_ph, new_eh,
                                      new_pa, new_ea, new_epp, new_sg, new_gm],
                    'Dataset Mean': [
                        round(trait_means['days_to_anthesis'], 1),
                        round(trait_means['Days_to_silking'], 1),
                        round(trait_means['anthesis_silking_interval'], 1),
                        round(trait_means['plant_height'], 0),
                        round(trait_means['ear_height'], 0),
                        round(trait_means['plant_aspect'], 2),
                        round(trait_means['ear_aspect'], 2),
                        round(trait_means['ear_per_plant'], 2),
                        round(trait_means['staygreen'], 1),
                        round(trait_means['grain_moisture'], 1)
                    ]
                })
                trait_compare['vs Mean'] = trait_compare.apply(
                    lambda row: '🟢 Better' if (
                        (row['Trait'] in ['Days to Anthesis','Days to Silking','ASI',
                                          'Plant Aspect','Ear Aspect','Staygreen',
                                          'Grain Moisture'] and
                         row['Your Genotype'] < row['Dataset Mean']) or
                        (row['Trait'] in ['Plant Height (cm)','Ear Height (cm)',
                                          'Ears/Plant'] and
                         row['Your Genotype'] > row['Dataset Mean'])
                    ) else '🔴 Worse', axis=1
                )
                st.dataframe(trait_compare, hide_index=True, use_container_width=True)
                st.caption("""
                🟢 Better = trait value associated with higher yield
                (e.g. lower ASI, lower ear aspect score, higher plant height).
                This comparison uses dataset-wide means across all environments.
                Note: prediction confidence is lower for new genotypes since the model
                has not seen this variety's specific GxE pattern. CV0 R² = 0.718.
                """)


# ================================================================
# PAGE 5 — LOCATION CLUSTERING
# ================================================================
elif page == "🗺️ Location Clustering":

    st.markdown("## 🗺️ Trial Network Optimisation — Location Clustering")
    st.markdown("""
    Which trial locations give **redundant** environmental information?
    Removing redundant locations can save breeder resources without losing prediction accuracy.
    """)

    if df_agg is None:
        st.error("⚠️ maize_clean.csv not found.")
        st.stop()

    # Build location matrix
    location_matrix = df_agg.pivot_table(
        index='region', columns='Name', values='grain_yield', aggfunc='mean'
    ).fillna(lambda x: x.mean())
    location_matrix = location_matrix.apply(lambda col: col.fillna(col.mean()))

    scaler_loc    = StandardScaler()
    loc_scaled    = scaler_loc.fit_transform(location_matrix)

    # Silhouette scores
    scores = {}
    for k in range(2, 5):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(loc_scaled)
        scores[k] = silhouette_score(loc_scaled, labels)

    optimal_k = max(scores, key=scores.get)
    km_final  = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters  = km_final.fit_predict(loc_scaled)

    # Location data
    loc_data = df_agg.groupby('region').agg({
        'latitude': 'mean', 'longitude': 'mean',
        'agro_ecological_zone': 'first',
        'rainfall_mm': 'mean', 'mean_temperature': 'mean',
        'elevation': 'mean'
    }).reset_index()
    loc_data['Cluster'] = clusters
    loc_data['Cluster_label'] = loc_data['Cluster'].map(
        {i: f'Cluster {i+1}' for i in range(optimal_k)}
    )

    tab1, tab2, tab3 = st.tabs(["Nigeria Map", "Clustering Analysis", "Cost-Benefit"])

    with tab1:
        st.markdown("<div class='section-title'>Trial Locations on Map of Nigeria</div>",
                    unsafe_allow_html=True)

        col_map, col_leg = st.columns([2, 1])

        with col_map:
            cluster_colors_map = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#f39c12']
            nigeria_map = folium.Map(location=[9.0, 7.5], zoom_start=6,
                                     tiles='CartoDB positron')

            for _, row in loc_data.iterrows():
                color = cluster_colors_map[int(row['Cluster'])]
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=20, color='white', weight=2,
                    fill=True, fill_color=color, fill_opacity=0.85,
                    popup=folium.Popup(
                        f"<b>{row['region']}</b><br>"
                        f"Zone: {row['agro_ecological_zone']}<br>"
                        f"Cluster: {row['Cluster_label']}<br>"
                        f"Rainfall: {row['rainfall_mm']:.0f} mm<br>"
                        f"Temp: {row['mean_temperature']:.1f}°C",
                        max_width=220
                    ),
                    tooltip=f"{row['region']} — {row['Cluster_label']}"
                ).add_to(nigeria_map)

                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:10px; font-weight:bold; '
                             f'color:white; text-align:center; margin-top:6px;">'
                             f'{row["region"]}</div>',
                        icon_size=(80, 20), icon_anchor=(40, -8)
                    )
                ).add_to(nigeria_map)

            st_folium(nigeria_map, width=620, height=480)

        with col_leg:
            st.markdown("<div class='section-title'>Cluster Summary</div>",
                        unsafe_allow_html=True)
            for _, row in loc_data.iterrows():
                color_hex = cluster_colors_map[int(row['Cluster'])]
                st.markdown(f"""
                <div style='background:white; border-left: 5px solid {color_hex};
                     padding: 0.8rem; border-radius:6px; margin-bottom:0.6rem;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.1);'>
                    <b>{row['region']}</b><br>
                    <span style='color:#666; font-size:0.85rem;'>
                    {row['Cluster_label']}<br>
                    {row['agro_ecological_zone']}<br>
                    Rainfall: {row['rainfall_mm']:.0f} mm<br>
                    Elevation: {row['elevation']:.0f} m
                    </span>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        col_sil, col_heat = st.columns(2)

        with col_sil:
            st.markdown("<div class='section-title'>Silhouette Scores by k</div>",
                        unsafe_allow_html=True)
            fig_sil = go.Figure(go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#2e8b43' if k == optimal_k else '#ccc'
                              for k in scores.keys()],
                text=[f'{v:.3f}' for v in scores.values()],
                textposition='outside'
            ))
            fig_sil.update_layout(
                title=f'Optimal k = {optimal_k} (highest silhouette score)',
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Silhouette Score',
                plot_bgcolor='white', paper_bgcolor='white', height=350
            )
            st.plotly_chart(fig_sil, use_container_width=True)

        with col_heat:
            st.markdown("<div class='section-title'>Location × Genotype Heatmap (sample)</div>",
                        unsafe_allow_html=True)
            sample_matrix = location_matrix.iloc[:, :60]
            fig_heat, ax = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(sample_matrix, cmap='YlOrRd', ax=ax,
                        xticklabels=False, cbar_kws={'label': 'Yield (kg/ha)'})
            ax.set_title('Location Performance Profiles\n(first 60 genotypes)')
            ax.set_xlabel('Genotypes')
            ax.set_ylabel('Location')
            plt.tight_layout()
            st.pyplot(fig_heat)

        # Dendrogram
        st.markdown("<div class='section-title'>Hierarchical Clustering Dendrogram</div>",
                    unsafe_allow_html=True)
        linked = linkage(loc_scaled, method='ward')
        fig_dend, ax_dend = plt.subplots(figsize=(8, 4))
        dendrogram(linked, labels=location_matrix.index.tolist(),
                   ax=ax_dend, color_threshold=0,
                   above_threshold_color='#2e8b43')
        ax_dend.set_title('Ward Hierarchical Clustering of Trial Locations')
        ax_dend.set_ylabel('Ward Distance')
        ax_dend.set_xlabel('Location')
        plt.tight_layout()
        st.pyplot(fig_dend)

    with tab3:
        st.markdown("<div class='section-title'>Cost-Benefit: What if We Drop a Location?</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        This analysis trains the model **without each location** and measures how much
        prediction accuracy changes. A small RMSE increase means that location is
        **redundant** and could potentially be removed from the trial network.
        """)

        FEATURES_CB = [
            'geno_enc', 'institution_enc', 'maturity_enc',
            'region_enc', 'zone_enc', 'condition_enc', 'season_enc', 'YEAR',
            'latitude', 'longitude', 'elevation',
            'rainfall_mm', 'mean_temperature',
            'soil_pH', 'soil_N_content', 'soil_P_content', 'soil_K_content',
            'days_to_anthesis', 'Days_to_silking', 'anthesis_silking_interval',
            'plant_height', 'ear_height', 'husk_cover', 'plant_aspect',
            'ear_per_plant', 'ear_aspect', 'grain_moisture', 'staygreen',
            'geno_mean_yield', 'env_mean_yield', 'loc_mean_yield',
            'stress_tolerance_index', 'geno_yield_cv'
        ]

        X_full = df_agg[FEATURES_CB]
        y_full = df_agg['grain_yield']

        baseline_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                        learning_rate=0.1, random_state=42))
        ])
        y_pred_base = cross_val_predict(
            baseline_pipe, X_full, y_full,
            cv=KFold(n_splits=5, shuffle=True, random_state=42)
        )
        baseline_rmse = np.sqrt(mean_squared_error(y_full, y_pred_base))

        drop_rows = []
        for loc in df_agg['region'].unique():
            mask   = df_agg['region'] != loc
            X_red  = df_agg[mask][FEATURES_CB]
            y_red  = df_agg[mask]['grain_yield']
            pipe_r = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBRegressor(n_estimators=100, max_depth=6,
                                            learning_rate=0.1, random_state=42))
            ])
            y_pred_r = cross_val_predict(
                pipe_r, X_red, y_red,
                cv=KFold(n_splits=5, shuffle=True, random_state=42)
            )
            rmse_r = np.sqrt(mean_squared_error(y_red, y_pred_r))
            change = rmse_r - baseline_rmse
            pct    = change / baseline_rmse * 100
            drop_rows.append({
                'Location Dropped': loc,
                'RMSE Without (kg/ha)': round(rmse_r, 1),
                'RMSE Change (kg/ha)': round(change, 1),
                'Change (%)': round(pct, 2),
                'Redundancy': '🟢 High' if pct < 5 else ('🟡 Medium' if pct < 15 else '🔴 Low')
            })

        drop_df = pd.DataFrame(drop_rows).sort_values('RMSE Change (kg/ha)')
        st.markdown(f"**Baseline RMSE (all locations):** {baseline_rmse:.1f} kg/ha")
        st.dataframe(drop_df, hide_index=True, use_container_width=True)

        fig_drop = px.bar(
            drop_df.sort_values('RMSE Change (kg/ha)'),
            x='Location Dropped', y='Change (%)',
            color='Change (%)',
            color_continuous_scale='RdYlGn_r',
            title='RMSE Increase When Each Location is Removed<br>'
                  '<sup>Smaller increase = more redundant location</sup>',
            labels={'Change (%)': 'RMSE Increase (%)'}
        )
        fig_drop.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_drop, use_container_width=True)

        st.info("""
        **How to interpret this for breeders:**
        - 🟢 **High redundancy:** This location can potentially be removed with minimal loss
          of prediction accuracy — consider dropping to save costs
        - 🟡 **Medium redundancy:** Some information is unique, dropping may affect predictions
          in certain environments
        - 🔴 **Low redundancy:** This location provides unique environmental information
          that no other location captures — keep it in the network
        """)


# ================================================================
# PAGE 6 — ABOUT
# ================================================================
elif page == "ℹ️ About":

    st.markdown("## ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Project
        **Machine Learning for Genotype × Environment Interactions
        in Nigerian Maize Breeding Programs**

        **Course:** Practical Machine Learning
        **Programme:** Master's in Green Data Science
        **Institution:** Instituto Superior de Agronomia,
        Universidade de Lisboa
        **Year:** 2025–2026

        ### Team
        - Olawale Serifdeen Aboderin (29206) 
        - Francis Chinaecherem Uzor (29260)  

        ### Supervisor
        Prof. Manuel Campagnolo
        """)

    with col2:
        st.markdown("""
        ### Methods
        | Component | Details |
        |---|---|
        | **Models** | Ridge, Lasso, SVR, Random Forest, XGBoost |
        | **CV Schemes** | CV1 (random), CV2 (leave-env), CV0 (leave-geno) |
        | **Clustering** | K-Means + Hierarchical (Ward) |
        | **Framework** | scikit-learn, XGBoost, Streamlit |
        | **Visualisation** | Plotly, Folium, Seaborn |

        ### Data
        - **21,330 observations**
        - **237 genotypes** from IAR, Unilorin, CIMMYT/IITA
        - **5 locations** across 3 agroecological zones
        - **3 stress conditions:** Drought, Low-N, Optimum
        - **Years:** 2020, 2021, 2022
        """)

    st.markdown("""
    ### References
    - Crossa, J. et al. (2021). Prediction of genetic values of quantitative traits
      in plant breeding using pedigree and molecular markers. *Genetics*, 186(2), 713–724.
    - Yan, W. & Kang, M.S. (2003). *GGE biplot analysis*. CRC Press.
    - Xu, Y. et al. (2020). Enhancing genetic gain through genomic selection.
      *Plant Communications*, 1(1).
    """)

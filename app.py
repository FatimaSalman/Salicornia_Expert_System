import streamlit as st
import pandas as pd
import io
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
from fpdf import FPDF
import os
import csv
from datetime import datetime

# ==========================================
# 1. DATA LOADING
# ==========================================
# Ciechocinek: fully open data (Ďurkovič et al. 2026, Sci Rep 16, 964) -> used for AI training
# Inowrocław: cell-wall stiffness/biomass data originally from Cárdenas Pérez et al. 2024,
#   Environ. Exp. Bot. 218, 105606 (Elsevier). We hold an Elsevier RightsLink license
#   (License #6302010540138) to reproduce Table S1 as a REFERENCE TABLE in the manuscript.
#   That license explicitly excludes using the material "in combination with an artificial
#   intelligence tool ... to train an algorithm." Therefore Inowrocław data is displayed
#   for reference only in this app and is NOT used to train any model.

data_ciechocinek = """NaCl_mM,Stiffness,FW
0,1.790,3.74
200,1.278,9.73
400,1.304,10.10
1000,0.357,0.50"""

# NEW (2026-08-08): Stiffness values for 600 and 800 mM received from S. Cardenas Perez
# (Force_Curve_Analysis_Ciechocinek_600_800__080726.xlsx, "Full Table" sheet, "Average Ciech"
# column). Fresh weight (biomass) for these two points has NOT been received yet, so the
# Biomass model is still trained on the original 4 points only, while the Stiffness model
# now uses all 6 points.
data_ciechocinek_stiffness_extra = """NaCl_mM,Stiffness
600,1.003
800,0.605"""

# NEW (2026-08-08 / confirmed 2026-08-XX): Fresh Weight values for 600 and 800 mM received
# from S. Cardenas Perez (Biomass_Ciech_and_Inow_Stefany_Cardenas.xlsx, "Średnia F.W." column).
# Note: the same file lists 1000 mM Ciechocinek FW as 0.58 g, which differs from the published
# Table S1 value (0.50 g) currently used below -- this discrepancy is pending clarification
# from the data provider, so the 1000 mM point is intentionally left unchanged for now.
data_ciechocinek_biomass_extra = """NaCl_mM,FW
600,6.47
800,1.09"""

data_inowroclaw = """NaCl_mM,Stiffness,FW
0,0.518,3.44
200,0.278,5.85
400,0.102,15.74
1000,0.039,2.18"""

df_ciech = pd.read_csv(io.StringIO(data_ciechocinek))
df_inow = pd.read_csv(io.StringIO(data_inowroclaw))
df_ciech_stiff_extra = pd.read_csv(io.StringIO(data_ciechocinek_stiffness_extra))
df_ciech_fw_extra = pd.read_csv(io.StringIO(data_ciechocinek_biomass_extra))

# Full Stiffness dataset (6 points): original 4 + the new 600/800 mM values
df_ciech_stiffness = pd.concat(
    [df_ciech[['NaCl_mM', 'Stiffness']], df_ciech_stiff_extra],
    ignore_index=True
).sort_values('NaCl_mM').reset_index(drop=True)

# Full Biomass dataset (6 points): original 4 (1000 mM kept at the published 0.50 g pending
# clarification) + the newly confirmed 600/800 mM values
df_ciech_biomass = pd.concat(
    [df_ciech[['NaCl_mM', 'FW']], df_ciech_fw_extra],
    ignore_index=True
).sort_values('NaCl_mM').reset_index(drop=True)

# Combined reference table for transparent display
df_ciech_combined = pd.merge(df_ciech_stiffness, df_ciech_biomass, on='NaCl_mM', how='outer').sort_values('NaCl_mM').reset_index(drop=True)

TRAINING_POINTS_CIECH_STIFFNESS = sorted(df_ciech_stiffness['NaCl_mM'].tolist())
TRAINING_POINTS_CIECH_BIOMASS = sorted(df_ciech_biomass['NaCl_mM'].tolist())
# Kept for any legacy reference; biomass is currently the more limited of the two models
TRAINING_POINTS_CIECH = TRAINING_POINTS_CIECH_BIOMASS

# Pearson correlation matrix (Table S2, Ciechocinek population).
# Source: Cárdenas Pérez et al. 2026, Scientific Reports 16, article 964,
# doi.org/10.1038/s41598-025-30480-w
CORR_CSV = """Variable,E stiffness,Pectin HM-HG,Cellulose,S/G,H/G,S,G,H,Lignin-Total yield,FW
E stiffness,1.0,0.556,0.950,0.239,-0.002,-0.080,-0.602,-0.889,-0.349,0.477
Pectin HM-HG,0.556,1.0,0.708,0.907,-0.825,0.744,0.275,-0.755,0.535,0.897
Cellulose,0.950,0.708,1.0,0.367,-0.186,0.061,-0.484,-0.987,-0.216,0.519
S/G,0.239,0.907,0.367,1.0,-0.960,0.948,0.626,-0.412,0.824,0.944
H/G,-0.002,-0.825,-0.186,-0.960,1.0,-0.981,-0.768,0.268,-0.912,-0.813
S,-0.080,0.744,0.061,0.948,-0.981,1.0,0.840,0.411,0.961,0.818
G,-0.602,0.275,-0.484,0.626,-0.768,0.840,1.0,0.958,-0.146,0.410
H,-0.889,-0.755,-0.987,-0.412,0.268,0.411,0.958,1.0,0.074,-0.516
Lignin-Total yield,-0.349,0.535,-0.216,0.824,-0.912,0.961,-0.146,0.074,1.0,0.646
FW,0.477,0.897,0.519,0.944,-0.813,0.818,0.410,-0.516,0.646,1.0"""
df_corr = pd.read_csv(io.StringIO(CORR_CSV), index_col=0)

# ==========================================
# 2. AI MODEL TRAINING (Ciechocinek only)
# ==========================================
# Biomass: trained on the 4 originally available points (0, 200, 400, 1000 mM)
X_biomass_c = df_ciech_biomass[['NaCl_mM']]
y_biomass_c = df_ciech_biomass['FW']

# Stiffness: trained on all 6 points now that 600/800 mM values are available
X_stiff_c = df_ciech_stiffness[['NaCl_mM']]
y_stiff_c = df_ciech_stiffness['Stiffness']

model_biomass_c = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_biomass_c, y_biomass_c)
model_stiff_c = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_stiff_c, y_stiff_c)

# ==========================================
# 2b. MODEL EVALUATION (LOO-CV)
# ==========================================

def compute_loo_cv(X, y, model_class, model_params=None):
    """Perform Leave-One-Out Cross-Validation and return R², RMSE, MAE."""
    loo = LeaveOneOut()
    preds, actuals = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = model_class(**(model_params or {}))
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test)[0])
        actuals.append(y_test.values[0])
    return {
        'R²': r2_score(actuals, preds),
        'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
        'MAE': mean_absolute_error(actuals, preds)
    }


@st.cache_data
def get_model_comparison():
    """Compute LOO-CV metrics for all candidate models on both targets.
    Cached so the computation only runs once per session.
    Note: as of 2026-08-08, both Stiffness and Biomass use n=6 points (600/800 mM
    added). The 1000 mM Biomass value stays at the published Table S1 figure
    (0.50 g); S. Cardenas Perez confirmed a newer raw file's 0.58 g likely included
    an extra high-variability replicate and should not replace it."""
    models = {
        'Random Forest': (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
        'Linear Regression': (LinearRegression, {}),
        'SVR (RBF)': (Pipeline, {
            'steps': [('scaler', StandardScaler()),
                      ('svr', SVR(kernel='rbf', C=10, gamma=0.01))]
        }),
        'Polynomial (deg=2)': (Pipeline, {
            'steps': [('poly', PolynomialFeatures(degree=2)),
                      ('lr', LinearRegression())]
        })
    }
    results = []
    for target_name, X, y in [('Stiffness', X_stiff_c, y_stiff_c), ('Biomass', X_biomass_c, y_biomass_c)]:
        for name, (cls, params) in models.items():
            if cls == Pipeline:
                loo = LeaveOneOut()
                preds, actuals = [], []
                for train_idx, test_idx in loo.split(X):
                    m = Pipeline(**params)
                    m.fit(X.iloc[train_idx], y.iloc[train_idx])
                    preds.append(m.predict(X.iloc[test_idx])[0])
                    actuals.append(y.iloc[test_idx].values[0])
                metrics = {
                    'R²': r2_score(actuals, preds),
                    'RMSE': np.sqrt(mean_squared_error(actuals, preds)),
                    'MAE': mean_absolute_error(actuals, preds)
                }
            else:
                metrics = compute_loo_cv(X, y, cls, params)
            results.append({
                'Target': target_name,
                'Model': name,
                'N': len(X),
                **metrics
            })
    return pd.DataFrame(results)


df_model_comparison = get_model_comparison()

# ==========================================
# 3. DYNAMIC INFERENCE ENGINE
# ==========================================
# This function actually QUERIES the Pearson correlation matrix (df_corr) at runtime
# to build its explanation, rather than returning a pre-written static sentence.
# This is what makes it a genuine rule-based inference engine reading a live knowledge
# base, matching what the manuscript describes.

def get_regime_description(nacl_input, predicted_stiffness):
    if nacl_input == 0:
        return "At 0 mM (freshwater), the cell wall is expected to reach its stiffest state."
    elif nacl_input >= 800:
        return "Under extreme salinity, a marked loss of stiffness is expected."
    elif predicted_stiffness > 1.0:
        return "High predicted stiffness."
    elif predicted_stiffness < 0.1:
        return "Extreme predicted loss of stiffness."
    else:
        return "Moderate predicted stiffness, indicating a transitional response."


def get_dynamic_inference(nacl_input, predicted_stiffness, pop_name, corr_df,
                           target='E stiffness', top_n=2):
    regime_text = get_regime_description(nacl_input, predicted_stiffness)

    if target not in corr_df.index:
        return f"🔬 **Scientific Reason ({pop_name}):** {regime_text}"

    # Rank all other variables by absolute correlation strength with the target trait
    correlations = corr_df.loc[target].drop(target)
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    top_vars = correlations.head(top_n)

    driver_text = "; ".join(
        f"**{var}** (r = {val:+.3f})" for var, val in top_vars.items()
    )

    return (
        f"🔬 **Scientific Reason ({pop_name}):** {regime_text} This is consistent with the "
        f"strongest correlates of cell wall stiffness in the Pearson correlation matrix "
        f"(Table S2): {driver_text}."
    )


def get_extrapolation_warning(nacl_input, training_points):
    """Warn when the requested salinity is far from any point the model was
    actually trained on -- the model is interpolating/extrapolating, not
    recalling a measured value."""
    nearest = min(training_points, key=lambda x: abs(x - nacl_input))
    gap = abs(nearest - nacl_input)
    if gap == 0:
        return None
    if gap > 100:
        return (f"⚠️ **Extrapolation warning:** {nacl_input} mM is {gap} mM away from the "
                 f"nearest tested salinity ({nearest} mM). With only {len(training_points)} "
                 f"training points, this prediction is a rough interpolation and should be "
                 f"treated as a hypothesis, not an empirical measurement.")
    return (f"ℹ️ Note: {nacl_input} mM is {gap} mM from the nearest tested point "
            f"({nearest} mM); prediction is interpolated.")


# ==========================================
# 3b. REFERENCE PLANT PHOTOS (Ciechocinek)
# ==========================================
# Real repo layout (as uploaded by S. Cardenas Perez):
#   images/ciechocinek/C<conc>/Cie <conc>/<several raw DSCF####.JPG files>
# We just grab the first raw photo (sorted) inside each concentration's
# "Cie <conc>" subfolder, skipping any processed/derivative files that sit
# loose directly under C<conc> (e.g. *_outlined.jpg, *_RGB.jpg).
import re

CIECH_PHOTO_ROOT = "images/ciechocinek"
CIECH_PHOTO_CONCENTRATIONS = [0, 200, 400, 600, 800, 1000]

# Strict raw-camera-photo pattern: exactly "DSCF" + digits + ".JPG" (uppercase extension).
# This automatically excludes every processed/derivative file we've seen
# (e.g. DSCF5465-1_outlined.jpg, DSCF5465-3 RGB.jpg, DSCF5465Salic_C_b.jpg, *.jpg
# lowercase variants) without needing to list them one by one.
RAW_PHOTO_PATTERN = re.compile(r'^DSCF\d+\.JPG$')


def get_nearest_photos(nacl_input, root=CIECH_PHOTO_ROOT,
                        available=CIECH_PHOTO_CONCENTRATIONS, max_photos=3):
    """Return (list_of_paths, documented_conc) for the nearest concentration that has
    raw reference photos on disk. Returns up to `max_photos` photos (sorted) so the
    natural biological variability between replicate plants is visible.

    Some concentration folders (e.g. C200) contain verified, pot-labelled raw photos
    sitting directly under `C<conc>/` in addition to the `Cie <conc>/` subfolder (which
    may hold duplicated/mislabeled files). We therefore check the loose folder first and
    only fall back to the `Cie <conc>/` subfolder if nothing raw is found there.
    Returns ([], documented_conc) if no matching raw photos are found yet.
    """
    nearest = min(available, key=lambda c: abs(c - nacl_input))
    conc_dir = os.path.join(root, f"C{nearest}")

    def raw_photos_in(folder):
        if not os.path.isdir(folder):
            return []
        return sorted(f for f in os.listdir(folder) if RAW_PHOTO_PATTERN.match(f))

    # 1) Prefer verified raw photos sitting directly in C<conc>/
    loose = raw_photos_in(conc_dir)
    if loose:
        chosen = loose[:max_photos]
        return [os.path.join(conc_dir, f) for f in chosen], nearest

    # 2) Fall back to the Cie <conc>/ subfolder
    sub_dir = os.path.join(conc_dir, f"Cie {nearest}")
    sub = raw_photos_in(sub_dir)
    if sub:
        chosen = sub[:max_photos]
        return [os.path.join(sub_dir, f) for f in chosen], nearest

    return [], nearest


# ==========================================
# 4. USER INTERFACE (STREAMLIT APP)
# ==========================================

st.set_page_config(page_title="Salicornia AI Expert System", page_icon="🌱", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    div[data-testid="stMetricValue"] {
        background-color: #ffffff; padding: 10px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #1f77b4; font-size: 28px;
    }
    h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    .st-emotion-cache-yn44r9 h3 {
    padding: 0.5rem 0px 0.5rem;
}
    .st-emotion-cache-yn44r9 h4{
    padding: 0.5rem 0px 0.5rem;
}
    .st-emotion-cache-yn44r9 h1 {
    font-size: 2rem;
    text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 AI Expert System for *Salicornia europaea*")

col_img, col_text = st.columns([1, 2.5])

with col_img:
    st.image("images/salicornia.jpeg", caption="*Salicornia europaea*", width='stretch')

with col_text:
    st.markdown("""
    <div style="display: flex; align-items: center; text-align: justify;">
        <div style="width: 100%;">
            <h4 style="color: #2c3e50;">Project Overview</h4>
            <p style="color: #555; line-height: 1.2;">
                This interactive expert system utilizes Machine Learning to predict the nanomechanical 
                and biomass responses of <i>Salicornia europaea</i> under varying salinity stress.
            </p>
            <h4 style="color: #3498db;">Data Source</h4>
            <p style="color: #555; line-height: 1.2;">
                Ciechocinek population data developed based on the datasets provided by 
                <strong>Stefany Cárdenas Pérez and Jaroslav Ďurkovič</strong> in their study published in 
                <i>(Cárdenas Pérez et al. 2026. Scientific Reports 16, article number 964, 
    doi.org/10.1038/s41598-025-30480-w)</i>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- POPULATION SELECTOR (real, functional) ---
st.subheader("🧬 Plant Population")
selected_pop = st.selectbox(
    "Select Population",
    ["Ciechocinek (full AI prediction)", "Inowrocław (reference data only)"],
    help="Inowrocław data is shown for reference only. Under our current Elsevier "
         "license (License #6302010540138) for its original source (Cárdenas Pérez "
         "et al. 2024, Environ. Exp. Bot. 218, 105606), this data may not be used "
         "to train an AI model in this application."
)

is_ciechocinek = selected_pop.startswith("Ciechocinek")

if not is_ciechocinek:
    st.info(
        "📌 **Reference data only.** Cell-wall stiffness and biomass data for the "
        "Inowrocław population originate from Cárdenas Pérez et al. (2024a: *Environ. "
        "Exp. Bot.* 218, 105606; 2024b: *BMC Plant Biol.* 24), reproduced here with "
        "publisher permission (Elsevier License #6302010540138). This license covers "
        "display of the table but explicitly excludes use of the data to train an AI "
        "model, so no prediction is generated for this population in the app."
    )
    st.subheader("📊 Reference Data - Inowrocław")
    display_df_inow = df_inow.rename(columns={
        'NaCl_mM': 'NaCl (mM)',
        'Stiffness': 'Stiffness (MPa)',
        'FW': 'FW (g)'
    })
    st.dataframe(display_df_inow.style.hide(axis="index").background_gradient(subset=['FW (g)'], cmap='Greens'))
    st.stop()

# ==========================================
# From here on: Ciechocinek only (full AI pipeline)
# ==========================================
current_df = df_ciech_combined  # transparent reference table (all 6 concentrations)
m_biomass = model_biomass_c
m_stiff = model_stiff_c

# Input Slider
st.subheader("⚙️ Input Parameters")
nacl_input = st.slider(label="Select NaCl Concentration (mM)", min_value=0, max_value=1200, value=400, step=50)

# Predictions
input_df = pd.DataFrame([[nacl_input]], columns=['NaCl_mM'])
predicted_biomass = m_biomass.predict(input_df)[0]
predicted_stiffness = m_stiff.predict(input_df)[0]

# Extrapolation warning: both models now share the same 6 training points
# (0, 200, 400, 600, 800, 1000 mM), so a single shared warning is sufficient.
extrap_msg = get_extrapolation_warning(nacl_input, TRAINING_POINTS_CIECH_STIFFNESS)
if extrap_msg:
    if extrap_msg.startswith("⚠️"):
        st.warning(extrap_msg)
    else:
        st.caption(extrap_msg)

# Display Metrics
st.subheader("🧠 Expert System Analysis")
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"Predicted Fresh Biomass ({selected_pop})", value=f"{predicted_biomass:.2f} g")
with col2:
    st.metric(label=f"Predicted Stiffness ({selected_pop})", value=f"{predicted_stiffness:.3f} MPa")

# --- REFERENCE PLANT PHOTOS (multiple, to show natural biological variability) ---
photo_paths, documented_conc = get_nearest_photos(nacl_input)
if photo_paths:
    st.caption(
        (f"📷 Reference photos: {len(photo_paths)} replicate plants at {documented_conc} mM NaCl "
         f"(Cárdenas Pérez et al., original photographic documentation)."
         if documented_conc == nacl_input else
         f"📷 Closest documented reference photos available: {documented_conc} mM NaCl "
         f"(no photo was taken at exactly {nacl_input} mM).")
        + " Note the natural variability between replicate plants grown under the same "
          "treatment — this is expected biological variance, not measurement error."
    )
    photo_cols = st.columns(len(photo_paths))
    for col, path in zip(photo_cols, photo_paths):
        with col:
            st.image(path, width='stretch')
else:
    st.caption(f"📷 Reference photos not yet available for {documented_conc} mM "
               f"(closest documented concentration). Expected folder: "
               f"`{CIECH_PHOTO_ROOT}/C{documented_conc}/Cie {documented_conc}/`")

# Expert Rules
st.write("---")
if nacl_input > 800:
    st.error(f"🚨 **Critical Stress Alert for {selected_pop}!** Extreme salinity. NOT recommended.")
elif nacl_input > 600:
    st.warning(f"🔶 **High Stress Zone for {selected_pop}.** Steep decline expected between 600 and 800 mM.")
elif 200 <= nacl_input <= 600:
    st.success(f"✅ **Optimal Growth Zone for {selected_pop}.** Ideal osmotic potential.")
elif 0 < nacl_input < 200:
    st.warning(f"⚠️ **Sub-optimal for {selected_pop}.** Survives but lower biomass.")
else:
    st.info(f"💧 **Freshwater for {selected_pop}.** Stiffens, but lower halophytic biomass.")

st.info(get_dynamic_inference(nacl_input, predicted_stiffness, "Ciechocinek", df_corr))

# --- PREDICTION LOG (with CSV persistence) ---
st.write("---")
st.subheader("📋 Prediction History Log")

LOG_CSV_PATH = "prediction_log.csv"

# Load persisted log from CSV if it exists
if 'prediction_log' not in st.session_state:
    if os.path.exists(LOG_CSV_PATH):
        st.session_state.prediction_log = pd.read_csv(LOG_CSV_PATH)
    else:
        st.session_state.prediction_log = pd.DataFrame(columns=["Population", "NaCl (mM)", "Predicted Biomass (g)", "Predicted Stiffness (MPa)"])

if st.button("📝 Save Current Prediction to Log"):
    new_entry = pd.DataFrame([{
        "Population": selected_pop,
        "NaCl (mM)": nacl_input,
        "Predicted Biomass (g)": round(predicted_biomass, 2),
        "Predicted Stiffness (MPa)": round(predicted_stiffness, 3)
    }])
    st.session_state.prediction_log = pd.concat([st.session_state.prediction_log, new_entry], ignore_index=True)
    # Persist to CSV
    st.session_state.prediction_log.to_csv(LOG_CSV_PATH, index=False)
    st.success("Prediction saved successfully!")

if not st.session_state.prediction_log.empty:
    st.dataframe(st.session_state.prediction_log, width='stretch')
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        csv_data = st.session_state.prediction_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Export Log as CSV",
            data=csv_data,
            file_name=f"salicornia_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    with col_btn2:
        if st.button("🗑️ Clear Log"):
            st.session_state.prediction_log = pd.DataFrame(columns=["Population", "NaCl (mM)", "Predicted Biomass (g)", "Predicted Stiffness (MPa)"])
            if os.path.exists(LOG_CSV_PATH):
                os.remove(LOG_CSV_PATH)
            st.rerun()
else:
    st.caption("No predictions saved yet.")

# --- MODEL PERFORMANCE SECTION ---
st.write("---")
with st.expander("📊 Model Performance Evaluation (LOO-CV)", expanded=False):
    st.caption("**Leave-One-Out Cross-Validation** was used to evaluate model generalization. "
              "Both the Stiffness and Biomass models now use n=6 training points (600 and 800 mM "
              "NaCl were added on 2026-08-08). Each model is trained on n−1 points and tested "
              "on the held-out point, repeated n times per target.")

    for target in ['Stiffness', 'Biomass']:
        sub = df_model_comparison[df_model_comparison['Target'] == target].copy()
        unit = 'MPa' if target == 'Stiffness' else 'g'
        n_points = sub['N'].iloc[0] if not sub.empty else '?'
        st.markdown(f"**Target: Cell Wall {target} ({unit}) — n = {n_points} training points**")

        styled = sub.drop(columns=['Target', 'N']).style.format({
            'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}'
        })

        def highlight_best(col):
            if col.name == 'R²':
                return ['font-weight: bold; color: #2e7d32' if v == col.max() else '' for v in col]
            else:
                return ['font-weight: bold; color: #2e7d32' if v == col.min() else '' for v in col]

        styled = styled.apply(highlight_best, subset=['R²', 'RMSE', 'MAE'])
        st.dataframe(styled.hide(axis="index"), width='stretch')

    st.info("**Best model per target (green highlight):** The model with the highest R² and lowest RMSE/MAE. "
            "Note: Negative R² values indicate that the model performs worse than simply predicting the mean.")

    st.warning("**Important note on model selection:** Although Leave-One-Out CV may reveal that simpler models "
                "achieve higher R² on this small dataset, Random Forest was retained as the system's predictive "
                "layer for two key reasons:\n\n"
                "1. **Architectural scalability:** RF is non-parametric and improves as additional data points "
                "become available without requiring structural code changes — as demonstrated by both the "
                "Stiffness and Biomass models, which expanded from 4 to 6 points (600 and 800 mM NaCl) without "
                "any code changes.\n\n"
                "2. **System objective:** The primary contribution of this Expert System is the Explainable AI "
                "inference layer, not predictive accuracy per se. The predictions serve as exploratory hypotheses "
                "to be interpreted by the rule-based engine, not as definitive empirical measurements.")

# --- DYNAMIC CHARTS & TABLES ---
st.write("---")
st.subheader(f"📈 Biomass Trend - {selected_pop}")

area = alt.Chart(df_ciech_biomass).mark_area(color='#81c784', opacity=0.3, interpolate='monotone').encode(
    x=alt.X('NaCl_mM:Q', title='NaCl (mM)'), y=alt.Y('FW:Q', title='Fresh Biomass (g)', scale=alt.Scale(zero=False))
)
line = alt.Chart(df_ciech_biomass).mark_line(color='#2e7d32', strokeWidth=3, interpolate='monotone', point=alt.OverlayMarkDef(size=80, filled=True)).encode(
    x='NaCl_mM:Q', y='FW:Q'
)
user_pred_df = pd.DataFrame({'NaCl_mM': [nacl_input], 'FW': [predicted_biomass]})
point = alt.Chart(user_pred_df).mark_point(color='#d32f2f', size=250, filled=True, stroke='white', strokeWidth=3).encode(
    x='NaCl_mM:Q', y='FW:Q', tooltip=['NaCl_mM', 'FW']
)
st.altair_chart(area + line + point)
st.caption("ℹ️ Biomass trend is plotted from all 6 tested concentrations (0, 200, 400, 600, "
           "800, 1000 mM).")

st.write("---")
st.subheader(f"📊 Reference Data - {selected_pop}")
st.caption("Stiffness and Fresh Weight are now available for all 6 tested concentrations. "
           "Note: the 1000 mM Fresh Weight value (0.50 g) matches the published Table S1. "
           "A separate raw data file listed 0.58 g for the same point, but S. Cárdenas Pérez "
           "confirmed this likely reflects an additional high-variability replicate, so the "
           "published 0.50 g value was retained.")
display_df = current_df.rename(columns={
    'NaCl_mM': 'NaCl (mM)',
    'Stiffness': 'Stiffness (MPa)',
    'FW': 'FW (g)'
})
st.dataframe(display_df.style.hide(axis="index").background_gradient(subset=['FW (g)'], cmap='Greens'))

# ==========================================
# 5. SCIENTIFIC REPORT GENERATOR (PDF)
# ==========================================

st.write("---")
st.subheader("📄 Generate Scientific Report")

# Font path: try multiple locations (Linux, macOS, project-local fonts/)
_FONT_SEARCH = [
    "/usr/share/fonts/truetype/dejavu/",                       # Linux
    "/System/Library/Fonts/Supplemental/",                      # macOS
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"),  # project-local
]

def _find_font(name):
    """Return full path for a font file, or None if not found."""
    for d in _FONT_SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

FONT_REGULAR  = _find_font("DejaVuSans.ttf")
FONT_BOLD     = _find_font("DejaVuSans-Bold.ttf")
FONT_ITALIC   = _find_font("DejaVuSans-Oblique.ttf") or FONT_REGULAR
FONT_BOLD_ITALIC = _find_font("DejaVuSans-BoldOblique.ttf") or FONT_BOLD

if not FONT_REGULAR:
    raise FileNotFoundError("DejaVu fonts not found. Download them and place in a 'fonts/' folder next to app.py")

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "",  FONT_REGULAR)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_font("DejaVu", "I", FONT_ITALIC)
        self.add_font("DejaVu", "BI", FONT_BOLD_ITALIC)

    def header(self):
        self.set_font('DejaVu', 'B', 14)
        self.cell(0, 10, 'Salicornia Expert System - Analysis Report', border=False, ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')


def create_report(current_pop, current_nacl, current_biomass, current_stiffness, log_df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.ln(5)

    entries_to_report = []

    is_current_saved = False
    if not log_df.empty:
        # Check ALL rows, not just the last one -- otherwise a live prediction that
        # matches an earlier (non-last) log entry gets duplicated in the report.
        matches = log_df[
            (log_df['Population'] == current_pop) &
            (log_df['NaCl (mM)'] == current_nacl) &
            (log_df['Predicted Biomass (g)'].round(2) == round(current_biomass, 2))
        ]
        if not matches.empty:
            is_current_saved = True

    if not is_current_saved:
        entries_to_report.append({
            'Population': current_pop, 'NaCl': current_nacl,
            'Biomass': current_biomass, 'Stiffness': current_stiffness
        })

    if not log_df.empty:
        for _, row in log_df.iterrows():
            entries_to_report.append({
                'Population': row['Population'], 'NaCl': row['NaCl (mM)'],
                'Biomass': row['Predicted Biomass (g)'], 'Stiffness': row['Predicted Stiffness (MPa)']
            })

    for i, entry in enumerate(entries_to_report):
        explanation = get_dynamic_inference(entry['NaCl'], entry['Stiffness'], entry['Population'], df_corr)
        # Clean markdown formatting (DejaVu handles Unicode natively, no need for character replacement)
        clean_text = explanation.replace("**", "").replace("*", "").replace("🔬", "[Explanation]: ")

        if i > 0:
            pdf.ln(5)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

        pdf.set_font('DejaVu', 'B', 13)
        pdf.set_text_color(52, 152, 219)
        pdf.cell(0, 8, f'Analysis #{i+1}: {entry["Population"]} at {entry["NaCl"]} mM', ln=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font('DejaVu', 'B', 11)
        pdf.cell(0, 7, '1. Input Parameters:', ln=True)
        pdf.set_font('DejaVu', '', 10)
        pdf.cell(0, 6, f'   - Plant Population: {entry["Population"]}', ln=True)
        pdf.cell(0, 6, f'   - NaCl Concentration: {entry["NaCl"]} mM', ln=True)
        pdf.ln(2)

        pdf.set_font('DejaVu', 'B', 11)
        pdf.cell(0, 7, '2. AI Model Predictions:', ln=True)
        pdf.set_font('DejaVu', '', 10)
        pdf.cell(0, 6, f'   - Predicted Fresh Biomass: {entry["Biomass"]:.2f} g', ln=True)
        pdf.cell(0, 6, f'   - Predicted Cell Wall Stiffness: {entry["Stiffness"]:.3f} MPa', ln=True)
        pdf.ln(2)

        pdf.set_font('DejaVu', 'B', 11)
        pdf.cell(0, 7, '3. Biochemical Explanation (dynamically derived from Table S2):', ln=True)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(0, 6, f'   {clean_text}')

    # --- Global Model Reliability Summary ---
    # LOO-CV metrics describe the trained model as a whole and are identical for every
    # scenario above -- they do not vary with the selected NaCl concentration, so this
    # section appears once rather than being repeated per analysis.
    pdf.ln(8)
    pdf.set_draw_color(150, 150, 150)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 13)
    pdf.set_text_color(52, 152, 219)
    pdf.cell(0, 8, 'Model Reliability (Leave-One-Out Cross-Validation)', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('DejaVu', '', 9)
    pdf.multi_cell(0, 5.5,
        'The metrics below describe the trained Random Forest model as a whole and apply '
        'equally to every scenario in this report -- they do not change based on which '
        'NaCl concentration is examined.')
    pdf.ln(2)

    rf_rows = df_model_comparison[df_model_comparison['Model'] == 'Random Forest']
    pdf.set_font('DejaVu', 'B', 10)
    pdf.cell(45, 6, 'Target', border=1)
    pdf.cell(20, 6, 'n', border=1, align='C')
    pdf.cell(38, 6, 'R\u00b2', border=1, align='C')
    pdf.cell(38, 6, 'RMSE', border=1, align='C')
    pdf.cell(38, 6, 'MAE', border=1, align='C', ln=True)
    pdf.set_font('DejaVu', '', 10)
    for _, row in rf_rows.iterrows():
        pdf.cell(45, 6, str(row['Target']), border=1)
        pdf.cell(20, 6, str(row['N']), border=1, align='C')
        pdf.cell(38, 6, f"{row['R\u00b2']:.4f}", border=1, align='C')
        pdf.cell(38, 6, f"{row['RMSE']:.4f}", border=1, align='C')
        pdf.cell(38, 6, f"{row['MAE']:.4f}", border=1, align='C', ln=True)

    pdf.ln(3)
    pdf.set_font('DejaVu', 'I', 9)
    pdf.multi_cell(0, 5.5,
        'Random Forest was retained as the predictive layer despite not always achieving the '
        'highest cross-validated R\u00b2 on this small dataset, for architectural scalability '
        'and explainability reasons (see manuscript Section 5 for details).')

    pdf.ln(8)
    pdf.set_font('DejaVu', 'I', 9)
    pdf.cell(0, 8, 'Data Source: Cárdenas Pérez et al. (2026), Scientific Reports 16, 964.', ln=True)

    return bytes(pdf.output())


if st.button("⬇️ Download Analysis Report (PDF)"):
    pdf_bytes = create_report(
        current_pop=selected_pop,
        current_nacl=nacl_input,
        current_biomass=predicted_biomass,
        current_stiffness=predicted_stiffness,
        log_df=st.session_state.prediction_log
    )

    st.download_button(
        label="Click here to save the PDF file",
        data=pdf_bytes,
        file_name=f"Salicornia_Full_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
        mime="application/octet-stream"
    )

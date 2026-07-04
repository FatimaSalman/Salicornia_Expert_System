import streamlit as st
import pandas as pd
import io
import altair as alt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from fpdf import FPDF

# ==========================================
# 1. DATA LOADING (BOTH POPULATIONS)
# ==========================================

# Data for Population 1: Ciechocinek
data_ciechocinek = """NaCl_mM,Stiffness,FW
0,1.790,3.74
200,1.278,9.73
400,1.304,10.10
1000,0.357,0.50"""

# Data for Population 2: Inowrocław (Extracted from Table S1)
data_inowroclaw = """NaCl_mM,Stiffness,FW
0,0.518,3.44
200,0.278,5.85
400,0.102,15.74
1000,0.039,2.18"""

df_ciech = pd.read_csv(io.StringIO(data_ciechocinek))
df_inow = pd.read_csv(io.StringIO(data_inowroclaw))

# ==========================================
# 2. AI MODEL TRAINING (TWO MODELS)
# ==========================================

X_c = df_ciech[['NaCl_mM']]
y_biomass_c = df_ciech['FW']
y_stiff_c = df_ciech['Stiffness']

X_i = df_inow[['NaCl_mM']]
y_biomass_i = df_inow['FW']
y_stiff_i = df_inow['Stiffness']

# Train models for Ciechocinek
model_biomass_c = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_c, y_biomass_c)
model_stiff_c = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_c, y_stiff_c)

# Train models for Inowrocław
model_biomass_i = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_i, y_biomass_i)
model_stiff_i = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_i, y_stiff_i)

# ==========================================
# 3. SCIENTIFIC EXPLANATION ENGINE
# ==========================================

def get_scientific_explanation(nacl_input, predicted_stiffness, pop_name):
    if nacl_input == 0:
        return f"🔬 **Scientific Reason ({pop_name}):** At 0 mM (Freshwater), the cell wall exhibits maximum stiffness due to high **Cellulose (r=0.950)** and **HM-HG Pectin (r=0.556)**."
    elif nacl_input >= 800:
        return f"🔬 **Scientific Reason ({pop_name}):** Under extreme salinity, severe loss of stiffness is linked to a massive drop in **Cellulose** and accumulation of **H-lignin monomers (r=-0.889)**."
    elif predicted_stiffness > 1.0:
        return f"🔬 **Scientific Reason ({pop_name}):** High stiffness driven by strong positive correlation with **Cellulose (r=0.950)**."
    elif predicted_stiffness < 0.1:
        return f"🔬 **Scientific Reason ({pop_name}):** Extreme loss of stiffness due to drastic accumulation of **H-lignin monomers (r=-0.889)**."
    else:
        return f"🔬 **Scientific Reason ({pop_name}):** Moderate stiffness indicates a transitional phase. Shift in **S/G lignin ratio (r=0.239)** and balanced reduction in cellulose."

# ==========================================
# 4. USER INTERFACE (STREAMLIT APP)
# ==========================================

st.set_page_config(page_title="Salicornia AI Expert System", page_icon="🌱",layout="centered")

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

# Create two columns: one for the image, one for the text
col_img, col_text = st.columns([1, 2.5]) # Image takes 1 part, Text takes 2.5 parts

with col_img:
    # Display the local image
    # Make sure the image file name matches exactly what you saved
    st.image("images/salicornia.jpeg", caption="*Salicornia europaea*", width='stretch')

with col_text:
    # Using HTML/CSS Flexbox to perfectly center the text vertically next to the image
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
                Developed based on the precise datasets provided by 
                <strong>Dr. Stefany Cárdenas Pérez and Prof. Jaroslav Ďurkovič</strong> in their study published in 
                <i>(Cárdenas Pérez et al. 2026. Scientific Reports 16, article number 964, 
    doi.org/10.1038/s41598-025-30480-w)</i>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# --- NEW FEATURE: POPULATION SELECTOR ---
st.subheader("🧬 Plant Population")
st.markdown("**Currently Analyzing:** *Ciechocinek* (Open Access Data)")
selected_pop = "Ciechocinek"

# Dynamically assign data and models based on selection
if selected_pop == "Ciechocinek":
    current_df = df_ciech
    m_biomass = model_biomass_c
    m_stiff = model_stiff_c
else:
    current_df = df_inow
    m_biomass = model_biomass_i
    m_stiff = model_stiff_i

# Input Slider
st.subheader("⚙️ Input Parameters")
nacl_input = st.slider(label="Select NaCl Concentration (mM)", min_value=0, max_value=1200, value=400, step=50)

# Predictions
input_df = pd.DataFrame([[nacl_input]], columns=['NaCl_mM'])
predicted_biomass = m_biomass.predict(input_df)[0]
predicted_stiffness = m_stiff.predict(input_df)[0]

# Display Metrics
st.subheader("🧠 Expert System Analysis")
col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"Predicted Fresh Biomass ({selected_pop})", value=f"{predicted_biomass:.2f} g")
with col2:
    st.metric(label=f"Predicted Stiffness ({selected_pop})", value=f"{predicted_stiffness:.3f} MPa")

# Expert Rules
st.write("---")
if nacl_input > 800:
    st.error(f"🚨 **Critical Stress Alert for {selected_pop}!** Extreme salinity. NOT recommended.")
elif 200 <= nacl_input <= 600:
    st.success(f"✅ **Optimal Growth Zone for {selected_pop}.** Ideal osmotic potential.")
elif 0 < nacl_input < 200:
    st.warning(f"⚠️ **Sub-optimal for {selected_pop}.** Survives but lower biomass.")
else:
    st.info(f"💧 **Freshwater for {selected_pop}.** Stiffens, but lower halophytic biomass.")

st.info(get_scientific_explanation(nacl_input, predicted_stiffness, selected_pop))

# --- PREDICTION LOG ---
st.write("---")
st.subheader("📋 Prediction History Log")

if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = pd.DataFrame(columns=["Population", "NaCl (mM)", "Predicted Biomass (g)", "Predicted Stiffness (MPa)"])

if st.button("📝 Save Current Prediction to Log"):
    new_entry = pd.DataFrame([{
        "Population": selected_pop,
        "NaCl (mM)": nacl_input,
        "Predicted Biomass (g)": round(predicted_biomass, 2),
        "Predicted Stiffness (MPa)": round(predicted_stiffness, 3)
    }])
    st.session_state.prediction_log = pd.concat([st.session_state.prediction_log, new_entry], ignore_index=True)
    st.success("Prediction saved successfully!")

if not st.session_state.prediction_log.empty:
    st.dataframe(st.session_state.prediction_log)
else:
    st.caption("No predictions saved yet.")

# --- DYNAMIC CHARTS & TABLES ---
st.write("---")
st.subheader(f"📈 Biomass Trend - {selected_pop}")

area = alt.Chart(current_df).mark_area(color='#81c784', opacity=0.3, interpolate='monotone').encode(
    x=alt.X('NaCl_mM:Q', title='NaCl (mM)'), y=alt.Y('FW:Q', title='Fresh Biomass (g)', scale=alt.Scale(zero=False))
)
line = alt.Chart(current_df).mark_line(color='#2e7d32', strokeWidth=3, interpolate='monotone', point=alt.OverlayMarkDef(size=80, filled=True)).encode(
    x='NaCl_mM:Q', y='FW:Q'
)
user_pred_df = pd.DataFrame({'NaCl_mM': [nacl_input], 'FW': [predicted_biomass]})
point = alt.Chart(user_pred_df).mark_point(color='#d32f2f', size=250, filled=True, stroke='white', strokeWidth=3).encode(
    x='NaCl_mM:Q', y='FW:Q', tooltip=['NaCl_mM', 'FW']
)
st.altair_chart(area + line + point)

st.write("---")
st.subheader(f"📊 Reference Data - {selected_pop}")
st.dataframe(current_df.style.hide(axis="index").background_gradient(subset=['FW'], cmap='Greens'))

# ==========================================
# 5. SCIENTIFIC REPORT GENERATOR (PDF)
# ==========================================

st.write("---")
st.subheader("📄 Generate Scientific Report")

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Salicornia Expert System - Analysis Report', border=False, ln=True, align='C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_report(current_pop, current_nacl, current_biomass, current_stiffness, log_df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Report Metadata
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.ln(5)
    
    # --- PREPARE ALL ENTRIES TO AVOID DUPLICATION ---
    entries_to_report = []
    
    # Check if the current state is already saved in the log
    is_current_saved = False
    if not log_df.empty:
        last_row = log_df.iloc[-1]
        if (last_row['Population'] == current_pop and 
            last_row['NaCl (mM)'] == current_nacl and
            round(last_row['Predicted Biomass (g)'], 2) == round(current_biomass, 2)):
            is_current_saved = True
            
    # If it's a new state (not saved yet), add it first
    if not is_current_saved:
        entries_to_report.append({
            'Population': current_pop, 'NaCl': current_nacl,
            'Biomass': current_biomass, 'Stiffness': current_stiffness
        })
        
    # Add all saved log entries
    if not log_df.empty:
        for _, row in log_df.iterrows():
            entries_to_report.append({
                'Population': row['Population'], 'NaCl': row['NaCl (mM)'],
                'Biomass': row['Predicted Biomass (g)'], 'Stiffness': row['Predicted Stiffness (MPa)']
            })

    # --- PRINT EACH ENTRY AS A FULL BLOCK ---
    for i, entry in enumerate(entries_to_report):
        
        # Generate specific explanation for this exact entry
        explanation = get_scientific_explanation(entry['NaCl'], entry['Stiffness'], entry['Population'])
        clean_text = explanation.replace("**", "").replace("*", "").replace("🔬", "[Explanation]: ")
        clean_text = clean_text.replace("ł", "l").replace("ą", "a").replace("ę", "e")
        pop_safe = entry['Population'].replace("ł", "l").replace("ą", "a").replace("ę", "e")
        
        # Draw a separator line between different analyses
        if i > 0:
            pdf.ln(5)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
        # Entry Header
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(52, 152, 219) # Blue color for title
        pdf.cell(0, 8, f'Analysis #{i+1}: {pop_safe} at {entry["NaCl"]} mM', ln=True)
        pdf.set_text_color(0, 0, 0) # Reset to black
        
        # 1. Input Parameters
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, '1. Input Parameters:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'   - Plant Population: {pop_safe}', ln=True)
        pdf.cell(0, 6, f'   - NaCl Concentration: {entry["NaCl"]} mM', ln=True)
        pdf.ln(2)
        
        # 2. AI Predictions
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, '2. AI Model Predictions:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'   - Predicted Fresh Biomass: {entry["Biomass"]:.2f} g', ln=True)
        pdf.cell(0, 6, f'   - Predicted Cell Wall Stiffness: {entry["Stiffness"]:.3f} MPa', ln=True)
        pdf.ln(2)
        
        # 3. Biochemical Explanation
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 7, '3. Biochemical Explanation (Based on Table S2):', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, f'   {clean_text}')

    # Footer Credit
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.cell(0, 8, 'Data Source: Durkovic et al. (2025), Scientific Reports (Nature).', ln=True)
    
    return bytes(pdf.output())

# Button to generate and download
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
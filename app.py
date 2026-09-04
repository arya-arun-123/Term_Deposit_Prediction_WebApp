"""
Term Deposit Subscriber Predictor Web App
------------------------------------------
A high-contrast, dark & emerald-green themed Streamlit application using pre-trained ML models
(encoder.pkl, scaler.pkl, model.pkl) to predict term deposit subscription likelihood.
Includes tab navigation buttons, verification checkboxes per tab, and high-legibility styling.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Term Deposit Subscriber Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
TAB_NAMES = [
    "👤 1. Customer Profile",
    "💰 2. Financial Info",
    "📞 3. Campaign & Predict"
]

if "current_tab" not in st.session_state:
    st.session_state.current_tab = TAB_NAMES[0]

if "tab1_verified" not in st.session_state:
    st.session_state.tab1_verified = False
if "tab2_verified" not in st.session_state:
    st.session_state.tab2_verified = False
if "tab3_verified" not in st.session_state:
    st.session_state.tab3_verified = False

# Sync radio selection with session state current_tab
if "nav_tab_radio" not in st.session_state:
    st.session_state.nav_tab_radio = st.session_state.current_tab

# =========================================================
# CUSTOM CSS — HIGH READABILITY & CONTRAST DARK THEME
# =========================================================
st.markdown(
    """
    <style>
        /* Overall Dark Theme */
        .stApp {
            background-color: #0B0F17;
            color: #F8FAFC;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* Hero Header */
        .hero-container {
            text-align: center;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #0F172A 0%, #064E3B 100%);
            border: 1px solid #10B981;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #A7F3D0;
            font-weight: 500;
        }

        /* Card Container */
        .card-box {
            background-color: #1E293B;
            padding: 1.8rem;
            border-radius: 16px;
            border: 1px solid #334155;
            margin-bottom: 1.5rem;
        }

        /* HIGH READABILITY FOR ALL INPUT WIDGETS & LABELS */
        label, [data-testid="stWidgetLabel"] p, .stMarkdown p, .stRadio label p {
            color: #FFFFFF !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }

        /* Selectbox dropdown & input text */
        div[data-baseweb="select"] > div, input, textarea {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border: 1px solid #10B981 !important;
            border-radius: 10px !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }

        /* Selectbox current value text */
        div[data-baseweb="select"] span {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Dropdown Options Popup */
        div[role="listbox"] {
            background-color: #0F172A !important;
            border: 1px solid #10B981 !important;
        }

        div[role="listbox"] ul li, div[role="listbox"] div {
            color: #FFFFFF !important;
            background-color: #0F172A !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }

        div[role="listbox"] ul li:hover, div[role="listbox"] div:hover {
            background-color: #10B981 !important;
            color: #0B0F17 !important;
        }

        /* Radio Options Buttons Text */
        div[role="radiogroup"] label {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }

        div[role="radiogroup"] label[data-checked="true"] {
            border-color: #10B981 !important;
            background-color: #064E3B !important;
        }

        /* Slider track and label colors */
        .stSlider label p, .stSlider div[data-testid="stTickBar"] div {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* Navigation Radio Tab Bar */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 12px;
            background-color: #0F172A;
            padding: 10px;
            border-radius: 14px;
            border: 1px solid #10B981;
            margin-bottom: 1.5rem;
        }

        /* Result Cards */
        .result-success {
            background: #064E3B;
            border: 2px solid #10B981;
            padding: 1.8rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
        }

        .result-danger {
            background: #7F1D1D;
            border: 2px solid #EF4444;
            padding: 1.8rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS & CATEGORIES
# =========================================================
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]
MARITAL_OPTIONS = ["married", "single", "divorced"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]
POUTCOME_OPTIONS = ["unknown", "failure", "other", "success"]

JOB_DISPLAY = {j: j.replace(".", "").capitalize() for j in JOB_OPTIONS}
MONTH_DISPLAY = {
    "jan": "January", "feb": "February", "mar": "March",
    "apr": "April", "may": "May", "jun": "June",
    "jul": "July", "aug": "August", "sep": "September",
    "oct": "October", "nov": "November", "dec": "December"
}

CAT_COLS = ["job", "marital", "education", "contact", "month", "poutcome"]

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_artifacts():
    paths = {
        "encoder": os.path.join(ARTIFACT_DIR, "encoder.pkl"),
        "scaler": os.path.join(ARTIFACT_DIR, "scaler.pkl"),
        "model": os.path.join(ARTIFACT_DIR, "model.pkl"),
    }
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing
    try:
        return {
            "encoder": joblib.load(paths["encoder"]),
            "scaler": joblib.load(paths["scaler"]),
            "model": joblib.load(paths["model"]),
        }, []
    except Exception as e:
        return None, [str(e)]

artifacts, missing_files = load_artifacts()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🔮 Term Deposit Subscriber Predictor</div>
        <div class="hero-subtitle">Complete all 3 steps below to predict bank term deposit subscription probability</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if artifacts is None:
    st.error(f"Missing required model artifacts: {', '.join(missing_files)}. Please check the `artifacts/` folder.")
    st.stop()

encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
model = artifacts["model"]
expected_feature_names = getattr(scaler, "feature_names_in_", None)

# =========================================================
# NAVIGATION TABS (RADIO CONTROLLED)
# =========================================================
def on_tab_radio_change():
    st.session_state.current_tab = st.session_state.nav_tab_radio

def format_tab_title(tab):
    if tab == TAB_NAMES[0]:
        return tab + ("  ✅" if st.session_state.tab1_verified else "  ⏳")
    elif tab == TAB_NAMES[1]:
        return tab + ("  ✅" if st.session_state.tab2_verified else "  ⏳")
    elif tab == TAB_NAMES[2]:
        return tab + ("  ✅" if st.session_state.tab3_verified else "  ⏳")
    return tab

# Ensure nav_tab_radio matches current_tab index
current_index = TAB_NAMES.index(st.session_state.current_tab) if st.session_state.current_tab in TAB_NAMES else 0

st.radio(
    "Navigation Bar",
    options=TAB_NAMES,
    format_func=format_tab_title,
    index=current_index,
    key="nav_tab_radio",
    on_change=on_tab_radio_change,
    horizontal=True,
    label_visibility="collapsed",
)

current_tab = st.session_state.current_tab

# =========================================================
# TAB 1: CUSTOMER PROFILE
# =========================================================
if current_tab == TAB_NAMES[0]:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("👤 Step 1: Customer Profile Demographics")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (Years)", 18, 95, 38, key="age")
        job = st.selectbox("Job / Occupation", JOB_OPTIONS, format_func=lambda j: JOB_DISPLAY.get(j, j), index=4, key="job")
    with col2:
        marital = st.radio("Marital Status", MARITAL_OPTIONS, format_func=lambda m: m.capitalize(), index=0, horizontal=True, key="marital")
        education = st.radio("Education Level", EDUCATION_OPTIONS, format_func=lambda e: e.capitalize(), index=2, horizontal=True, key="education")

    st.markdown("---")
    st.session_state.tab1_verified = st.checkbox(
        "✅ I confirm all Customer Profile inputs are entered & verified",
        value=st.session_state.tab1_verified,
        key="chk_tab1"
    )

    st.markdown("---")
    c_space, c_next = st.columns([1, 1])
    with c_next:
        if st.button("Next: Financial Info ➡️", type="primary", use_container_width=True, key="btn_next_t1"):
            if not st.session_state.tab1_verified:
                st.warning("⚠️ Please check the verification box above before moving to Step 2.")
            else:
                st.session_state.current_tab = TAB_NAMES[1]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2: FINANCIAL INFO
# =========================================================
elif current_tab == TAB_NAMES[1]:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("💰 Step 2: Financial & Account Details")
    
    col_a, col_b = st.columns(2)
    with col_a:
        balance = st.slider("Average Yearly Balance (€)", -10000, 100000, 1500, step=250, key="balance")
        housing_str = st.radio("Housing Loan?", ["No", "Yes"], index=0, horizontal=True, key="housing")
    with col_b:
        loan_str = st.radio("Personal Loan?", ["No", "Yes"], index=0, horizontal=True, key="loan")
        default_str = st.radio("Has Credit Default?", ["No", "Yes"], index=0, horizontal=True, key="default")

    st.markdown("---")
    st.session_state.tab2_verified = st.checkbox(
        "✅ I confirm all Financial Info inputs are entered & verified",
        value=st.session_state.tab2_verified,
        key="chk_tab2"
    )

    st.markdown("---")
    c_prev, c_next2 = st.columns([1, 1])
    with c_prev:
        if st.button("⬅️ Previous: Customer Profile", use_container_width=True, key="btn_prev_t2"):
            st.session_state.current_tab = TAB_NAMES[0]
            st.rerun()
    with c_next2:
        if st.button("Next: Campaign & Predict ➡️", type="primary", use_container_width=True, key="btn_next_t2"):
            if not st.session_state.tab2_verified:
                st.warning("⚠️ Please check the verification box above before moving to Step 3.")
            else:
                st.session_state.current_tab = TAB_NAMES[2]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3: CAMPAIGN & PREDICT
# =========================================================
else:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("📞 Step 3: Campaign Details & Final Prediction")
    
    col_x, col_y = st.columns(2)
    with col_x:
        contact = st.radio("Contact Type", CONTACT_OPTIONS, format_func=lambda c: c.capitalize(), index=0, horizontal=True, key="contact")
        month = st.selectbox("Last Contact Month", MONTH_OPTIONS, format_func=lambda m: MONTH_DISPLAY.get(m, m), index=4, key="month")
        campaign = st.slider("Contacts Executed This Campaign", 1, 50, 2, key="campaign")
    with col_y:
        pdays = st.slider("Days Since Last Contact (-1 = Never)", -1, 999, -1, key="pdays")
        previous = st.slider("Previous Contacts Count", 0, 50, 0, key="previous")
        poutcome = st.radio("Previous Campaign Outcome", POUTCOME_OPTIONS, format_func=lambda p: p.capitalize(), index=0, horizontal=True, key="poutcome")

    st.markdown("---")
    st.session_state.tab3_verified = st.checkbox(
        "✅ I confirm all Campaign Details inputs are entered & verified",
        value=st.session_state.tab3_verified,
        key="chk_tab3"
    )

    st.markdown("---")
    c_prev3, c_info = st.columns([1, 1])
    with c_prev3:
        if st.button("⬅️ Previous: Financial Info", use_container_width=True, key="btn_prev_t3"):
            st.session_state.current_tab = TAB_NAMES[1]
            st.rerun()

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Term Deposit Subscription", type="primary", use_container_width=True, key="btn_predict")

    if predict_clicked:
        all_ok = (
            st.session_state.tab1_verified and 
            st.session_state.tab2_verified and 
            st.session_state.tab3_verified
        )

        if not all_ok:
            unverified_list = []
            if not st.session_state.tab1_verified: unverified_list.append("Step 1: Customer Profile")
            if not st.session_state.tab2_verified: unverified_list.append("Step 2: Financial Info")
            if not st.session_state.tab3_verified: unverified_list.append("Step 3: Campaign Details")
            
            st.warning(
                f"⚠️ **Verification Incomplete**: Please review and check the verification box on:\n" +
                "\n".join([f"- **{item}**" for item in unverified_list])
            )
        else:
            try:
                age_val = st.session_state.get("age", 38)
                job_val = st.session_state.get("job", "management")
                marital_val = st.session_state.get("marital", "married")
                education_val = st.session_state.get("education", "tertiary")
                balance_val = st.session_state.get("balance", 1500)
                housing_val = 1 if st.session_state.get("housing", "No") == "Yes" else 0
                loan_val = 1 if st.session_state.get("loan", "No") == "Yes" else 0
                contact_val = st.session_state.get("contact", "cellular")
                month_val = st.session_state.get("month", "may")
                campaign_val = st.session_state.get("campaign", 2)
                pdays_val = st.session_state.get("pdays", -1)
                previous_val = st.session_state.get("previous", 0)
                poutcome_val = st.session_state.get("poutcome", "unknown")

                df_num_binary = pd.DataFrame([{
                    "age": age_val,
                    "balance": balance_val,
                    "housing": housing_val,
                    "loan": loan_val,
                    "campaign": campaign_val,
                    "pdays": pdays_val,
                    "previous": previous_val,
                }])

                df_cat = pd.DataFrame([{
                    "job": job_val,
                    "marital": marital_val,
                    "education": education_val,
                    "contact": contact_val,
                    "month": month_val,
                    "poutcome": poutcome_val,
                }])

                encoded_array = encoder.transform(df_cat)
                encoded_cols = encoder.get_feature_names_out(CAT_COLS)
                df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols)

                full_df = pd.concat([df_num_binary, df_encoded], axis=1)

                if expected_feature_names is not None:
                    full_df = full_df.reindex(columns=expected_feature_names, fill_value=0)

                scaled_features = scaler.transform(full_df)
                prediction = model.predict(scaled_features)[0]

                probability = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba_arr = model.predict_proba(scaled_features)[0]
                        probability = float(proba_arr[1])
                    except Exception:
                        probability = None

                prob_display = f"{probability * 100.0:.1f}%" if probability is not None else "N/A"

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-success">
                            <h2 style="color: #34D399; margin: 0; font-size: 2rem;">🎯 Customer Likely to Subscribe!</h2>
                            <p style="font-size: 1.3rem; color: #FFFFFF; margin-top: 0.6rem; font-weight: 700;">Prediction Probability: {prob_display}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-danger">
                            <h2 style="color: #F87171; margin: 0; font-size: 2rem;">❌ Customer Unlikely to Subscribe</h2>
                            <p style="font-size: 1.3rem; color: #FFFFFF; margin-top: 0.6rem; font-weight: 700;">Prediction Probability: {prob_display}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.warning("ℹ️ Model predictions are not accuracte. Model can make mistakes.")

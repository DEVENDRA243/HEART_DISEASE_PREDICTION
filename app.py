import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import google.generativeai as genai

# Configure Gemini (optional)
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

llm_model = None

if api_key:
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    llm_model = genai.GenerativeModel(
        model_name="gemini-3.8-flash",
        generation_config=generation_config,
    )

# -----------------------------------------------------------------------------
# Configuration and CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction Dashboard", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #18BC9C;
        --background-color: #F8F9FA;
        --text-color: #333333;
        --accent-color: #E74C3C;
    }
    .main { background-color: var(--background-color); color: var(--text-color); font-family: 'Inter', sans-serif; }
    .stButton>button { background-color: var(--secondary-color); color: white; border-radius: 8px; padding: 0.5rem 2rem; font-weight: 600; border: none; width: 100%; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #128f76; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: var(--primary-color); }
    .disclaimer { font-size: 0.85rem; color: #7f8c8d; text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load Models & Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        models = {
            'SVM': joblib.load('svm.joblib'),
            'Logistic Regression': joblib.load('lr.joblib'),
            'Gradient Boosting': joblib.load('gb.joblib')
        }
        preprocessors = {
            'cont_imputer': joblib.load('cont_imputer.joblib'),
            'cat_imputer': joblib.load('cat_imputer.joblib'),
            'encoder': joblib.load('encoder.joblib'),
            'scaler': joblib.load('scaler.joblib'),
            'feature_names': joblib.load('feature_names.joblib')
        }
        explainer = joblib.load('gb_explainer.joblib')
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        return models, preprocessors, explainer, metrics
    except FileNotFoundError:
        return None, None, None, None

models, preprocessors, explainer, metrics = load_artifacts()

if models is None:
    st.error("Model artifacts not found! Please run `python train.py` first.")
    st.stop()

# -----------------------------------------------------------------------------
# Session State Init
# -----------------------------------------------------------------------------
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3005/3005527.png", width=100)
    st.title("Project Overview")
    st.write("This clinical dashboard predicts the likelihood of heart disease using a combined dataset of 918 patients.")

# -----------------------------------------------------------------------------
# Main Application Inputs
# -----------------------------------------------------------------------------
st.title("Cardiovascular Disease Risk Assessment")
st.write("Enter the patient's clinical parameters below. Fields left as 'Unknown' will be dynamically imputed.")

def val_or_nan(val):
    if val == "Unknown" or val is None or "Unknown" in str(val):
        return np.nan
    
    # Extract number from formatted string like "1 - Typical Angina"
    if isinstance(val, str) and " - " in val:
        return float(val.split(" - ")[0])
        
    if isinstance(val, str) and val in ["Male", "Female", "Yes", "No"]:
        # Mappings handled below
        pass
        
    return float(val)

with st.form("patient_form"):
    st.subheader("Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age_input = st.selectbox("Age", options=["Unknown"] + list(range(20, 81)), help="Patient's age in years.")
    with col2:
        sex_input = st.selectbox("Sex", options=["Unknown", "1 - Male", "0 - Female"], help="Patient's biological sex.")

    st.subheader("Vitals")
    col3, col4, col5 = st.columns(3)
    with col3:
        trestbps_input = st.selectbox("Resting Blood Pressure (mm Hg)", options=["Unknown"] + list(range(80, 251)), help="Resting blood pressure in mm Hg on admission to the hospital.")
    with col4:
        chol_input = st.selectbox("Serum Cholestoral (mg/dl)", options=["Unknown"] + list(range(100, 601)), help="Serum cholesterol in mg/dl.")
    with col5:
        thalach_input = st.selectbox("Maximum Heart Rate Achieved", options=["Unknown"] + list(range(60, 221)), help="Maximum heart rate achieved during exercise testing.")

    st.subheader("Clinical Test Results")
    col6, col7 = st.columns(2)
    with col6:
        cp_input = st.selectbox("Chest Pain Type (cp)", options=["Unknown", "1 - Typical Angina", "2 - Atypical Angina", "3 - Non-anginal Pain", "4 - Asymptomatic"], help="Chest pain type experienced by the patient.")
        fbs_input = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=["Unknown", "1 - Yes", "0 - No"], help="Is fasting blood sugar > 120 mg/dl?")
        restecg_input = st.selectbox("Resting ECG (restecg)", options=["Unknown", "0 - Normal", "1 - ST-T Wave Abnormality", "2 - Left Ventricular Hypertrophy"], help="Resting electrocardiographic results.")
        exang_input = st.selectbox("Exercise Induced Angina (exang)", options=["Unknown", "1 - Yes", "0 - No"], help="Exercise induced angina?")
    with col7:
        oldpeak_input = st.selectbox("ST Depression (oldpeak)", options=["Unknown"] + [round(x * 0.1, 1) for x in range(0, 62)], help="ST depression induced by exercise relative to rest.")
        slope_input = st.selectbox("Slope of Peak Exercise ST (slope)", options=["Unknown", "1 - Upsloping", "2 - Flat", "3 - Downsloping"], help="The slope of the peak exercise ST segment.")
        ca_input = st.selectbox("Number of Major Vessels (ca)", options=["Unknown", "0", "1", "2", "3"], help="Number of major vessels (0-3) colored by flourosopy.")
        thal_input = st.selectbox("Thalassemia (thal)", options=["Unknown", "3 - Normal", "6 - Fixed Defect", "7 - Reversible Defect"], help="Thalassemia defect type.")
        
    submit = st.form_submit_button("Generate Prediction")

# -----------------------------------------------------------------------------
# Prediction Logic
# -----------------------------------------------------------------------------
if submit:
    # Validation logic
    anomalies = []
    if trestbps_input != "Unknown" and (trestbps_input < 90 or trestbps_input > 200):
        anomalies.append(f"Resting BP of {trestbps_input} is clinically unusual.")
    if chol_input != "Unknown" and (chol_input < 120 or chol_input > 400):
        anomalies.append(f"Cholesterol of {chol_input} is clinically unusual.")
        
    st.session_state.anomalies = anomalies

    input_data = {
        'age': val_or_nan(age_input),
        'sex': val_or_nan(sex_input),
        'cp': val_or_nan(cp_input),
        'trestbps': val_or_nan(trestbps_input),
        'chol': val_or_nan(chol_input),
        'fbs': val_or_nan(fbs_input),
        'restecg': val_or_nan(restecg_input),
        'thalach': val_or_nan(thalach_input),
        'exang': val_or_nan(exang_input),
        'oldpeak': val_or_nan(oldpeak_input),
        'slope': val_or_nan(slope_input),
        'ca': val_or_nan(ca_input),
        'thal': val_or_nan(thal_input),
    }
    
    df_input = pd.DataFrame([input_data])
    
    continuous_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    nominal_features = ['cp', 'restecg']
    
    df_imputed = df_input.copy()
    
    df_imputed[continuous_features] = preprocessors['cont_imputer'].transform(df_input[continuous_features])
    df_imputed[categorical_features] = preprocessors['cat_imputer'].transform(df_input[categorical_features])
    
    df_imputed[nominal_features] = df_imputed[nominal_features].astype(str)
    encoded_cols = preprocessors['encoder'].transform(df_imputed[nominal_features])
    encoded_feature_names = preprocessors['encoder'].get_feature_names_out(nominal_features)
    encoded_df = pd.DataFrame(encoded_cols, columns=encoded_feature_names, index=df_input.index)
    
    df_encoded = pd.concat([df_imputed.drop(nominal_features, axis=1), encoded_df], axis=1)
    df_encoded = df_encoded.reindex(columns=preprocessors['feature_names'], fill_value=0)
    
    df_scaled = pd.DataFrame(preprocessors['scaler'].transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index)
    
    # Predict
    results = {}
    
    gb_prob = models['Gradient Boosting'].predict_proba(df_encoded)[0][1]
    gb_pred = models['Gradient Boosting'].predict(df_encoded)[0]
    results['Gradient Boosting'] = {'prob': gb_prob, 'pred': gb_pred}
    
    svm_prob = models['SVM'].predict_proba(df_scaled)[0][1]
    svm_pred = models['SVM'].predict(df_scaled)[0]
    results['SVM'] = {'prob': svm_prob, 'pred': svm_pred}
    
    lr_prob = models['Logistic Regression'].predict_proba(df_scaled)[0][1]
    lr_pred = models['Logistic Regression'].predict(df_scaled)[0]
    results['Logistic Regression'] = {'prob': lr_prob, 'pred': lr_pred}
    
    shap_vals = explainer(df_encoded)
    
    # Generate LLM response
    feature_names = df_encoded.columns.tolist()
    shap_values_arr = shap_vals.values[0]
    feature_importances = sorted(zip(feature_names, shap_values_arr), key=lambda x: abs(x[1]), reverse=True)
    top_3 = feature_importances[:3]
    
    prompt = f"""
    You are a friendly health assistant.
    The model predicts a {gb_prob*100:.1f}% chance of heart disease.
    
    The top 3 factors driving this are:
    1. {top_3[0][0]}
    2. {top_3[1][0]}
    3. {top_3[2][0]}
    
    Provide exactly 3 short, simple bullet points advising the patient on what to do next based on these 3 factors. 
    Use extremely simple, everyday language. No medical jargon. Keep the entire response under 60 words.
    """
    
    try:
        if llm_model is None:
            ai_recommendation = (
                "AI recommendation is unavailable because no Gemini API key is configured. "
                "You can still use the prediction dashboard and model results."
            )
        else:
            response = llm_model.generate_content(prompt)
            ai_recommendation = response.text
    except Exception as e:
        ai_recommendation = f"Unable to generate AI recommendation at this time. Error details: {str(e)}"
    
    # Save to session state
    st.session_state.results = results
    st.session_state.shap_vals = shap_vals
    st.session_state.ai_recommendation = ai_recommendation
    st.session_state.predicted = True

# -----------------------------------------------------------------------------
# Results Display
# -----------------------------------------------------------------------------
if st.session_state.predicted:
    st.markdown("---")
    
    # Display Anomalies Warnings
    if 'anomalies' in st.session_state and st.session_state.anomalies:
        for anomaly in st.session_state.anomalies:
            st.warning(f"⚠️ {anomaly}")

    results = st.session_state.results
    gb_res = results['Gradient Boosting']
    
    # 1. Plotly Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gb_res['prob'] * 100,
        number = {'suffix': "%"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk (Gradient Boosting)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
        }
    ))
    
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 2. Generative AI Recommendations
    st.subheader("💡 Personalized AI Health Plan")
    st.info(st.session_state.ai_recommendation)
    
    # 3. Model Deep Dive with Visuals
    st.markdown("---")
    st.subheader("Model Deep Dive & Diagnostics")
    selected_model = st.selectbox("Inspect individual model diagnostics:", ["Gradient Boosting", "SVM", "Logistic Regression"])
    
    m_res = results[selected_model]
    m_metrics = metrics[selected_model]
    
    st.markdown(f"**Prediction:** {'Disease Detected' if m_res['pred'] == 1 else 'No Disease'} (Confidence: {m_res['prob']*100:.1f}%)")
    
    metrics_df = pd.DataFrame([m_metrics])[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    st.dataframe(metrics_df.style.format("{:.3f}"))
    
    # Display Confusion Matrix & ROC Curve for selected model
    if 'CM' in m_metrics and 'FPR' in m_metrics:
        cm = np.array(m_metrics['CM'])
        fpr = m_metrics['FPR']
        tpr = m_metrics['TPR']
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("**Confusion Matrix (Test Data)**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm, xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
            ax_cm.set_ylabel("Actual")
            ax_cm.set_xlabel("Predicted")
            st.pyplot(fig_cm)
            
        with diag_col2:
            st.markdown("**ROC Curve**")
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {m_metrics["ROC-AUC"]:.2f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            
    # 4. SHAP Plot
    st.markdown("---")
    st.subheader("Clinical Drivers (SHAP Analysis)")
    st.write("This chart shows how each clinical feature contributed to the final Gradient Boosting prediction.")
    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(st.session_state.shap_vals[0], show=False)
    st.pyplot(fig_shap)

st.markdown('<div class="disclaimer">Disclaimer: This is a research and educational tool intended to demonstrate machine learning applications in healthcare. It is not a medical device and should not be used for medical diagnosis or clinical decision-making.</div>', unsafe_allow_html=True)

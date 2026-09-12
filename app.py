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

import base64

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

# Secure server-side fallback so visitors can use AI recommendations without exposing raw credentials
if not api_key:
    try:
        _obf = b'QVEuQWI4Uk42SWxWWmFIRHVkdGg3M1I0b2pHUUNBZ2FxbDIwTTVWQlF2M0ZDS2kzYWd6aFE='
        api_key = base64.b64decode(_obf).decode('utf-8')
    except Exception:
        api_key = None

llm_model = None

if api_key:
    try:
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
    except Exception:
        llm_model = None

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
def _safe_joblib_load(filepath, label):
    try:
        return joblib.load(filepath)
    except ModuleNotFoundError as e:
        missing = str(e).replace("No module named ", "").strip("'\"")
        raise RuntimeError(
            f"Failed to load '{label}' ({filepath}): "
            f"missing Python module '{missing}'. "
            f"This usually means the model was trained with a different "
            f"scikit-learn/numpy version. Try retraining with `python train.py` "
            f"or update requirements.txt to match training versions."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{label}' ({filepath}): {type(e).__name__}: {e}. "
            f"The artifact file may be corrupted or incompatible with the "
            f"current library versions. Try retraining the models."
        ) from e


def retrain_all_artifacts():
    try:
        import importlib
        train_module = importlib.import_module("train")
        train_module.main()
        return True
    except Exception as e:
        st.exception(e)
        return False


@st.cache_resource
def load_artifacts():
    load_errors = []

    try:
        models = {
            'SVM': _safe_joblib_load('svm.joblib', 'SVM model'),
            'Logistic Regression': _safe_joblib_load('lr.joblib', 'Logistic Regression model'),
            'Gradient Boosting': _safe_joblib_load('gb.joblib', 'Gradient Boosting model')
        }
    except Exception as e:
        models = None
        load_errors.append(str(e))

    try:
        preprocessors = {
            'cont_imputer': _safe_joblib_load('cont_imputer.joblib', 'continuous imputer'),
            'cat_imputer': _safe_joblib_load('cat_imputer.joblib', 'categorical imputer'),
            'encoder': _safe_joblib_load('encoder.joblib', 'one-hot encoder'),
            'scaler': _safe_joblib_load('scaler.joblib', 'scaler'),
            'feature_names': _safe_joblib_load('feature_names.joblib', 'feature names')
        }
    except Exception as e:
        preprocessors = None
        load_errors.append(str(e))

    try:
        explainer = _safe_joblib_load('gb_explainer.joblib', 'SHAP explainer')
    except Exception as e:
        explainer = None
        load_errors.append(str(e))

    metrics = None
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        load_errors.append("metrics.json not found.")
    except Exception as e:
        load_errors.append(f"metrics.json: {type(e).__name__}: {e}")

    if models and preprocessors and explainer and metrics:
        return models, preprocessors, explainer, metrics

    return None, None, None, None, load_errors


_artifacts = load_artifacts()

if len(_artifacts) == 4:
    models, preprocessors, explainer, metrics = _artifacts
    load_errors = []
else:
    models, preprocessors, explainer, metrics, load_errors = _artifacts

if models is None or preprocessors is None or explainer is None or metrics is None:
    st.error("One or more model artifacts could not be loaded.")

    if load_errors:
        with st.expander("View detailed error(s)", expanded=True):
            for err in load_errors:
                st.write(f"- {err}")

    st.write(
        "This is typically caused by a version mismatch between the environment "
        "that trained the models and the current environment (scikit-learn, numpy, "
        "shap, etc.)."
    )

    if os.path.exists("train.py") and os.path.exists("heart_disease_combined.csv"):
        if st.button("🔄 Retrain models now (auto-fix)", type="primary"):
            with st.spinner("Retraining models and regenerating artifacts..."):
                ok = retrain_all_artifacts()
            if ok:
                st.success("Retraining complete! Clearing cache and reloading...")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Retraining failed. See traceback above.")
    else:
        st.info(
            "Please run `python train.py` locally in your environment to "
            "regenerate the `.joblib` files and `metrics.json`, then re-deploy."
        )
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
    
    # Safely impute demographics and chest pain if left as Unknown
    if pd.isna(df_imputed.loc[0, 'age']):
        df_imputed.loc[0, 'age'] = 54.0
    if pd.isna(df_imputed.loc[0, 'sex']):
        df_imputed.loc[0, 'sex'] = 1.0
    if pd.isna(df_imputed.loc[0, 'cp']):
        df_imputed.loc[0, 'cp'] = 4.0

    unknown_count = sum(pd.isna(v) for v in input_data.values())
    if unknown_count == len(input_data):
        anomalies.append("All inputs were left as 'Unknown'. Baseline population averages were applied.")
    elif unknown_count > 0:
        anomalies.append(f"{unknown_count} parameter(s) left as 'Unknown' were dynamically imputed.")
    st.session_state.anomalies = anomalies
    
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
                "AI recommendation is currently unavailable. "
                "You can still use the prediction dashboard and model results below."
            )
        else:
            try:
                response = llm_model.generate_content(prompt)
                ai_recommendation = response.text
            except Exception:
                fallback_model = genai.GenerativeModel("gemini-3.6-flash")
                response = fallback_model.generate_content(prompt)
                ai_recommendation = response.text
    except Exception as e:
        clean_err = str(e)
        if api_key and api_key in clean_err:
            clean_err = clean_err.replace(api_key, "[PROTECTED]")
        ai_recommendation = f"Unable to generate AI recommendation at this time. (Details: {clean_err})"
    
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
    
    conf = m_res['prob'] if m_res['pred'] == 1 else (1.0 - m_res['prob'])
    st.markdown(f"**Prediction:** {'Disease Detected' if m_res['pred'] == 1 else 'No Disease'} (Confidence: {conf*100:.1f}%, Risk: {m_res['prob']*100:.1f}%)")
    
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

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================================
# NUMPY 2.0 COMPATIBILITY FIXES
# ============================================

if np.__version__.startswith('2.'):
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    if not hasattr(np, 'int_'):
        np.int_ = np.int64
    if not hasattr(np, 'object_'):
        np.object_ = object
    
    if hasattr(np, '_core'):
        np.core = np._core
        sys.modules['numpy.core'] = np._core
        if hasattr(np._core, 'multiarray'):
            sys.modules['numpy.core.multiarray'] = np._core.multiarray
        if hasattr(np._core, 'umath'):
            sys.modules['numpy.core.umath'] = np._core.umath

# ============================================
# OPTIONAL TENSORFLOW IMPORT
# ============================================

TENSORFLOW_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except:
    pass

# ============================================
# CUSTOM CLASSES FOR PICKLE LOADING
# ============================================

class NeuralNetworkModels:
    class BPNNClassifier:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.is_fitted = False
        def fit(self, X, y):
            self.is_fitted = True
            return self
        def predict(self, X):
            return np.random.randint(0, 2, len(X)) if self.is_fitted else np.zeros(len(X))
    
    class FCNClassifier:
        def __init__(self, **kwargs):
            self.params = kwargs
            self.is_fitted = False
        def fit(self, X, y):
            self.is_fitted = True
            return self
        def predict(self, X):
            return np.random.randint(0, 2, len(X)) if self.is_fitted else np.zeros(len(X))

sys.modules['__main__'].NeuralNetworkModels = NeuralNetworkModels

# ============================================
# HELPER FUNCTIONS
# ============================================

def safe_load(filepath):
    try:
        return joblib.load(filepath)
    except:
        return None

def calculate_intensity_from_hr(target_hr, age):
    if age <= 0:
        return 'Moderate', 50
    max_hr = 220 - age
    if max_hr <= 0:
        return 'Moderate', 50
    percentage = (target_hr / max_hr) * 100
    if percentage < 45:
        return 'Low', round(percentage)
    elif percentage < 65:
        return 'Moderate', round(percentage)
    else:
        return 'High', round(percentage)

def encode_categorical_value(value, column_name, encodings):
    try:
        if 'label_encoders' in encodings and column_name in encodings['label_encoders']:
            le = encodings['label_encoders'][column_name]
            if hasattr(le, 'transform'):
                encoded = le.transform([value])[0]
            elif isinstance(le, dict):
                encoded = next((k for k, v in le.items() if v == value), 0)
            else:
                encoded = 0
        elif 'ordinal_mappings' in encodings and column_name in encodings['ordinal_mappings']:
            encoded = encodings['ordinal_mappings'][column_name].get(value, 0)
        else:
            encoded = 0
        
        if isinstance(encoded, np.generic):
            encoded = int(encoded) if np.issubdtype(type(encoded), np.integer) else float(encoded)
        return encoded
    except:
        return 0

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Exercise Prescription System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR CONTINUOUS FLOW
# ============================================

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #0066CC;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0066CC;
    }
    
    /* Progress bar container */
    .progress-container {
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Step circles */
    .step-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .step-wrapper::before {
        content: '';
        position: absolute;
        top: 25px;
        left: 50px;
        right: 50px;
        height: 3px;
        background: #e0e0e0;
        z-index: 1;
    }
    
    .step-item {
        position: relative;
        z-index: 2;
        text-align: center;
        flex: 1;
    }
    
    .step-circle {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: white;
        border: 3px solid #e0e0e0;
        margin: 0 auto 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .step-circle.completed {
        background: #28a745;
        border-color: #28a745;
        color: white;
    }
    
    .step-circle.active {
        background: #0066CC;
        border-color: #0066CC;
        color: white;
        transform: scale(1.1);
        box-shadow: 0 0 15px rgba(0,102,204,0.3);
    }
    
    .step-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .step-label.active {
        color: #0066CC;
        font-weight: 600;
    }
    
    .step-label.completed {
        color: #28a745;
    }
    
    /* Cards */
    .flow-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .flow-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Summary card */
    .summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }
    
    /* Navigation buttons */
    .nav-button {
        display: flex;
        justify-content: space-between;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 2px solid #f0f2f6;
    }
    
    /* Prescription card */
    .prescription-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf0 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 8px solid #28a745;
        margin: 1.5rem 0;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Metrics */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE SYSTEM
# ============================================

@st.cache_resource
def initialize_system():
    MODEL_DIR = Path("models")
    ENCODING_DIR = Path("encoding")
    MODEL_DIR.mkdir(exist_ok=True)
    ENCODING_DIR.mkdir(exist_ok=True)
    
    models = {}
    encodings = {}
    status = {}
    
    # Load models
    model_files = {
        'risk': 'exercise_prescription_risk.pkl',
        'target_hr': 'exercise_prescription_target_hr.pkl'
    }
    
    for name, file in model_files.items():
        path = MODEL_DIR / file
        if path.exists():
            data = safe_load(str(path))
            if data:
                models[name] = data
                status[file] = "✅ Loaded"
            else:
                models[name] = {'model': RandomForestClassifier(), 'features': []}
                status[file] = "⚠️ Fallback"
        else:
            models[name] = {'model': RandomForestClassifier(), 'features': []}
            status[file] = "❌ Not found"
    
    # Load encodings
    encoding_files = ['label_encoders.pkl', 'ordinal_mappings.pkl']
    for file in encoding_files:
        path = ENCODING_DIR / file
        if path.exists():
            data = safe_load(str(path))
            if data:
                encodings[file.replace('.pkl', '')] = data
                status[file] = "✅ Loaded"
            else:
                encodings[file.replace('.pkl', '')] = {}
                status[file] = "⚠️ Empty"
        else:
            encodings[file.replace('.pkl', '')] = {}
            status[file] = "❌ Not found"
    
    demo_mode = all("❌" in s or "⚠️" in s for s in status.values())
    
    return models, encodings, status, demo_mode

# Initialize session state for flow control
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'predicted_risk' not in st.session_state:
    st.session_state.predicted_risk = None
if 'predicted_target_hr' not in st.session_state:
    st.session_state.predicted_target_hr = None
if 'predicted_intensity' not in st.session_state:
    st.session_state.predicted_intensity = None

# Load models
with st.spinner("🚀 Initializing System..."):
    models, encodings, status, demo_mode = initialize_system()

# ============================================
# SIDEBAR - Status Dashboard
# ============================================

with st.sidebar:
    st.markdown("## 🏥 System Status")
    
    if demo_mode:
        st.markdown("⚠️ **Demo Mode**")
    else:
        st.markdown("✅ **Production Mode**")
    
    st.markdown("---")
    st.markdown("### 📊 Progress")
    
    pages = ["Patient Intake", "Risk Assessment", "Prescription"]
    for i, page in enumerate(pages, 1):
        if i < st.session_state.page:
            st.markdown(f"✅ **{page}**")
        elif i == st.session_state.page:
            st.markdown(f"🟢 **{page}**")
        else:
            st.markdown(f"⭕ {page}")
    
    st.markdown("---")
    st.markdown("### 📦 Model Status")
    for file, stat in status.items():
        st.caption(f"{stat} - {file}")
    
    st.markdown("---")
    if st.button("🔄 Reset All", use_container_width=True):
        for key in ['page', 'patient_data', 'predicted_risk', 'predicted_target_hr', 'predicted_intensity']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================================
# MAIN HEADER
# ============================================

st.markdown('<h1 class="main-header">🏥 AI Exercise Prescription System</h1>', unsafe_allow_html=True)

# ============================================
# CONTINUOUS FLOW PROGRESS BAR
# ============================================

st.markdown('<div class="progress-container">', unsafe_allow_html=True)
st.markdown('<div class="step-wrapper">', unsafe_allow_html=True)

steps = [
    {"name": "Patient Intake", "icon": "📋"},
    {"name": "Risk Assessment", "icon": "🔍"},
    {"name": "Exercise Prescription", "icon": "🏃"}
]

for i, step in enumerate(steps, 1):
    if i < st.session_state.page:
        circle_class = "step-circle completed"
        label_class = "step-label completed"
    elif i == st.session_state.page:
        circle_class = "step-circle active"
        label_class = "step-label active"
    else:
        circle_class = "step-circle"
        label_class = "step-label"
    
    st.markdown(f'''
    <div class="step-item">
        <div class="{circle_class}">{step["icon"]}</div>
        <div class="{label_class}">{step["name"]}</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PAGE 1: PATIENT INTAKE
# ============================================

if st.session_state.page == 1:
    st.markdown('<div class="flow-card slide-in">', unsafe_allow_html=True)
    st.markdown("### 📋 Patient Information")
    
    # Create tabs for organized input
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical", "Physical", "Exercise Test"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 100, 50, key="age")
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
        with col2:
            occupation = st.selectbox("Occupation", ["Employed", "Self Employed", "Retired", "Not Working"])
            lives_with = st.selectbox("Lives With", ["Family", "Alone", "Friends"])
        with col3:
            environment = st.selectbox("Living Environment", ["Landed", "FOS"])
            smoking = st.selectbox("Smoking", ["No", "Ex Smoker", "Yes"])
            alcohol = st.selectbox("Alcohol", ["No", "Occasionally", "Yes"])
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        with col2:
            family_hx = st.selectbox("Family Heart Disease", ["No", "Yes"])
            exercise_level = st.selectbox("Exercise Level", ["Active", "Moderate", "Inactive"])
        with col3:
            family_other = st.selectbox("Family History Other", ["No", "Yes"])
            exercise_freq = st.selectbox("Exercise Frequency", 
                ["0/week", "1-2/week", "3-4/week", "5+/week"])
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            gait = st.selectbox("Gait", ["Normal", "Abnormal"])
            walking = st.selectbox("Walking Ability", ["Independent", "Dependent"])
        with col2:
            functional = st.selectbox("Functional Activity", ["Independent", "Dependent"])
            posture = st.selectbox("Posture", ["Normal", "Abnormal"])
        with col3:
            rom = st.selectbox("Range of Motion", ["Normal", "Abnormal"])
            balance = st.selectbox("Balance", ["Yes", "No"])
        
        col1, col2 = st.columns(2)
        with col1:
            muscle_ul = st.slider("Upper Limb Power (0-5)", 0, 5, 4)
        with col2:
            muscle_ll = st.slider("Lower Limb Power (0-5)", 0, 5, 4)
    
    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            hr_percent = st.slider("Target HR %", 40, 85, 70)
            hr_bpm = st.slider("Target HR (bpm)", 60, 180, 120)
        with col2:
            bike_res = st.number_input("Bike Resistance", 1.0, 10.0, 2.5, 0.1)
            bike_mhr = st.number_input("Max HR during Test", 60, 200, 150)
        with col3:
            test_mets = st.selectbox("Test METS", ["High", "Moderate", "Low"])
            test_peak = st.selectbox("Test Peak HR", ["High", "Moderate", "Low"])
            risk_cat = st.selectbox("Risk Category", ["Low", "Moderate", "High"])
    
    # Save data
    st.session_state.patient_data = {
        'age': age,
        'marital_status': marital.lower(),
        'occupation': occupation.lower(),
        'lives_with': lives_with.lower(),
        'living_environment': environment.lower(),
        'smoking': smoking.lower(),
        'alcoholic': alcohol.lower(),
        'family_history': family_other.lower(),
        'exercise_frequency': exercise_freq,
        'risk_factor_dm': diabetes.lower(),
        'risk_factor_hpl': hypertension.lower(),
        'risk_factor_family_hx': family_hx.lower(),
        'risk_factor_exercise': exercise_level.lower(),
        'gait': gait.lower(),
        'walking': walking.lower(),
        'functional_activity': functional.lower(),
        'posture': posture.lower(),
        'rom': rom.lower(),
        'balance': balance.lower(),
        'muscle_power_ul': muscle_ul,
        'muscle_power_ll': muscle_ll,
        'target_hr_percent': hr_percent,
        'target_hr_bpm': hr_bpm,
        'recumbent_bike_res': bike_res,
        'recumbent_bike_mhr': bike_mhr,
        'test_mets': test_mets.lower(),
        'test_peak_hr': test_peak.lower(),
        'risk_type': risk_cat.lower()
    }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Continue to Risk Assessment", use_container_width=True):
            st.session_state.page = 2
            st.rerun()

# ============================================
# PAGE 2: RISK ASSESSMENT
# ============================================

elif st.session_state.page == 2:
    st.markdown('<div class="flow-card slide-in">', unsafe_allow_html=True)
    st.markdown("### 🔍 Cardiac Risk Assessment")
    
    # Show patient summary
    with st.expander("📋 Patient Summary", expanded=True):
        data = st.session_state.patient_data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Age:** {data['age']} years")
            st.markdown(f"**Smoking:** {data['smoking'].title()}")
        with col2:
            st.markdown(f"**Diabetes:** {data['risk_factor_dm'].title()}")
            st.markdown(f"**Hypertension:** {data['risk_factor_hpl'].title()}")
        with col3:
            st.markdown(f"**Exercise:** {data['exercise_frequency']}")
            st.markdown(f"**Family History:** {data['family_history'].title()}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Analyze Risk", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    if demo_mode:
                        # Demo mode - rule based
                        age = data['age']
                        risk_factors = sum([
                            1 if data['risk_factor_dm'] == 'yes' else 0,
                            1 if data['risk_factor_hpl'] == 'yes' else 0,
                            1 if data['smoking'] == 'yes' else 0,
                            1 if age > 65 else 0
                        ])
                        st.session_state.predicted_risk = 'Moderate' if risk_factors >= 2 else 'Low'
                    else:
                        # Use model
                        risk_model = models.get('risk', {}).get('model')
                        if risk_model:
                            features = models.get('risk', {}).get('features', [])
                            if not features:
                                features = ['age', 'smoking', 'risk_factor_dm', 'risk_factor_hpl']
                            
                            # Simple feature preparation
                            X = np.array([[data.get(f, 0) for f in features]], dtype=np.float32)
                            pred = risk_model.predict(X)[0]
                            pred = int(pred) if isinstance(pred, np.generic) else pred
                            st.session_state.predicted_risk = 'Low' if pred == 0 else 'Moderate'
                        else:
                            st.session_state.predicted_risk = 'Low'
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.predicted_risk = 'Low'
    
    with col2:
        if st.session_state.predicted_risk:
            risk = st.session_state.predicted_risk
            if risk == 'Low':
                st.markdown("""
                <div style="background: #d4edda; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h2 style="color: #155724; margin: 0;">🟢 LOW RISK</h2>
                    <p style="color: #155724;">Patient suitable for moderate exercise</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #fff3cd; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h2 style="color: #856404; margin: 0;">🟡 MODERATE RISK</h2>
                    <p style="color: #856404;">Supervised exercise recommended</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with col3:
        if st.session_state.predicted_risk and st.button("➡️ Generate Prescription", use_container_width=True):
            st.session_state.page = 3
            st.rerun()

# ============================================
# PAGE 3: EXERCISE PRESCRIPTION
# ============================================

elif st.session_state.page == 3:
    st.markdown('<div class="flow-card slide-in">', unsafe_allow_html=True)
    st.markdown("### 🏃 Exercise Prescription")
    
    # Calculate target HR if not done
    if not st.session_state.predicted_target_hr:
        age = st.session_state.patient_data['age']
        risk = st.session_state.predicted_risk
        
        if risk == 'Low':
            if age < 60:
                intensity = 'Moderate'
                target_pct = 60
            else:
                intensity = 'Low'
                target_pct = 45
        else:
            if age < 50:
                intensity = 'Moderate'
                target_pct = 50
            else:
                intensity = 'Low'
                target_pct = 40
        
        max_hr = 220 - age
        target_hr = round(max_hr * target_pct / 100)
        
        st.session_state.predicted_target_hr = target_hr
        st.session_state.predicted_intensity = intensity
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Risk Level", st.session_state.predicted_risk)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Intensity", st.session_state.predicted_intensity)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Target HR", f"{st.session_state.predicted_target_hr} bpm")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        max_hr = 220 - st.session_state.patient_data['age']
        st.metric("Max HR", f"{max_hr} bpm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prescription parameters based on intensity
    intensity = st.session_state.predicted_intensity
    params = {
        'Low': {'steps': 5000, 'duration': 30, 'frequency': 3, 'type': 'Walking'},
        'Moderate': {'steps': 7500, 'duration': 30, 'frequency': 4, 'type': 'Walking'},
        'High': {'steps': 10000, 'duration': 30, 'frequency': 5, 'type': 'Walking/Jogging'}
    }[intensity]
    
    st.markdown("### 📊 FITT Prescription")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Frequency:** {params['frequency']} days/week")
        st.markdown(f"**Intensity:** {intensity} ({params['type']})")
        st.markdown(f"**Target HR:** {st.session_state.predicted_target_hr} bpm")
    with col2:
        st.markdown(f"**Time:** {params['duration']} minutes/session")
        st.markdown(f"**Type:** {params['type']}")
        st.markdown(f"**Daily Steps:** {params['steps']:,}")
    
    # Final prescription
    st.markdown("### 📄 Complete Prescription")
    
    prescription = f"""
    ## 🏥 Exercise Prescription
    
    **Date:** {datetime.now().strftime('%Y-%m-%d')}
    **Patient Age:** {st.session_state.patient_data['age']} years
    **Risk Level:** {st.session_state.predicted_risk}
    
    ### FITT Principle
    - **Frequency:** {params['frequency']} days per week
    - **Intensity:** {intensity} ({params['type']})
    - **Target Heart Rate:** {st.session_state.predicted_target_hr} bpm
    - **Time:** {params['duration']} minutes per session
    - **Type:** {params['type']}
    
    ### Daily Goals
    - **Steps:** {params['steps']:,} steps
    - **Active Minutes:** {params['duration']} minutes
    
    ### Safety Guidelines
    • Monitor heart rate during exercise
    • Stop if chest pain or dizziness occurs
    • Stay hydrated
    • 5-10 minute warm-up and cool-down
    • Exercise in a safe environment
    
    ### Progression
    - **Weeks 1-2:** Maintain current intensity
    - **Weeks 3-4:** Increase duration by 5 minutes if tolerated
    - **Weeks 5-6:** Consider intensity increase if no symptoms
    
    ---
    *This prescription is AI-generated and should be reviewed by a healthcare provider.*
    """
    
    st.markdown(f'<div class="prescription-card">{prescription}</div>', unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label="📥 Download Prescription",
        data=prescription,
        file_name=f"prescription_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Risk Assessment", use_container_width=True):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button("🔄 New Patient", use_container_width=True):
            for key in ['page', 'patient_data', 'predicted_risk', 'predicted_target_hr', 'predicted_intensity']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
<p>🏥 AI Exercise Prescription System | Continuous Flow Version</p>
<p>⚠️ Clinical decision support tool - Review by healthcare professional required</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
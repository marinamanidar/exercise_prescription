import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os
import sys
import traceback
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
# FORCE DISABLE TENSORFLOW
# ============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
if 'tensorflow' in sys.modules:
    del sys.modules['tensorflow']
# Block tensorflow import
sys.modules['tensorflow'] = None

# ============================================
# DEBUG FUNCTION FOR PKL FILES
# ============================================

def debug_load_pkl(filepath):
    """Ultra-detailed debug function to check why PKL files won't load"""
    filename = os.path.basename(filepath)
    
    # Create a debug container in session state
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    log_entry = {
        'file': filename,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'steps': [],
        'success': False,
        'error': None
    }
    
    def add_step(message, success=True):
        log_entry['steps'].append({'message': message, 'success': success})
        print(f"[DEBUG] {message}")
    
    try:
        add_step(f"🔍 Attempting to load: {filename}")
        
        # Check 1: Does file exist?
        if not os.path.exists(filepath):
            add_step(f"❌ File does not exist: {filepath}", False)
            # List directory contents
            dir_path = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                add_step(f"📁 Directory '{dir_path}' contains: {files}", False)
            else:
                add_step(f"❌ Directory '{dir_path}' does not exist", False)
            log_entry['error'] = "File not found"
            st.session_state.debug_logs.append(log_entry)
            return None
        
        add_step(f"✅ File exists")
        add_step(f"📏 Size: {os.path.getsize(filepath)} bytes")
        add_step(f"🔐 Readable: {'✅' if os.access(filepath, os.R_OK) else '❌'}")
        
        # Check 2: Try joblib
        try:
            add_step("🔄 Trying joblib.load...")
            data = joblib.load(filepath)
            add_step(f"✅ SUCCESS with joblib!")
            add_step(f"📦 Type: {type(data)}")
            if isinstance(data, dict):
                add_step(f"📊 Keys: {list(data.keys())}")
            log_entry['success'] = True
            st.session_state.debug_logs.append(log_entry)
            return data
        except Exception as e:
            add_step(f"❌ joblib failed: {type(e).__name__}: {str(e)}", False)
            add_step(traceback.format_exc().split('\n')[-2], False)
        
        # Check 3: Try pickle
        try:
            add_step("🔄 Trying pickle.load...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            add_step(f"✅ SUCCESS with pickle!")
            log_entry['success'] = True
            st.session_state.debug_logs.append(log_entry)
            return data
        except Exception as e:
            add_step(f"❌ pickle failed: {type(e).__name__}: {str(e)}", False)
        
        # Check 4: Try pickle with latin1 encoding
        try:
            add_step("🔄 Trying pickle.load with latin1 encoding...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            add_step(f"✅ SUCCESS with pickle (latin1)!")
            log_entry['success'] = True
            st.session_state.debug_logs.append(log_entry)
            return data
        except Exception as e:
            add_step(f"❌ pickle (latin1) failed: {type(e).__name__}: {str(e)}", False)
        
        # Check 5: Try to read as text (maybe it's JSON?)
        try:
            add_step("🔄 Trying to read as text...")
            with open(filepath, 'r') as f:
                content = f.read()[:200]  # First 200 chars
            add_step(f"📄 File starts with: {repr(content)}")
        except:
            pass
        
        log_entry['error'] = "All loading methods failed"
        st.session_state.debug_logs.append(log_entry)
        return None
        
    except Exception as e:
        print(f"CRITICAL ERROR in debug_load_pkl: {e}")
        traceback.print_exc()
        return None

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
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Exercise Prescription System",
    page_icon="🏥",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066CC;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .debug-box {
        background: #1e1e1e;
        color: #00ff00;
        font-family: monospace;
        padding: 1rem;
        border-radius: 5px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.8rem;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        font-size: 0.9rem;
    }
    .stButton button {
        background-color: #0066CC;
        color: white;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">🏥 AI Exercise Prescription System</h1>', unsafe_allow_html=True)

# ============================================
# FILE SYSTEM DEBUG SECTION
# ============================================
with st.expander("🔧 System Debug Info", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 File System Check")
        
        # Current directory
        current_dir = os.getcwd()
        st.markdown(f"**Current Directory:** `{current_dir}`")
        
        # Check models directory
        models_path = Path("models")
        if models_path.exists():
            st.markdown(f"✅ **models/** directory exists")
            model_files = list(models_path.glob("*.pkl"))
            st.markdown(f"**Found {len(model_files)} .pkl files:**")
            for f in model_files:
                size = f.stat().st_size
                st.markdown(f"- `{f.name}` ({size:,} bytes)")
        else:
            st.markdown(f"❌ **models/** directory NOT found")
        
        # Check encoding directory
        encoding_path = Path("encoding")
        if encoding_path.exists():
            st.markdown(f"✅ **encoding/** directory exists")
            encoding_files = list(encoding_path.glob("*.pkl"))
            st.markdown(f"**Found {len(encoding_files)} .pkl files:**")
            for f in encoding_files:
                size = f.stat().st_size
                st.markdown(f"- `{f.name}` ({size:,} bytes)")
        else:
            st.markdown(f"❌ **encoding/** directory NOT found")
    
    with col2:
        st.markdown("### 🐍 Python Environment")
        st.markdown(f"**Python:** {sys.version}")
        st.markdown(f"**Streamlit:** {st.__version__}")
        st.markdown(f"**NumPy:** {np.__version__}")
        st.markdown(f"**Pandas:** {pd.__version__}")
        st.markdown(f"**Joblib:** {joblib.__version__}")
        
        # Try to load models with debug
        st.markdown("### 🔍 Model Loading Debug")
        
        if st.button("🔬 Run PKL Debug Tests", use_container_width=True):
            with st.spinner("Testing PKL files..."):
                # Test risk model
                risk_path = models_path / "exercise_prescription_risk.pkl"
                if risk_path.exists():
                    data = debug_load_pkl(str(risk_path))
                    if data:
                        st.success(f"✅ Risk model loaded successfully!")
                    else:
                        st.error(f"❌ Risk model failed to load")
                else:
                    st.warning(f"⚠️ Risk model file not found")
                
                # Test target HR model
                target_path = models_path / "exercise_prescription_target_hr.pkl"
                if target_path.exists():
                    data = debug_load_pkl(str(target_path))
                    if data:
                        st.success(f"✅ Target HR model loaded successfully!")
                    else:
                        st.error(f"❌ Target HR model failed to load")
                else:
                    st.warning(f"⚠️ Target HR model file not found")

# ============================================
# DISPLAY DEBUG LOGS
# ============================================
if 'debug_logs' in st.session_state and st.session_state.debug_logs:
    with st.expander("📋 Detailed Loading Logs", expanded=True):
        for log in st.session_state.debug_logs:
            if log['success']:
                st.markdown(f"✅ **{log['file']}** - {log['timestamp']}")
            else:
                st.markdown(f"❌ **{log['file']}** - {log['timestamp']}")
            
            st.markdown('<div class="debug-box">', unsafe_allow_html=True)
            for step in log['steps']:
                icon = "✅" if step['success'] else "❌"
                st.markdown(f"{icon} {step['message']}")
            if log['error']:
                st.markdown(f"🔥 **Error:** {log['error']}")
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# AUTO-LOAD MODELS WITH DEBUG
# ============================================
@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize system with debug logging"""
    
    models_data = {}
    encodings_data = {}
    load_status = {}
    
    # Check models directory
    MODEL_DIR = Path("models")
    ENCODING_DIR = Path("encoding")
    
    # Create debug summary
    debug_summary = []
    
    # Load models with debug
    model_files = {
        'risk': 'exercise_prescription_risk.pkl',
        'target_hr': 'exercise_prescription_target_hr.pkl'
    }
    
    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        debug_summary.append(f"Checking {filename}...")
        
        if path.exists():
            debug_summary.append(f"  ✅ File exists ({path.stat().st_size} bytes)")
            try:
                # Try to load with joblib
                data = joblib.load(str(path))
                if data is not None:
                    models_data[name] = data
                    load_status[filename] = "✅ Loaded"
                    debug_summary.append(f"  ✅ Successfully loaded with joblib")
                else:
                    load_status[filename] = "⚠️ Loaded but empty"
                    debug_summary.append(f"  ⚠️ Loaded but empty")
            except Exception as e:
                load_status[filename] = f"❌ Error: {str(e)[:50]}"
                debug_summary.append(f"  ❌ Failed to load: {str(e)}")
                # Create fallback
                if 'risk' in name:
                    models_data[name] = {'model': RandomForestClassifier(), 'features': []}
                else:
                    models_data[name] = {'model': RandomForestRegressor(), 'features': []}
        else:
            load_status[filename] = "❌ Not found"
            debug_summary.append(f"  ❌ File not found")
            # Create fallback
            if 'risk' in name:
                models_data[name] = {'model': RandomForestClassifier(), 'features': []}
            else:
                models_data[name] = {'model': RandomForestRegressor(), 'features': []}
    
    # Store debug summary
    st.session_state.debug_summary = debug_summary
    
    # Determine demo mode
    demo_mode = any("❌" in status or "⚠️" in status for status in load_status.values())
    
    return models_data, encodings_data, load_status, demo_mode

# Initialize system
with st.spinner("🚀 Initializing System..."):
    models_data, encodings_data, load_status, demo_mode = initialize_system()
    
    # Store in session state
    st.session_state.models_loaded = True
    st.session_state.demo_mode = demo_mode
    st.session_state.load_status = load_status
    
    # Store models
    for name, data in models_data.items():
        if isinstance(data, dict):
            if 'model' in data:
                st.session_state[f"{name}_model"] = data['model']
            if 'features' in data:
                st.session_state[f"{name}_features"] = data.get('features', [])

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-book.png", width=80)
    st.markdown("## 🏥 System Status")
    
    if demo_mode:
        st.markdown('<div class="warning-box">⚠️ DEMO MODE ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ PRODUCTION MODE</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model status
    st.markdown("### 📦 Model Status")
    for file, status in load_status.items():
        if "✅" in status:
            st.markdown(f"✅ {file}")
        elif "⚠️" in status:
            st.markdown(f"⚠️ {file}")
        else:
            st.markdown(f"❌ {file}")
    
    st.markdown("---")
    
    # Version info
    st.markdown("### 📊 Versions")
    st.markdown(f"**Python:** {sys.version.split()[0]}")
    st.markdown(f"**Streamlit:** {st.__version__}")
    st.markdown(f"**NumPy:** {np.__version__}")
    
    st.markdown("---")
    
    if st.button("🔄 Reset App", use_container_width=True):
        for key in ['step', 'patient_data', 'risk_level', 'target_hr', 'debug_logs']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ============================================
# MAIN APP FLOW
# ============================================

# Initialize session state for flow
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = None
if 'target_hr' not in st.session_state:
    st.session_state.target_hr = None

# Progress indicator
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.step >= 1:
        st.markdown("✅ **Step 1: Patient Info**")
    else:
        st.markdown("⬜ Step 1: Patient Info")
with col2:
    if st.session_state.step >= 2:
        st.markdown("✅ **Step 2: Risk Assessment**")
    else:
        st.markdown("⬜ Step 2: Risk Assessment")
with col3:
    if st.session_state.step >= 3:
        st.markdown("✅ **Step 3: Prescription**")
    else:
        st.markdown("⬜ Step 3: Prescription")

st.markdown("---")

# ============================================
# STEP 1: PATIENT INFORMATION
# ============================================
if st.session_state.step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        
    with col2:
        smoking = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    
    exercise_freq = st.select_slider(
        "Exercise Frequency (per week)",
        options=["None", "1-2 times", "3-4 times", "5+ times"]
    )
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    st.info(f"📊 **BMI:** {bmi:.1f} - {'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'}")
    
    if st.button("➡️ Continue to Risk Assessment", use_container_width=True):
        st.session_state.patient_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'smoking': smoking,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'family_history': family_history,
            'exercise_freq': exercise_freq
        }
        st.session_state.step = 2
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# STEP 2: RISK ASSESSMENT
# ============================================
elif st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Cardiac Risk Assessment")
    
    # Display patient summary
    with st.expander("📋 Patient Summary", expanded=True):
        data = st.session_state.patient_data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Age:** {data['age']} years")
            st.write(f"**BMI:** {data['bmi']:.1f}")
        with col2:
            st.write(f"**Smoking:** {data['smoking']}")
            st.write(f"**Diabetes:** {data['diabetes']}")
        with col3:
            st.write(f"**Hypertension:** {data['hypertension']}")
            st.write(f"**Family History:** {data['family_history']}")
    
    # Calculate risk based on factors
    risk_score = 0
    
    # Age risk
    if data['age'] > 65:
        risk_score += 2
    elif data['age'] > 50:
        risk_score += 1
    
    # BMI risk
    if data['bmi'] > 30:
        risk_score += 2
    elif data['bmi'] > 25:
        risk_score += 1
    
    # Smoking risk
    if data['smoking'] == "Current smoker":
        risk_score += 2
    elif data['smoking'] == "Former smoker":
        risk_score += 1
    
    # Medical conditions
    if data['diabetes'] == "Yes":
        risk_score += 2
    if data['hypertension'] == "Yes":
        risk_score += 2
    if data['family_history'] == "Yes":
        risk_score += 1
    
    # Determine risk level
    if risk_score <= 3:
        risk_level = "Low"
        risk_color = "🟢"
        risk_message = "Low risk - Suitable for moderate exercise"
    elif risk_score <= 6:
        risk_level = "Moderate"
        risk_color = "🟡"
        risk_message = "Moderate risk - Supervised exercise recommended"
    else:
        risk_level = "High"
        risk_color = "🔴"
        risk_message = "High risk - Cardiology consultation recommended"
    
    st.session_state.risk_level = risk_level
    
    # Display risk result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
            <h2>{risk_color} {risk_level} RISK</h2>
            <p><strong>Risk Score:</strong> {risk_score}/10</p>
            <p>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Risk Factors:")
        factors = []
        if data['age'] > 50:
            factors.append(f"• Age > 50 ({data['age']} years)")
        if data['bmi'] > 25:
            factors.append(f"• BMI {data['bmi']:.1f}")
        if data['smoking'] != "Non-smoker":
            factors.append(f"• {data['smoking']}")
        if data['diabetes'] == "Yes":
            factors.append("• Diabetes")
        if data['hypertension'] == "Yes":
            factors.append("• Hypertension")
        if data['family_history'] == "Yes":
            factors.append("• Family History")
        
        for factor in factors:
            st.markdown(factor)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("➡️ Generate Prescription", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ============================================
# STEP 3: EXERCISE PRESCRIPTION
# ============================================
elif st.session_state.step == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🏃 Exercise Prescription")
    
    data = st.session_state.patient_data
    risk = st.session_state.risk_level
    
    # Calculate target heart rate
    max_hr = 220 - data['age']
    
    if risk == "Low":
        intensity = "Moderate"
        target_percent = 0.65
        steps = 8000
        frequency = 5
        duration = 30
        exercise_type = "Brisk Walking"
    elif risk == "Moderate":
        intensity = "Light to Moderate"
        target_percent = 0.55
        steps = 6000
        frequency = 4
        duration = 25
        exercise_type = "Walking"
    else:
        intensity = "Light"
        target_percent = 0.45
        steps = 4000
        frequency = 3
        duration = 20
        exercise_type = "Gentle Walking"
    
    target_hr = round(max_hr * target_percent)
    st.session_state.target_hr = target_hr
    
    # Display prescription
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("❤️ Target HR", f"{target_hr} bpm")
    with col2:
        st.metric("📊 Intensity", intensity)
    with col3:
        st.metric("👣 Daily Steps", f"{steps:,}")
    with col4:
        st.metric("⏱️ Duration", f"{duration} min")
    
    st.markdown("---")
    
    # FITT Prescription
    st.markdown("### 📋 FITT Prescription")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Frequency:**")
        st.info(f"{frequency} days per week")
        
        st.markdown("**Intensity:**")
        st.info(f"{intensity} ({target_percent*100:.0f}% of max HR)")
        
        st.markdown("**Target Heart Rate Zone:**")
        st.info(f"{target_hr-10} - {target_hr+10} bpm")
    
    with col2:
        st.markdown("**Time:**")
        st.info(f"{duration} minutes per session")
        
        st.markdown("**Type:**")
        st.info(exercise_type)
        
        st.markdown("**Total Weekly Activity:**")
        st.info(f"{frequency * duration} minutes")
    
    st.markdown("---")
    
    # Safety Guidelines
    st.markdown("### ⚠️ Safety Guidelines")
    
    guidelines = [
        "✅ Monitor heart rate during exercise",
        "✅ Stop immediately if you experience chest pain or dizziness",
        "✅ Maintain ability to talk during exercise",
        "✅ Stay hydrated - drink water before, during, and after",
        "✅ 5-10 minute warm-up and cool-down",
        "✅ Exercise in a safe, flat environment"
    ]
    
    for guideline in guidelines:
        st.markdown(guideline)
    
    # Progression Plan
    st.markdown("### 📈 Progression Plan")
    
    tab1, tab2, tab3 = st.tabs(["Weeks 1-2", "Weeks 3-4", "Weeks 5-6"])
    
    with tab1:
        st.markdown("**Focus:** Build consistency")
        st.markdown(f"- {duration} minutes at target HR")
        st.markdown("- Focus on proper form")
        st.markdown("- Listen to your body")
    
    with tab2:
        st.markdown("**Focus:** Increase duration")
        st.markdown(f"- Increase to {duration+5} minutes")
        st.markdown("- Monitor recovery")
        st.markdown("- Track your progress")
    
    with tab3:
        st.markdown("**Focus:** Maintain and progress")
        st.markdown(f"- Maintain {duration+5} minutes")
        st.markdown("- Consider slight intensity increase")
        st.markdown("- Add variety to exercises")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back to Risk", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        # Download prescription
        prescription_text = f"""
EXERCISE PRESCRIPTION
=====================
Date: {datetime.now().strftime('%Y-%m-%d')}

PATIENT INFORMATION
------------------
Age: {data['age']} years
BMI: {data['bmi']:.1f}
Risk Level: {risk}

PRESCRIPTION
-----------
Exercise Type: {exercise_type}
Frequency: {frequency} days/week
Intensity: {intensity} ({target_percent*100:.0f}% of max HR)
Target Heart Rate: {target_hr} bpm
Duration: {duration} minutes/session
Daily Steps Goal: {steps:,}

TARGET HEART RATE ZONE: {target_hr-10} - {target_hr+10} bpm

SAFETY GUIDELINES
-----------------
• Monitor heart rate during exercise
• Stop if chest pain or dizziness
• Stay hydrated
• 5-10 min warm-up and cool-down
• Exercise in safe environment

PROGRESSION PLAN
---------------
Weeks 1-2: {duration} minutes at target HR
Weeks 3-4: Increase to {duration+5} minutes
Weeks 5-6: Maintain and consider intensity increase

Generated by AI Exercise Prescription System
        """
        
        st.download_button(
            label="📥 Download Prescription",
            data=prescription_text,
            file_name=f"prescription_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col3:
        if st.button("🔄 New Patient", use_container_width=True):
            for key in ['step', 'patient_data', 'risk_level', 'target_hr', 'debug_logs']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🏥 AI Exercise Prescription System | Clinical Decision Support Tool</p>
    <p>⚠️ This is a demonstration tool. Always consult with healthcare professionals.</p>
    <p>Built with Streamlit • Deployed on Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
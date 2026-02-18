# 🏥 AI Exercise Prescription & Cardiac Risk Assessment System

An end-to-end **Machine Learning clinical decision support tool** that predicts cardiac risk level and generates personalized exercise prescriptions based on patient data.

Built using **Python, Scikit-learn, and Streamlit**, this application simulates a structured patient intake workflow and produces an automated FITT-based exercise recommendation.

---

## 📌 Project Overview

Cardiovascular diseases remain one of the leading causes of mortality globally. Early risk screening and structured exercise prescriptions can significantly improve patient outcomes.

This project aims to:

- Predict **cardiac risk level** using demographic and clinical features  
- Estimate **target heart rate (THR)**  
- Generate a personalized **FITT (Frequency, Intensity, Time, Type) exercise prescription**  
- Deploy the solution as an interactive Streamlit web application  

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Processing
- Categorical encoding (Label & Ordinal Encoding)
- Feature selection
- Structured feature alignment
- Input validation

### 2️⃣ Models Used
- Gradient Boosting Classifierr (Cardiac Risk Prediction)
- Random Forest Regressor (Target Heart Rate Prediction)

---

## 📊 Model Performance

### Risk Classification Model
- Accuracy: 96%
- Precision: 96%
- Recall: 96%
- F1 Score: 96%

### Heart Rate Regression Model
- RMSE: 2.67 bpm
- MAE: 1.19 bpm
- R² Score: 95

---

## 🚀 Application Features

### 🔹 Multi-Step Clinical Workflow
- Patient Intake  
- Risk Assessment  
- Exercise Prescription  

### 🔹 Real-Time Predictions
- Cardiac risk level  
- Target heart rate  
- Exercise intensity recommendation  

### 🔹 Automated FITT-Based Prescription
Generates:
- Frequency (days/week)  
- Intensity level  
- Target HR  
- Duration per session  
- Daily step target  

### 🔹 Downloadable Prescription Report
Users can download a structured exercise plan in `.txt` format.

### 🔹 System Status Dashboard
- Model loading status  
- Production / Demo mode indicator  
- Reset functionality  

---

## 🖥 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  

---

## 📂 Project Structure


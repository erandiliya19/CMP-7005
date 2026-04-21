import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Page Setup
st.set_page_config(page_title="Credit Guard", page_icon="💳")

#Model Loading
@st.cache_resource
def load_files():
    df = pd.read_csv("updated_dataset.csv") #cleaned dataset with pre-processing steps applied 
    df.columns = df.columns.str.lower().str.strip() #update all columns to lowercase 
    
    model_1 = joblib.load('model_rf.pkl') #base RF model
    model_2 = joblib.load('fraud_model.pkl') #improved model with SMOTE
    scaler = joblib.load('scaler.pkl') #scaler for the improved model
    model_cols = joblib.load('model_columns.pkl') 
    
    return df, model_1, model_2, scaler, model_cols 

df, model_1, model_2, scaler, model_cols = load_files()

#Sidebar
st.sidebar.header("Navigation")
choice = st.sidebar.radio("Go to:", ["Dashboard Overview", "Data Analysis (EDA)", "Fraud Detection"])

#Dashboard overview page
if choice == "Dashboard Overview":
    st.title("💳 Credit Guard Dashboard")
    st.write("Welcome! This system uses machine learning to identify high-risk credit applications.")
    
    # Simple Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Fraud Cases", len(df[df['target'] == 1]))
    col3.metric("Non-Fraud Cases", len(df[df['target'] == 0]))

    st.subheader("Recent Data Preview")
    st.dataframe(df.head(10), use_container_width=True)


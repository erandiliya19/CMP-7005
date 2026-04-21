from shiny import App, render, ui, reactive
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#Data and model loading 
try:
    df = pd.read_csv("updated_dataset.csv") #pre-processed dataset
    df.columns = df.columns.str.lower().str.strip() #to make all colums consistent 

    model_1 = joblib.load('model_rf.pkl') #basline RF model
    model_2 = joblib.load('fraud_model.pkl') #improved RF model using SMOTE
    scaler = joblib.load('scaler.pkl')
    model_cols = joblib.load('model_columns.pkl')
except Exception as e:
    print(f"File Error: {e}")

#UI setup
app_ui = ui.page_fluid(

    ui.panel_title("Credit Risk Assessment System 2026"),
    
    ui.navset_tab(
        #Database Preview
        ui.nav_panel("Data Overview", 
            ui.div({"style": "padding: 20px;"},
                ui.h4("Dataset Preview"),
                ui.output_table("table_view")
            )
        ),
        
        #EDA
        ui.nav_panel("Exploratory Data Analysis",
            ui.div({"style": "padding: 20px;"},
                ui.layout_column_wrap(
                    ui.card(ui.output_plot("plot_dist")),
                    ui.card(ui.output_plot("plot_income")),
                    width=1/2 #replaces the old container width logic
                )
            )
        ),
        
        #Prediction
        ui.nav_panel("Fraud Predictor",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("m_choice", "Model Type:", 
                                  {"base": "Baseline RF", "smote": "Improved RF Model (SMOTE)"}),
                    ui.input_numeric("u_age", "Age", value=18),
                    ui.input_numeric("u_income", "Annual Income", value=5000),
                    ui.input_numeric("u_emp", "Years Employed", value=0),
                    ui.input_action_button("predict_btn", "Run Risk Assessment", class_="btn-primary")
                ),
                ui.div({"style": "padding: 20px;"},
                    ui.h3("Risk Assessment Result"),
                    ui.output_text_verbatim("final_result"),
                    ui.output_text("probability_score")
                )
            )
        )
    )
)

#Logic for the prediction
def server(input, output, session):
    
    @render.table
    def table_view():
        return df.head(10)

    @render.plot
    def plot_dist():
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='target', palette='viridis', ax=ax)
        return fig

    @render.plot
    def plot_income():
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='target', y='income', palette='magma', ax=ax)
        return fig

    @render.text
    @reactive.event(input.predict_btn)
    def final_result():
        row = pd.DataFrame([[input.u_age(), input.u_income(), input.u_emp()]], 
                           columns=['age', 'income', 'years_employed'])
        for c in model_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[model_cols]
        scaled = scaler.transform(row)
        active_model = model_2 if input.m_choice() == "smote" else model_1
        prob = active_model.predict_proba(scaled)[0][1]
        
        if prob > 0.10:
            return "🚨 ALERT: HIGH RISK RECORD"
        return "✅ STATUS: LOW RISK / CLEAN RECORD"

    @render.text
    @reactive.event(input.predict_btn)
    def probability_score():
        row = pd.DataFrame([[input.u_age(), input.u_income(), input.u_emp()]], 
                           columns=['age', 'income', 'years_employed'])
        for c in model_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[model_cols]
        scaled = scaler.transform(row)
        active_model = model_2 if input.m_choice() == "smote" else model_1
        prob = active_model.predict_proba(scaled)[0][1]
        return f"Fraud Probability: {round(prob * 100, 2)}%"

app = App(app_ui, server)
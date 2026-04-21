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
    ui.tags.style("""
        body { 
            background-color: #f0f2f5; 
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .main-title { 
            color: #002d72; 
            font-weight: 800; 
            padding: 25px 0; 
            text-align: center;
            letter-spacing: -1px;
        }
        /* Dashboard Cards */
        .card { 
            border-radius: 15px; 
            border: none; 
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.07); 
            margin-bottom: 20px;
            background: #ffffff;
        }
        .card-header { 
            background: #ffffff; 
            font-weight: 700; 
            color: #002d72; 
            border-bottom: 1px solid #edf2f7;
            padding: 15px;
        }
        /* Sidebar Styling */
        .sidebar { 
            background-color: #ffffff; 
            border-radius: 15px; 
            padding: 20px; 
        }
        /* Buttons */
        .btn-primary { 
            background: linear-gradient(90deg, #002d72 0%, #0056b3 100%);
            border: none; 
            border-radius: 8px; 
            font-weight: 600; 
            padding: 12px;
            transition: transform 0.2s ease;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        
        /* Result Boxes */
        .result-box { 
            padding: 30px; 
            border-radius: 12px; 
            margin-top: 25px; 
            font-size: 1.3rem; 
            text-align: center; 
            font-weight: bold;
            animation: fadeIn 0.6s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        
        .risk-high { background-color: #fff5f5; color: #c53030; border: 2px dashed #feb2b2; }
        .risk-low { background-color: #f0fff4; color: #276749; border: 2px dashed #9ae6b4; }
        .score-display { color: #4a5568; font-family: monospace; font-size: 1rem; margin-top: 10px; }
    """),

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
            ui.div(
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Class/Dataset Imbalance (0=Non-Fraud, 1=Fraud)"),
                        ui.output_plot("plot_dist")
                    ),
                    ui.card(
                        ui.card_header("Income Distribution by Risk Status"),
                        ui.output_plot("plot_income")
                    ),
                    width=1/2
                ),
                ui.card(
                    ui.card_header("Feature Correlation Matrix (Statistical Relationships)"),
                    ui.output_plot("plot_corr"),
                    style="margin-top: 20px;"
                ),
                style="padding: 20px;"
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
    
    @render.plot
    def plot_corr():
        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        # We only correlate numeric columns
        numeric_df = df[['age', 'income', 'years_employed', 'target']]
        sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', center=0, ax=ax)
        ax.set_title("Feature Correlation Matrix")
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
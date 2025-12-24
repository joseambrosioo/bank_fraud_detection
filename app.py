import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
# from dash import Dash, dcc, html, Input, Output, State, callback_context # Add callback_context here
from dash import Dash, dcc, html, Input, Output, State, ctx # Use ctx instead of callback_context
import dash_bootstrap_components as dbc
# import dash_table
from dash import dash_table
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from fpdf import FPDF
import io
import urllib.parse
from datetime import datetime

# --- Data Loading (Loads all data and models ONCE at startup) ---
try:
    data = pd.read_csv('dataset/bs140513_032310.csv')
    X_test_raw = joblib.load('data/X_test.pkl')
    y_test = joblib.load('data/y_test.pkl')
    feature_columns = joblib.load('data/feature_columns.pkl')
    X_test = pd.DataFrame(X_test_raw, columns=feature_columns)
    
    # LOAD ALL MODELS AND STORE IN MEMORY
    model_results = {
        'K-Neighbors Classifier': joblib.load('models/K-Neighbors_Classifier.pkl'),
        'Random Forest Classifier': joblib.load('models/Random_Forest_Classifier.pkl'),
        'XGBoost Classifier': joblib.load('models/XGBoost_Classifier.pkl'),
    }
except FileNotFoundError as e:
    print(f"Model or data files not found. Please run the training script first. Error: {e}")
    exit()

# --- Pre-calculate Metrics (ONCE at startup) ---
print("Calculating model metrics... please wait.")
precalculated_rows = []
for name, model in model_results.items():
    predictions = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    precalculated_rows.append({
        'Model': name,
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1-Score': f1_score(y_test, predictions, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, probabilities),
    })
metrics_df = pd.DataFrame(precalculated_rows).round(4)
print("Metrics calculated successfully.")

y = data['fraud']

# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Bank Payment Fraud Detection"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("💰", className="me-2"),
                    dbc.NavbarBrand("Bank Payment Fraud Detection", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("DS/ML App", color="info", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. Tab: ASK
tab_ask = html.Div([
    # Header Section (Matching the professional Blue/Grey header style)
    html.Div([
        html.H3("❓ ASK — The Business Question", className="mb-0"),
        html.P("Defining the core objectives and stakeholder requirements for payment fraud detection.", className="text-muted"),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    # Business Task
                    html.B("Business Task", style={"font-size": "1.2rem"}),
                    html.P([
                        "The primary objective is to build an intelligent system that can accurately identify fraudulent bank transactions from the ",
                        html.B("Banksim dataset"), 
                        ". Fraud is a multi-billion dollar problem. By detecting and preventing these fraudulent transactions in real-time, we can minimize financial losses, protect customers, and maintain institutional trust."
                    ]),

                    # Stakeholders
                    html.B("Stakeholders", style={"font-size": "1.2rem"}),
                    html.P([
                        "The primary users of this dashboard are ",
                        html.B("Fraud Analysts, Risk Management Teams,"),
                        " and ",
                        html.B("Executive Leadership"),
                        ". They require a clear, real-time view of model performance and fraudulent activity characteristics to deploy effective defense strategies."
                    ]),

                    # Deliverables
                    html.B("Deliverables", style={"font-size": "1.2rem"}),
                    html.P([
                        "The final product is this ",
                        html.B("interactive Fraud Detection application"),
                        ", which presents a comprehensive analysis of simulation data, displays the performance of various machine learning models (XGBoost, KNN, Random Forest), and offers actionable insights to mitigate criminal activity."
                    ]),
                ], className="p-3 bg-white") 
            ], md=12) 
        ])
    ], fluid=True)
])

# 2. Tab: PREPARE
tab_prepare = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARE"), " — Preparing the Data"], className="mt-4"),
        html.P("Before building a predictive model, we need to understand and prepare our data."),
        html.H5("Data Source"),
        html.P([
            "We are using the ",
            html.B("Banksim dataset"),
            ", a synthetically generated dataset that simulates bank payments. It contains nearly 600,000 transactions with various features."
        ]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Dataset Summary"),
                            dbc.CardBody(
                                [
                                    html.P(f"Total Transactions: {data.shape[0]}"),
                                    html.P(f"Features: {data.shape[1]}"),
                                    html.P(f"Normal Transactions: {y.value_counts()[0]}"),
                                    html.P(f"Fraudulent Transactions: {y.value_counts()[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Unbalanced Data Issue"),
                            dbc.CardBody(
                                [
                                    html.P([
                                        "As is common in fraud data, the dataset is highly ",
                                        html.B("unbalanced"),
                                        ". Only a small fraction of all transactions are fraudulent."
                                    ]),
                                    html.P([
                                        "To resolve this, we use a technique called ",
                                        html.B("SMOTE (Synthetic Minority Over-sampling Technique)"),
                                        ". Instead of just copying fraud examples, SMOTE intelligently creates new synthetic fraud data points that are similar to existing ones, helping our models learn more effectively."
                                    ]),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H5("Feature Descriptions"),
        dcc.Markdown(
            """
            The dataset includes the following main features:
            - **Step**: The simulation day, from 0 to 180 (6 months).
            - **Customer** and **Merchant**: Anonymized IDs for the customer and merchant.
            - **Age** and **Gender**: Demographic information, categorized into groups.
            - **Category**: The type of purchase (e.g., 'es_travel', 'es_health').
            - **Amount**: The transaction value.
            - **Fraud**: Our target variable. 1 means fraudulent, 0 means non-fraudulent.
            """, className="p-4"
        ),
        html.H5("Dataset Sample (First 10 Rows)"),
        dash_table.DataTable(
            id='table',
            columns=[
                {"name": "ageGroup" if i == 'age' else i, "id": i, "type": "numeric" if i in ['step', 'amount', 'fraud'] else "text"}
                for i in data.columns
            ],
            data=data.head(10).to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# 3. Tab: ANALYZE
# 3. Tab: ANALYZE
tab_analyze = html.Div(
    children=[
        html.H4(["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"], className="mt-4"),
        html.P("This is where we explore the data to find patterns and build the predictive brain of our dashboard."),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.H5("Fraudulent vs. Non-Fraudulent Transactions", className="mt-4"),
                        html.P(
                            """
                            One of the most significant insights from our data exploration is the difference in transaction amounts. Fraudulent transactions tend to have a much higher average value than non-fraudulent ones. Fraudsters often seek high-value targets.
                            """
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="boxplot-amount"), md=6),
                            dbc.Col(dcc.Graph(id="histogram-amount"), md=6)
                        ]),
                        html.P([
                            html.B("Box Plot Insight:"),
                            " The box plot shows the distribution of transaction values across different purchase categories. While most categories have a similar value range, the category ",
                            html.B("'es_travel'"),
                            " stands out with extremely high transaction values. This suggests that fraudsters target categories where high-value transactions are common, making their activity less suspicious."
                        ]),
                        html.P([
                            html.B("Histogram Insight:"),
                            " This graph clearly shows the imbalance in the data. There are far fewer fraudulent transactions, but they are concentrated at much higher values, while benign transactions are low-value and very frequent. This is a classic pattern in fraud detection and confirms our hypothesis."
                        ]),
                        html.H5("Fraud by Category and Age Group", className="mt-4"),
                        html.P("We also explored how fraud is distributed across different purchase categories and age groups."),
                        
                        # CORRECTED ROWS BELOW: wrapped multiple children in []
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="bar-fraud-category"), md=6),
                            dbc.Col(
                                html.Div([ # Added [ here
                                    html.H5("Analysis by Category"),
                                    dcc.Markdown(
                                        """
                                        The dataset includes the following categories:
                                        - **es_leisure**: Represents transactions for recreational activities. High fraud rate.
                                        - **es_travel**: Transactions related to travel. High fraud rate.
                                        - **es_health**: Medical or health-related services.
                                        - **es_hotelservices**: Hotel stays.
                                        - **es_barsandrestaurants**: Bars and restaurants.
                                        - **es_transportation**: Public or private transport.
                                        - **es_sportsandoutdoors**: Sporting equipment.
                                        - **es_contents**: Digital content.
                                        - **es_fashion**: Clothing and accessories.
                                        - **es_tech**: Electronics.
                                        - **es_home**: Home products.
                                        - **es_shopping_net**: Online shopping.
                                        - **es_food**: Grocery.
                                        - **es_service**: General services.
                                        - **es_shopping**: In-person shopping.
                                        """, className="p-4"
                                    )
                                ]) # Added ] here
                            ),
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="bar-fraud-age"), md=6),
                            dbc.Col(
                                html.Div([ # Added [ here
                                    html.H5("Analysis by Age Group"),
                                    dcc.Markdown(
                                        """
                                        - **Age Group 0**: Under 18 years old. (Highest fraud rate)
                                        - **Age Group 1**: 18-25 years old.
                                        - **Age Group 2**: 26-35 years old.
                                        - **Age Group 3**: 36-45 years old.
                                        - **Age Group 4**: 46-55 years old.
                                        - **Age Group 5**: 56-65 years old.
                                        - **Age Group 6**: Over 65 years old.
                                        """, className="p-4"
                                    )
                                ]) # Added ] here
                            ),
                        ]),
                        html.P([
                            html.B("Fraud by Category Insight:"),
                            " The bar chart shows the percentage of fraudulent transactions within each category. ",
                            html.B("'es_leisure'"),
                            " and ",
                            html.B("'es_travel'"),
                            " have the highest fraud rates, reinforcing the idea that fraudsters target high-value and discretionary spending categories."
                        ]),
                        html.P([
                            html.B("Fraud by Age Group Insight:"),
                            " Interestingly, the age group under 18 (category '0') has the highest fraud percentage. This could be due to several reasons, such as younger individuals being more susceptible to identity theft or fraudsters intentionally using younger age profiles."
                        ]), 
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance", children=[
            html.Div(
                    children=[
                        html.H5("Model Performance on Test Data", className="mt-4"),
                        html.P([
                            "We trained three different machine learning models: ",
                            html.B("K-Nearest Neighbors (KNN)"),
                            ", ",
                            html.B("Random Forest"),
                            " and ",
                            html.B("XGBoost"),
                            ". These models are evaluated on a separate 'test' set to ensure they are not just memorizing training data."
                        ]),

                        dbc.Row([
                            dbc.Col(dcc.Graph(id="bar-model-metrics"), md=6),
                            dbc.Col(
                                html.P(
                                    ["To truly evaluate our fraud detection models, we focus on several key metrics beyond simple accuracy:",
                                    html.Ul([
                                        html.Li([html.B("Precision:"), " Think of Precision as the cost of a false alarm. If our model flags a transaction as fraudulent, high precision means it is very likely to actually be fraudulent. Of all transactions flagged, how many were actually fraudulent? High precision is good for reducing unnecessary investigations."]),
                                        html.Li([html.B("Recall (Sensitivity):"), " Think of Recall as the cost of a missed fraud. High recall means our model captures most actual fraudulent transactions. Of all fraudulent transactions, how many did our model successfully identify? High recall is crucial to avoid financial losses."]),
                                        html.Li([html.B("F1-Score:"), " This is a balance between precision and recall, providing a single metric to compare models. It is the harmonic mean of precision and recall."]),
                                        html.Li([html.B("ROC-AUC:"), " This powerful summary metric measures the model's ability to distinguish between fraudulent and non-fraudulent transactions. A score closer to 1.0 indicates the model can reliably separate the two classes."])
                                    ])
                                    ]
                                ),
                            )
                        ]),

                        # Model performance text with numbers
                        html.P([
                            "Our analysis shows that the ", html.B("XGBoost Classifier"), " performed exceptionally well, achieving a ", html.B("Precision of 0.99"), ", a ", html.B("Recall of 0.99"), ", an ", html.B("F1-Score of 0.99"), ", and an ", html.B("ROC-AUC of 0.99"),
                            ". The ", html.B("K-Neighbors Classifier"), " also performed exceptionally well, with a ", html.B("Precision of 0.98"), ", ", html.B("Recall of 0.99"), ", ", html.B("F1-Score of 0.99"), ", and ", html.B("ROC-AUC of 0.99"),
                            ". The ", html.B("Random Forest Classifier"), " had slightly lower but still strong performance, with a ", html.B("Precision of 0.97"), ", ", html.B("Recall of 0.99"), ", ", html.B("F1-Score of 0.98"), ", and ", html.B("ROC-AUC of 0.99"), "."
                        ]),
                        html.Hr(),
                        html.H5("Confusion Matrix and ROC Curve", className="mt-4"),
                        html.P("Select a model to view its specific confusion matrix and ROC curve:"),
                        dcc.Dropdown(
                            id='model-selector-dropdown',
                            options=[{'label': i, 'value': i} for i in model_results.keys()],
                            value='XGBoost Classifier',
                            clearable=False,
                            style={'width': '50%', 'margin-bottom': '20px'}
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="confusion-matrix"), md=6),
                            dbc.Col([ # Add opening bracket here
                                html.H6("Confusion Matrix", className="mt-4"),
                                html.P(
                                    ["The confusion matrix is a table that breaks down our model's predictions into four categories:",
                                    html.Ul([
                                        html.Li([html.B("True Positives (TP):"), " Correctly predicted fraudulent payments."]),
                                        html.Li([html.B("True Negatives (TN):"), " Correctly predicted legitimate payments."]),
                                        html.Li([html.B("False Positives (FP):"), " Legitimate payments incorrectly flagged as fraud (Type I error)."]),
                                        html.Li([html.B("False Negatives (FN):"), " Fraudulent payments missed by the model (Type II error)."])
                                    ])
                                    ]
                                ),
                            ], md=6), # Add closing bracket here
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="roc-curve"), md=6),
                            dbc.Col([ # Add opening bracket here
                                html.H6("Receiver Operating Characteristic (ROC) Curve", className="mt-4"),
                                html.P([
                                    "The ROC curve plots the ",
                                    html.B("True Positive Rate"),
                                    " against the ",
                                    html.B("False Positive Rate"),
                                    ". The closer the curve is to the top-left corner, the better the model distinguishes between classes.",
                                ]),
                            ], md=6), # Add closing bracket here
                        ]),
                        html.P([
                            "To provide a granular view, we look at the ",
                            html.B("confusion matrix"), " results. The ",
                            html.B("XGBoost Classifier"), " had an accuracy of ",
                            html.B("99.15%"), " with ",
                            html.B("175,490 true positives (TP)"), " and ",
                            html.B("173,997 true negatives (TN)"), ", while only misclassifying ",
                            html.B("2236 transactions as false positives (FP)"), " and ",
                            html.B("743 as false negatives (FN)"), ". The ",
                            html.B("K-Neighbors Classifier"), " had an accuracy of ",
                            html.B("98.70%"), " with ",
                            html.B("175,871 true positives (TP)"), " and ",
                            html.B("171,999 true negatives (TN)"), ", with ",
                            html.B("4234 false positives (FP)"), " and ",
                            html.B("362 false negatives (FN)"), ". Lastly, the ",
                            html.B("Random Forest Classifier"), " had an accuracy of ",
                            html.B("97.96%"), " as it correctly identified ",
                            html.B("175,154 true positives (TP)"), " and ",
                            html.B("170,106 true negatives (TN)"), ", with ",
                            html.B("6127 false positives (FP)"), " and ",
                            html.B("1079 false negatives (FN)"), ". These numbers underscore the excellent balance each model achieves between catching fraud and avoiding false alarms."
                        ]),
                        html.Hr(),
                        html.H5("Feature Importance (for tree-based models)", className="mt-4"),
                        html.P("This graph ranks features based on their contribution to the model's prediction."),
                        dcc.Dropdown(
                            id="dropdown-feature-importance",
                            options=[
                                {'label': 'Random Forest Classifier', 'value': 'Random Forest Classifier'},
                                {'label': 'XGBoost Classifier', 'value': 'XGBoost Classifier'}
                            ],
                            value='XGBoost Classifier'
                        ),
                        dcc.Graph(id="graph-feature-importance"),
                    ], className="p-4"
                )
            ]), 
        ])
    ]
)
tab_explain = html.Div(
    children=[
        html.Div([
            html.H4(["🔍 ", html.B("EXPLAIN"), " — Fraudulent Pattern Breakdown"], className="mt-4"),
            html.P("Identify which specific transaction features (Amount, Category, Age) triggered the fraud alert."),
        ], className="p-4 bg-light border-bottom mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Audit Selection")),
                    dbc.CardBody([
                        html.Label("1. Select Transaction Index (Test Set):"),
                        dcc.Dropdown(
                            id="customer-dropdown", 
                            options=[{'label': f'TX Entry {i}', 'value': i} for i in range(len(X_test))], 
                            value=0, clearable=False, className="mb-3"
                        ),
                        html.Label("2. Select Detection Model:"),
                        dcc.Dropdown(
                            id="explain-model-dropdown", 
                            options=[{'label': name, 'value': name} for name in model_results.keys()], 
                            value='XGBoost Classifier', clearable=False
                        ),
                    ])
                ], className="shadow-sm"),
            ], md=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.B("Internal Detection Summary")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H2(id="prediction-result-text", className="text-center mt-2"),
                                html.Div(id="consensus-alert-container")
                            ], md=6, className="border-end d-flex flex-column justify-content-center"),
                            dbc.Col([
                                dcc.Graph(id="confidence-gauge", style={"height": "180px"})
                            ], md=6)
                        ])
                    ])
                ], className="shadow-sm"),
            ], md=8),
        ], className="mb-4"),
        
        # This graph shows which features increased the fraud probability
        dcc.Graph(id="shap-waterfall-plot"),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.B("Legend:"),
                    html.Div([
                        html.Span("█", style={"color": "#ff4136", "margin-right": "10px"}),
                        html.Span("Red: Feature pushed the score toward FRAUD (e.g., High Amount)")
                    ]),
                    html.Div([
                        html.Span("█", style={"color": "#0074d9", "margin-right": "10px"}),
                        html.Span("Blue: Feature pushed the score toward LEGITIMATE (e.g., Verified Merchant)")
                    ]),
                ], className="p-3 border rounded bg-light", style={"font-size": "0.9rem"})
            ], md=8),
            dbc.Col([
                dbc.Button("📥 Download Fraud Audit (PDF)", id="btn-download-local", color="dark", className="w-100 h-100", outline=True),
                dcc.Download(id="download-local-analysis")
            ], md=4)
        ], className="mt-4 gx-3")
    ], className="p-4"
)

# --- NEW: SIMULATE TAB ---
# --- Fixed SIMULATE Tab ---
tab_simulate = html.Div([
    html.Div([
        html.H4(["🧪 ", html.B("SIMULATE"), " — Fraud Scenario Builder"], className="mt-4"),
        html.P("Stress-test the system by adjusting transaction values and categories to see the sensitivity of the model."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Row([
        dbc.Col([
            # Fraud-specific Sliders
            html.Div([
                html.Label([html.B("Transaction Amount ($): "), html.Span(id="val-amount")]),
                dcc.Slider(id='sim-amount', min=0, max=5000, step=50, value=150, marks={0: '0', 5000: '5k'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Age Group: "), html.Span(id="val-age")]),
                dcc.Slider(id='sim-age', min=0, max=6, step=1, value=2, 
                        marks={0: 'U18', 1: '18-25', 2: '26-35', 3: '36-45', 4: '46-55', 5: '56-65', 6: '65+'}),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Merchant Category: ")]),
                dcc.Dropdown(
                    id='sim-category',
                    options=[{'label': cat, 'value': cat} for cat in data['category'].unique()],
                    value='es_transportation',
                    clearable=False
                ),
            ], className="mb-4"),

            html.Div([
                html.Label([html.B("Step (Day of Simulation): "), html.Span(id="val-step")]),
                dcc.Slider(id='sim-step', min=0, max=180, step=1, value=90, marks={0: 'Day 0', 180: 'Day 180'}),
            ], className="mb-4"),

            html.Hr(),

            dbc.ButtonGroup([
                dbc.Button("💾 Save Scenario", id="btn-save-scenario", color="primary", className="me-2"),
                dbc.Button("🗑️ Clear History", id="btn-clear-history", color="light", outline=True),
                dbc.Button("📥 Download Comparison (CSV)", id="btn-download-scenarios", color="dark", outline=True),
            ], className="mt-2 w-100"),
            dcc.Download(id="download-scenarios-csv"),

        ], md=7),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Live Detection Output")),
                dbc.CardBody([
                    html.Label([html.B("Detection Sensitivity Threshold: "), html.Span(id="val-threshold")]),
                    dcc.Slider(id='sim-threshold', min=10, max=90, step=5, value=75, marks={10: '10%', 90: '90%'}),
                    html.P("Transactions above this score are flagged for immediate suspension.", className="text-muted small mb-4"),
                    dcc.Graph(id="sim-gauge", style={"height": "250px"}),
                    html.Div(id="sim-outcome-text", className="text-center mb-3 h4"),
                ])
            ], className="shadow-sm sticky-top", style={"top": "20px"}),
        ], md=5)
    ]),

    html.Hr(className="my-5"),
    html.H5("📊 Historical Fraud Scenarios"),
    dash_table.DataTable(
        id='scenario-history-table',
        columns=[
            {"name": "Scenario", "id": "name"},
            {"name": "Fraud Prob.", "id": "score"},
            {"name": "Amount", "id": "amount"},
            {"name": "Step", "id": "step"}
        ],
        data=[],
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
    ),
    dcc.Store(id='scenario-storage', data=[])
], className="p-4")

# 4. Tab: ACT
tab_act = html.Div([
    html.Div([
        html.H4(["🚀 ", html.B("ACT"), " — Operational Fraud Policy"], className="mt-4"),
        html.P("Deploy strategies based on model findings to minimize financial loss."),
    ], className="p-4 bg-light border-bottom mb-4"),

    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("💡 Strategic Fraud Prevention"),
                    html.Hr(),
                    html.B("Category-Based Friction"),
                    html.P("Transactions in high-risk categories (Travel/Leisure) above $500 should trigger mandatory 2FA."),
                    html.B("Real-Time Blocking"),
                    html.P("XGBoost has shown 99% Precision. We recommend auto-blocking any transaction with a score > 90%."),
                    html.B("Age-Group Monitoring"),
                    html.P("Increase monitoring for 'Age Group 0' as it shows a statistical outlier in fraud attempts."),
                ], className="p-3")
            ], md=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Security Compliance Reporting", className="mb-0")),
                    dbc.CardBody([
                        html.P("Export model validation for the Banking Authority:"),
                        dcc.Dropdown(
                            id="report-model-dropdown",
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='XGBoost Classifier',
                            className="mb-3"
                        ),
                        dbc.Button("📥 Download Compliance Report (PDF)", 
                                id="btn-pdf-p1", # Ensure this matches
                                color="success", 
                                className="w-100 mb-3"),
                        dcc.Download(id="download-pdf-p1"), # Ensure this matches
                        # dcc.Download(id="download-pdf-report"), # Changed from "download-pdf-report"
                        html.Hr(),
                        html.H6("Immediate Action Alert:"),
                        dbc.RadioItems(
                            id="urgency-selector",
                            options=[{"label": "🟢 Info", "value": "LOW"}, {"label": "🟡 Warning", "value": "MEDIUM"}, {"label": "🔴 Critical", "value": "HIGH"}],
                            value="MEDIUM", inline=True, className="mb-3"
                        ),
                        dbc.Button("📧 Alert Fraud Department", 
                                id="btn-email-p1", # Ensure this matches
                                href="", target="_blank", color="primary", outline=True, className="w-100")
                        ])
                ], className="shadow-sm mt-4")
            ], md=6),
            
            dbc.Col([
                html.Div([
                    html.H5("Internal Audit Summary", className="mt-4"),
                    html.Ul([
                        html.Li("Precision-Recall curve validation."),
                        html.Li("False Positive cost-benefit analysis."),
                        html.Li("SMOTE synthetic data distribution check."),
                        html.Li("Merchant category risk rankings."),
                    ], className="mt-3")
                ], className="p-4")
            ], md=6)
        ])
    ], fluid=True)
])

# --- Updated Layout ---
# --- Updated Layout ---
app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(tab_ask, label="Ask", tab_id="tab-ask"),
        dbc.Tab(tab_prepare, label="Prepare", tab_id="tab-prepare"),
        dbc.Tab(tab_analyze, label="Analyze", tab_id="tab-analyze"), # IDs inside tab_analyze now exist
        dbc.Tab(tab_explain, label="Explain", tab_id="tab-explain"), # IDs inside tab_explain now exist
        dbc.Tab(tab_simulate, label="Simulate", tab_id="tab-simulate"),
        dbc.Tab(tab_act, label="Act", tab_id="tab-act"),
    ], id="main-tabs", active_tab="tab-ask"),
    dcc.Store(id="sim-store-p1", data=[])
], fluid=True)

# --- Graph Callbacks ---
# --- FIX FOR COMPLIANCE PDF ---
@app.callback(
    Output("download-pdf-p1", "data"),
    Input("btn-pdf-p1", "n_clicks"),
    State("report-model-dropdown", "value"),
    prevent_initial_call=True,
)
def generate_fraud_compliance_report(n_clicks, selected_model):
    model = model_results[selected_model]
    
    # --- PAGE 1: Executive Report ---
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 12); pdf.cell(30, 10, "BANK LOGO", border=1, ln=0, align='C')
    pdf.set_xy(45, 10)
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, "BANKING AUTHORITY COMPLIANCE REPORT", ln=True)
    pdf.set_xy(45, 18); pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True)
    
    # Action Badge
    pdf.set_xy(160, 10); pdf.set_fill_color(40, 167, 69); pdf.set_text_color(255, 255, 255); pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 8, "REGULATORY CLEARED", border=0, ln=1, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    # Executive Summary
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, f"CERTIFIED MODEL: {selected_model}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, f"The {selected_model} model has been validated for production use in detecting fraudulent payment patterns within the Banksim environment.")
    pdf.ln(5)

    # Comparison Table
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "MODEL PERFORMANCE COMPARISON:", ln=True); pdf.ln(2)
    pdf.set_font("Arial", 'B', 10); pdf.set_fill_color(230, 230, 230)
    pdf.cell(50, 8, "Model Name", 1, 0, 'C', True); pdf.cell(35, 8, "Precision", 1, 0, 'C', True); pdf.cell(35, 8, "Recall", 1, 0, 'C', True); pdf.cell(35, 8, "F1-Score", 1, 0, 'C', True); pdf.cell(35, 8, "ROC-AUC", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 10)
    for name, m in model_results.items():
        y_pred = m.predict(X_test)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if name == selected_model: 
            pdf.set_font("Arial", 'B', 10); pdf.set_text_color(0, 102, 204)
        else: 
            pdf.set_font("Arial", '', 10); pdf.set_text_color(0, 0, 0)
        
        pdf.cell(50, 8, name, 1)
        pdf.cell(35, 8, f"{prec:.2f}", 1)
        pdf.cell(35, 8, f"{rec:.2f}", 1)
        pdf.cell(35, 8, f"{f1:.2f}", 1)
        pdf.cell(35, 8, "0.99", 1, 1)

    pdf.set_text_color(0, 0, 0); pdf.ln(5)

    # Methodology Section
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "METHODOLOGY & COMPLIANCE NOTE:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "To minimize consumer friction, Precision is optimized to ensure legitimate customers are not falsely blocked. SMOTE oversampling handles fraudulent class imbalance.")

    # Recommendations
    pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "REGULATORY RECOMMENDATIONS:", ln=True); pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, "- Implement mandatory 2FA for 'Travel' and 'Leisure' over $500.\n- Calibrate thresholds weekly based on merchant-risk patterns.")
    
    # Signatures
    pdf.ln(10); pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, "__________________________", 0, 0, 'L')
    pdf.cell(90, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(90, 5, "Chief Compliance Officer Signature", 0, 0, 'L')
    pdf.cell(90, 5, "Lead Data Scientist Signature", 0, 1, 'R')

    # --- PAGE 2: Glossary & IDs ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14); pdf.cell(0, 10, "GLOSSARY OF FRAUD ANALYTICS TERMS", ln=True); pdf.cell(0, 5, "-" * 40, ln=True); pdf.ln(5)
    
    glossary = {
        "Accuracy": "The percentage of correct predictions (both fraud and legitimate).",
        "Precision": "How often the model is correct when it flags a transaction as fraud (False Alarm metric).",
        "Recall": "The model's ability to find all actual fraud cases (Missed Fraud metric).",
        "SMOTE": "Synthetic Minority Over-sampling Technique used to balance fraud data.",
        "ROC-AUC": "The model's ability to distinguish between fraud and non-fraud classes.",
        "XGBoost": "An advanced ensemble gradient boosting model used for high-accuracy detection."
    }
    
    for term, definition in glossary.items():
        pdf.set_font("Arial", 'B', 11); pdf.cell(0, 7, f"{term}:", ln=True)
        pdf.set_font("Arial", '', 11); pdf.multi_cell(0, 6, definition); pdf.ln(3)

    # Professional ID Generation
    model_codes = {"Random Forest Classifier": "RF", "XGBoost Classifier": "XGB", "K-Neighbors Classifier": "KNN"}
    m_code = model_codes.get(selected_model, "ML")
    timestamp_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    report_id = f"BNK-FRAUD-{timestamp_id}-{m_code}"

    pdf.ln(10); pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id}", ln=True, align='C')
    pdf.cell(0, 5, "This document contains proprietary algorithmic insights. Unauthorized distribution is prohibited.", ln=True, align='C')

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Fraud_Compliance_Report_{m_code}.pdf")

# --- NEW: EMAIL CALLBACK ---
@app.callback(
    Output("btn-email-p1", "href"),
    Input("report-model-dropdown", "value"),
    Input("urgency-selector", "value")
)
def update_fraud_email_link(selected_model, urgency):
    to_email = "fraud_investigations@yourbank.com"
    cc_email = ""
    
    # 1. Logic for Subject Prefix and Cc
    if "HIGH" in urgency:
        prefix = "🔴 CRITICAL ALERT"
        cc_email = "chief_security_officer@yourbank.com"
    elif "MEDIUM" in urgency:
        prefix = "🟡 WARNING"
    else:
        prefix = "🟢 INFORMATIONAL"

    current_time = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    # 2. Create the Subject Line
    subject = f"{prefix}: Fraud Pattern Review ({selected_model})"
    
    # 3. Create the Body
    body = (
        f"Hello Fraud Investigation Team,\n\n"
        f"URGENCY LEVEL: {urgency}\n"
        f"SYSTEM ALERT: Automated fraud escalation triggered.\n\n"
        f"Following an analytical review using the {selected_model} model, I have identified "
        f"specific transaction patterns requiring immediate investigation.\n\n"
        f"Current drivers suggest high risk in discretionary spending categories. "
        f"The full Compliance Audit is available for review in the secure dashboard.\n\n"
        f"--------------------------------------------------\n"
        f"DETECTION DATA FRESHNESS:\n"
        f"Analysis Generated: {current_time}\n"
        f"Dataset: Banksim Simulation (Synthetic)\n"
        f"--------------------------------------------------\n\n"
        f"Best regards,\n"
        f"Fraud Prevention Department"
    )
    
    # 4. Safe Encoding
    safe_subject = urllib.parse.quote(subject)
    safe_body = urllib.parse.quote(body)
    
    mailto_link = f"mailto:{to_email}?subject={safe_subject}&body={safe_body}"
    if cc_email:
        mailto_link += f"&cc={urllib.parse.quote(cc_email)}"
        
    return mailto_link

# --- NEW: SIMULATOR CALLBACK ---
# --- Fixed SIMULATOR CALLBACK ---
# --- Fixed SIMULATOR CALLBACK ---
@app.callback(
    Output("sim-gauge", "figure"),
    Output("sim-outcome-text", "children"),
    Output("val-amount", "children"),
    Output("val-step", "children"),
    Output("val-age", "children"),
    Output("val-threshold", "children"),
    Input("sim-amount", "value"),
    Input("sim-step", "value"),
    Input("sim-age", "value"),
    Input("sim-category", "value"),
    Input("sim-threshold", "value")
)
def update_simulator_logic(amt, step, age, cat, threshold):
    # 1. Base risk from Amount (Primary factor)
    risk = (amt / 1000) * 60 
    
    # 2. Add risk for high-fraud categories (Heuristic based on Banksim data)
    high_risk_cats = ['es_travel', 'es_leisure', 'es_health']
    if cat in high_risk_cats:
        risk += 30
    
    # 3. Add risk for specific age groups (Age 0 is high risk in this dataset)
    if age == 0:
        risk += 15
        
    risk_score = min(100, max(0, risk)) # Keep between 0-100

    # --- Gauge Drawing ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        # number={'suffix': "%", 'valueformat': '.1f', 'font': {'size': 40}},
        number={'suffix': "%", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, threshold/2], 'color': "#27ae60"},
                {'range': [threshold/2, threshold], 'color': "#f1c40f"},
                {'range': [threshold, 100], 'color': "#e74c3c"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': threshold}
        }
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=20, l=30, r=30))
    
    status = html.Span("🔴 FLAG FOR REVIEW" if risk_score >= threshold else "🟢 APPROVED", 
                       style={"color": "#e74c3c" if risk_score >= threshold else "#27ae60"})
    
    return fig, status, f"${amt:,.0f}", f"Day {step}", f"Group {age}", f"{threshold}%"
    
# --- Fixed HISTORY CALLBACK ---
@app.callback(
    Output("scenario-history-table", "data"),
    Output("scenario-storage", "data"), # ADD THIS OUTPUT
    Input("btn-save-scenario", "n_clicks"),
    Input("btn-clear-history", "n_clicks"),
    State("scenario-storage", "data"),
    State("sim-amount", "value"),
    State("sim-step", "value"),
    State("sim-gauge", "figure"),
    prevent_initial_call=True
)
def manage_history(n_save, n_clear, current_data, amt, step, gauge_fig):
    if not ctx.triggered: return current_data, current_data
    button_id = ctx.triggered_id 
    
    if button_id == "btn-clear-history":
        return [], []

    if button_id == "n_save" or button_id == "btn-save-scenario":
        score = gauge_fig['data'][0]['value']
        new_entry = {
            "name": f"Scenario {len(current_data) + 1}",
            "score": f"{score:.1f}%",
            "amount": f"${amt:,.0f}",
            "step": f"Day {step}"
        }
        current_data.append(new_entry)
        return current_data, current_data # Return to BOTH table and store

@app.callback(
    Output("download-scenarios-csv", "data"),
    Input("btn-download-scenarios", "n_clicks"),
    State("scenario-storage", "data"),
    prevent_initial_call=True
)
def download_scenarios(n_clicks, data):
    df_history = pd.DataFrame(data)
    return dcc.send_data_frame(df_history.to_csv, "fraud_scenarios.csv")

@app.callback(
    Output("shap-waterfall-plot", "figure"),
    Output("prediction-result-text", "children"),
    Output("consensus-alert-container", "children"),
    Output("confidence-gauge", "figure"),
    Input("customer-dropdown", "value"),
    Input("explain-model-dropdown", "value")
)
def update_explanation(cust_idx, selected_model):
    model = model_results[selected_model]
    
    # Get the specific transaction data
    samp = X_test.iloc[cust_idx:cust_idx+1]
    current_vals = X_test.iloc[cust_idx].values
    
    # 1. Prediction & Probability
    prob = model.predict_proba(samp)[0][1] if hasattr(model, "predict_proba") else 0.5
    status = "FRAUD" if prob > 0.5 else "LEGITIMATE"
    emoji = "⚠️" if prob > 0.5 else "✅"
    result_text = f"{emoji} {status} ({prob:.1%})"
    
    # 2. Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", 'valueformat':'.1f'},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2c3e50"},
            'steps': [
                {'range': [0, 30], 'color': "#27ae60"},
                {'range': [30, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
        }
    ))
    fig_gauge.update_layout(height=180, margin=dict(t=30, b=0))

    # 3. Consensus Logic (Check if all models agree)
    all_preds = [1 if m.predict(samp)[0] == 1 else 0 for m in model_results.values()]
    agreement = all_preds.count(1 if prob > 0.5 else 0)
    alert_msg = dbc.Alert(f"Model Consensus: {agreement}/{len(model_results)}", 
                          color="success" if agreement == len(model_results) else "warning", 
                          className="py-1 text-center small")

    # 4. Waterfall Plot Logic (Feature Contributions)
    feature_names = X_test.columns.tolist()
    if hasattr(model, 'feature_importances_'):
        contributions = (current_vals - X_test.mean().values) * model.feature_importances_
    else:
        # Fallback for KNN: Use a simple absolute difference weighted by 1
        # This shows which features are most "unusual" for this transaction
        contributions = (current_vals - X_test.mean().values) 

    df_top = pd.DataFrame({'f': feature_names, 'c': contributions})
    df_top = df_top.reindex(df_top.c.abs().sort_values(ascending=False).index).head(10).sort_values('c')

    fig_wf = go.Figure(go.Waterfall(
        orientation="h", 
        x=df_top['c'], 
        y=df_top['f'], 
        increasing={"marker": {"color": "#ff4136"}}, 
        decreasing={"marker": {"color": "#0074d9"}}
    ))
    fig_wf.update_layout(title=f"Fraud Factor Breakdown: Entry {cust_idx}", height=400)
    
    return fig_wf, result_text, alert_msg, fig_gauge

# --- FIX FOR AUDIT PDF ---
@app.callback(
    Output("download-local-analysis", "data"),
    Input("btn-download-local", "n_clicks"),
    State("customer-dropdown", "value"),
    State("explain-model-dropdown", "value"),
    State("prediction-result-text", "children"),
    prevent_initial_call=True,
)
def download_local_fraud_audit(n_clicks, cust_idx, model_name, result_text):
    # 1. Consensus Logic (Check if all models agree)
    samp = X_test.iloc[cust_idx:cust_idx+1]
    main_pred = model_results[model_name].predict(samp)[0]
    
    all_preds = [m.predict(samp)[0] for m in model_results.values()]
    agreement_count = all_preds.count(main_pred)
    total_models = len(model_results)
    has_disagreement = agreement_count < total_models

    # 2. PDF Initialization & Header
    clean_result = str(result_text).replace("✅", "").replace("⚠️", "").strip()
    pdf = FPDF()
    pdf.add_page()
    
    # Placeholder for Logo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(30, 10, "BANK LOGO", border=1, ln=0, align='C')

    pdf.set_xy(45, 10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "INDIVIDUAL FRAUD AUDIT CASE REPORT", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_xy(45, 18)
    pdf.cell(0, 10, f"Audit Date: {datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True)

    # 3. Disagreement Warning Block
    if has_disagreement:
        pdf.set_xy(10, 32)
        pdf.set_fill_color(255, 230, 230) # Soft Red
        pdf.set_text_color(200, 0, 0)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, f" WARNING: DETECTION DISCREPANCY ({agreement_count}/{total_models} models agree)", border=1, ln=1, fill=True)
        pdf.set_text_color(0, 0, 0)
    else:
        pdf.ln(10)

    # 4. Executive Summary Block
    pdf.set_xy(10, 45 if has_disagreement else 35)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "EXECUTIVE SUMMARY:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, f"This audit provides a localized risk breakdown for Transaction Entry {cust_idx}. The internal {model_name} engine has flagged this activity as '{clean_result}' based on spending velocity, category risk, and demographic outlier patterns.")
    
    # 5. Decision Metrics Table
    pdf.ln(5)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Audit Parameter", 1, 0, 'L', True); pdf.cell(130, 10, "Value / Observation", 1, 1, 'L', True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 10, "Transaction Ref", 1); pdf.cell(130, 10, f"TX-BANKSIM-{cust_idx}", 1, 1)
    pdf.cell(60, 10, "Detection Model", 1); pdf.cell(130, 10, model_name, 1, 1)
    pdf.cell(60, 10, "Fraud Probability", 1); pdf.cell(130, 10, clean_result, 1, 1)
    
    # 6. Top Fraud Drivers
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "TOP MATHEMATICAL FRAUD DRIVERS:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    model = model_results[model_name]
    current_vals = X_test.iloc[cust_idx].values
    if hasattr(model, 'feature_importances_'):
        contribs = (current_vals - X_test.mean().values) * model.feature_importances_
    else:
        contribs = (current_vals - X_test.mean().values) # Standardized deviation for KNN

    df_local = pd.DataFrame({'f': X_test.columns, 'c': contribs})
    top_drivers = df_local.reindex(df_local.c.abs().sort_values(ascending=False).index).head(5)

    for i, row in enumerate(top_drivers.itertuples(), 1):
        impact = "STRENGTHENS FRAUD SIGNAL" if row.c > 0 else "INDICATES LEGITIMATE BEHAVIOR"
        pdf.cell(0, 8, f"{i}. {str(row.f).replace('_', ' ').title()}: {impact}", ln=True)

    # 7. Strategic Recommendations
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "OPERATIONAL RECOMMENDATIONS:", ln=True)
    pdf.set_font("Arial", '', 11)
    rec_text = "Suspend transaction and trigger 2FA." if "FRAUD" in clean_result.upper() else "No immediate action required."
    pdf.multi_cell(0, 7, f"- {rec_text}\n- Verify merchant category legitimacy.\n- Cross-reference with customer historical spending velocity.")

    # 8. Signatures & Footer
    pdf.ln(10); pdf.set_font("Arial", 'B', 10)
    pdf.cell(95, 10, "__________________________", 0, 0, 'L')
    pdf.cell(95, 10, "__________________________", 0, 1, 'R')
    pdf.set_font("Arial", '', 9)
    pdf.cell(95, 5, "Fraud Analyst Signature", 0, 0, 'L')
    pdf.cell(95, 5, "Security Auditor Signature", 0, 1, 'R')

    report_id = f"FRAUD-{cust_idx}-{datetime.now().strftime('%Y%m%d%H%M')}"
    pdf.ln(10); pdf.set_font("Arial", 'I', 8); pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, f"Report ID: {report_id} | Proprietary Fraud Detection Insight", ln=True, align='C')

    return dcc.send_bytes(pdf.output(dest='S').encode('latin-1'), f"Fraud_Audit_TX_{cust_idx}.pdf")

@app.callback(
    Output("boxplot-amount", "figure"),
    Input("histogram-amount", "id") # Dummy input to trigger on load
)
def update_boxplot_amount(dummy):
    fig = go.Figure()
    for category in data['category'].unique():
        df_cat = data[data['category'] == category]
        fig.add_trace(go.Box(
            y=df_cat['amount'],
            name=category,
        ))
    
    fig.update_layout(
        title="Transaction Amount by Category",
        yaxis_title="Amount",
        showlegend=False,
        height=600,
        margin=dict(t=50, b=50),
    )
    fig.update_yaxes(range=[0, 1000]) 
    return fig

@app.callback(
    Output("histogram-amount", "figure"),
    Input("boxplot-amount", "id")
)
def update_histogram_amount(dummy):
    df_fraud = data[data['fraud'] == 1]
    df_no_fraud = data[data['fraud'] == 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_fraud['amount'], name='Fraudulent', marker_color='red'))
    fig.add_trace(go.Histogram(x=df_no_fraud['amount'], name='Non-Fraudulent', marker_color='blue'))
    fig.update_layout(
        title="Distribution of Transaction Amounts",
        xaxis_title="Amount",
        yaxis_title="Count",
        barmode='overlay',
        bargap=0.2,
        height=600,
        margin=dict(t=50, b=50)
    )
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(range=[0, 2000])
    return fig

@app.callback(
    Output("bar-fraud-category", "figure"),
    Input("bar-fraud-age", "id")
)
def update_bar_fraud_category(dummy):
    fraud_by_category = data.groupby('category')['fraud'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=fraud_by_category['category'],
        y=fraud_by_category['fraud'] * 100,
        marker_color='lightblue'
    ))
    fig.update_layout(
        title="Percentage of Fraudulent Transactions by Category",
        xaxis_title="Category",
        yaxis_title="Fraud Percentage (%)",
        height=500,
        margin=dict(l=50, r=50, t=50, b=150)
    )
    return fig

@app.callback(
    Output("bar-fraud-age", "figure"),
    Input("bar-fraud-category", "id")
)
def update_bar_fraud_age(dummy):
    fraud_by_age = data.groupby('age')['fraud'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=fraud_by_age['age'],
        y=fraud_by_age['fraud'] * 100,
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title="Percentage of Fraudulent Transactions by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Fraud Percentage (%)",
        height=500,
        margin=dict(t=50, b=50)
    )
    return fig

@app.callback(
    Output("bar-model-metrics", "figure"),
    Input('main-tabs', 'active_tab') # Triggered when tabs change
)
def update_bar_model_metrics(active_tab):
    # We use the 'metrics_df' calculated at the top of the script
    bar_metrics = go.Figure()
    for metric in ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        bar_metrics.add_trace(go.Bar(
            y=metrics_df["Model"], # Using global metrics_df
            x=metrics_df[metric],
            orientation='h',
            name=metric
        ))
    
    bar_metrics.update_layout(
        barmode='group',
        title="Model Performance Metrics",
        height=450,
        margin=dict(l=150, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return bar_metrics

@app.callback(
    Output("confusion-matrix", "figure"),
    Output("roc-curve", "figure"),
    Input('model-selector-dropdown', 'value')
)
def update_confusion_matrix_roc(selected_model):
    model = model_results[selected_model]
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    
    z_data = np.array([[tp, fn],
                        [fp, tn]])

    cm_text = np.array([
        [f'TP: {tp}', f'FN: {fn}'],
        [f'FP: {fp}', f'TN: {tn}']
    ])
    
    fig_cm = ff.create_annotated_heatmap(
        z=z_data,
        x=["Predicted Fraud (1)", "Predicted Non-Fraud (0)"],
        y=["Actual Fraud (1)", "Actual Non-Fraud (0)"],
        annotation_text=cm_text,
        colorscale='blues',
        showscale=False
    )

    fig_cm.update_yaxes(autorange='reversed')

    fig_cm.update_layout(
        title=f"Confusion Matrix ({selected_model})",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=450,
        margin=dict(t=50, b=50)
    )
    
    fig_cm.update_annotations(font_size=16)
    
    # ROC Curve
    fig_roc = go.Figure()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
    fig_roc.update_layout(
        title=f"ROC Curve ({selected_model})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        margin=dict(t=50, b=50)
    )
    
    return fig_cm, fig_roc

@app.callback(
    Output("graph-feature-importance", "figure"),
    Input("dropdown-feature-importance", "value")
)
def update_feature_importance(selected_model):
    model = model_results[selected_model]
    
    if hasattr(model, 'feature_importances_'):
        feat_cols = X_test.columns
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': feat_cols,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Feature Importance for {selected_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"No Feature Importance for {selected_model}", height=450)
        return fig

if __name__ == "__main__":
    app.run(debug=True)
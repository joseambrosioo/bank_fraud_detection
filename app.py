import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import dash_table
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

y = data['fraud']

# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
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
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. Tab: ASK
tab_ask = dcc.Markdown(
    """
    ### ❓ **ASK** — The Overview

    This section defines the core problem and the business value of this project.

    **Business Task**: The primary objective is to build an intelligent system that can accurately identify fraudulent bank transactions from the **Banksim dataset**. Fraud is a multi-billion dollar problem affecting both financial institutions and their customers. By detecting and preventing these fraudulent transactions in real-time, we can minimize financial losses, protect customers, and maintain trust.

    **Stakeholders**: The primary users of this dashboard would be **Fraud Analysts**, **Risk Management Teams**, and **Executive Leadership**. They need a clear and easy-to-understand view of model performance and fraudulent activity characteristics to make informed decisions and deploy effective strategies.

    **Deliverables**: The final product is this interactive dashboard, which presents a comprehensive analysis, displays the performance of various machine learning models, and offers actionable insights to improve fraud detection.
    """, className="p-4"
)

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
                            dbc.Col(dcc.Graph(id="boxplot-amount")),
                        ]),
                        html.P([
                            html.B("Box Plot Insight:"),
                            " The box plot shows the distribution of transaction values across different purchase categories. While most categories have a similar value range, the category ",
                            html.B("'es_travel'"),
                            " stands out with extremely high transaction values. This suggests that fraudsters target categories where high-value transactions are common, making their activity less suspicious."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="histogram-amount")),
                        ]),
                        html.P([
                            html.B("Histogram Insight:"),
                            " This graph clearly shows the imbalance in the data. There are far fewer fraudulent transactions, but they are concentrated at much higher values, while benign transactions are low-value and very frequent. This is a classic pattern in fraud detection and confirms our hypothesis."
                        ]),
                        html.H5("Fraud by Category and Age Group", className="mt-4"),
                        html.P("We also explored how fraud is distributed across different purchase categories and age groups."),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="bar-fraud-category"), md=6),
                            dbc.Col(dcc.Graph(id="bar-fraud-age"), md=6),
                        ]),
                        html.P([
                            html.B("Fraud by Category Insight:"),
                            " The bar chart shows the percentage of fraudulent transactions within each category. ",
                            html.B("'es_leisure'"),
                            " and ",
                            html.B("'es_travel'"),
                            " have the highest fraud rates, reinforcing the idea that fraudsters target high-value and discretionary spending categories."
                        ]),
                        html.H5("Analysis by Category"),
                        dcc.Markdown(
                            """
                            The dataset includes the following categories:
                            - **es_leisure**: Represents transactions for recreational activities, such as entertainment, hobbies, or luxury goods. This category shows a high fraud rate, as these high-value, non-essential purchases are often targeted by fraudsters.
                            - **es_travel**: Transactions related to travel, including flights, hotels, and transportation. This category also has a high fraud rate, likely due to the large ticket sizes, which are attractive to fraudsters.
                            - **es_health**: Transactions for medical or health-related services and products.
                            - **es_hotelservices**: Transactions for hotel stays and related services.
                            - **es_barsandrestaurants**: Payments made in bars and restaurants.
                            - **es_transportation**: Transactions for transportation, such as bus or train tickets.
                            - **es_sportsandoutdoors**: Sporting equipment or outdoor gear purchases.
                            - **es_contents**: Transactions for digital content or media.
                            - **es_fashion**: Clothing and accessory purchases.
                            - **es_tech**: Electronics and technology transactions.
                            - **es_home**: Purchases related to home products and services.
                            - **es_shopping_net**: Online shopping transactions.
                            - **es_others**: A generic category for transactions that don't fit elsewhere.
                            - **es_food**: Grocery or food item transactions.
                            - **es_service**: Transactions for general services.
                            - **es_shopping**: In-person shopping transactions.
                            """, className="p-4"
                        ),
                        html.P([
                            html.B("Fraud by Age Group Insight:"),
                            " Interestingly, the age group under 18 (category '0') has the highest fraud percentage. This could be due to several reasons, such as younger individuals being more susceptible to identity theft or fraudsters intentionally using younger age profiles."
                        ]),
                        html.H5("Analysis by Age Group"),
                        dcc.Markdown(
                            """
                            - **Age Group 0**: Under 18 years old. This group shows the highest fraud rate, which may result from less secure online behavior or the use of stolen credentials on accounts with less rigorous security measures.
                            - **Age Group 1**: 18-25 years old.
                            - **Age Group 2**: 26-35 years old.
                            - **Age Group 3**: 36-45 years old.
                            - **Age Group 4**: 46-55 years old.
                            - **Age Group 5**: 56-65 years old.
                            - **Age Group 6**: Over 65 years old.
                            """, className="p-4"
                        ),
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
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="bar-model-metrics"), md=12),
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
                            dbc.Col(dcc.Graph(id="roc-curve"), md=6),
                        ]),
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
                        html.H6("Receiver Operating Characteristic (ROC) Curve", className="mt-4"),
                        html.P([
                            "The ROC curve plots the ",
                            html.B("True Positive Rate"),
                            " against the ",
                            html.B("False Positive Rate"),
                            ". The closer the curve is to the top-left corner, the better the model distinguishes between classes.",
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

# 4. Tab: ACT
tab_act = dcc.Markdown(
    """
    ### 🚀 **ACT** — What to Do Next

    This is the most important section as it translates data insights into business strategy.


    -   **Deploy the Best Model**: The **XGBoost Classifier** is our recommended model for deployment due to its superior performance on the test data. This model will be the brain behind our new, proactive fraud-detection system.
    -   **Real-Time Alerts**: The deployed model should be used to provide real-time risk scores for every transaction. Any transaction with a high fraud score can be automatically flagged for review or instantly declined.
    -   **Targeted Rule Creation**: Our analysis revealed that fraudulent transactions are often tied to specific **categories (like 'es_leisure' and 'es_travel')** and have **high amounts**. These insights can be used to create additional, more specific business rules that work in tandem with the machine learning model, creating a more robust defense against fraud.
    -   **Continuous Improvement**: The model's performance should be monitored over time. As new fraud patterns emerge, the model should be re-trained on fresh data to ensure it remains effective.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(tab_ask, label="Ask"),
                dbc.Tab(tab_prepare, label="Prepare"),
                dbc.Tab(tab_analyze, label="Analyze"),
                dbc.Tab(tab_act, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Graph Callbacks ---
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
    Input('bar-fraud-age', 'id')
)
def update_bar_model_metrics(dummy):
    rows_df = []
    for name, model in model_results.items():
        try:
            predictions = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)[:, 1]
            else:
                probabilities = model.decision_function(X_test)

            rows_df.append({
                'Model': name,
                'Precision': precision_score(y_test, predictions, zero_division=0),
                'Recall': recall_score(y_test, predictions, zero_division=0),
                'F1-Score': f1_score(y_test, predictions, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, probabilities),
            })
        except Exception as e:
            print(f"ERROR calculating metrics for model {name}: {e}")
            rows_df.append({'Model': name, 'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'ROC-AUC': 0})
            
    metrics_df = pd.DataFrame(rows_df).round(4)
    
    bar_metrics = go.Figure()
    for metric in ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        bar_metrics.add_trace(go.Bar(
            y=metrics_df["Model"],
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
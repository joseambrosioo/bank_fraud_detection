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

# --- Data Loading (Load all data and models ONCE at startup) ---
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
app.title = "Bank Payments Fraud Detection"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("💰", className="me-2"),
                    dbc.NavbarBrand("Fraud Detection on Bank Payments", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. ASK Tab
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — The Big Picture

    This section defines the core problem and the business value of this project.

    **Business Task**: The primary goal is to build an intelligent system that can accurately identify fraudulent bank transactions from the **Banksim dataset**. Fraud is a multi-billion dollar problem that affects both financial institutions and their customers. By detecting and preventing these fraudulent transactions in real-time, we can minimize financial losses, protect customers, and maintain trust.

    **Stakeholders**: The key users of this dashboard would be **Fraud Analysts**, **Risk Management Teams**, and **Executive Leadership**. They need a clear, easy-to-understand view of the model's performance and the characteristics of fraudulent activity to make informed decisions and deploy effective strategies.

    **Deliverables**: The final product is this interactive dashboard, which presents a comprehensive analysis, showcases the performance of various machine learning models, and offers actionable insights to improve fraud detection.
    """, className="p-4"
)

# 2. PREPARE Tab
prepare_tab = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARE"), " — Getting the Data Ready"], className="mt-4"),
        html.P("Before we can build a predictive model, we need to understand and prepare our data."),
        html.H5("Data Source"),
        html.P([
            "We are using the ",
            html.B("Banksim dataset"),
            ", a synthetically generated dataset that simulates bank payments. It contains almost 600,000 transactions with various features."
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
                            dbc.CardHeader("Imbalanced Data Problem"),
                            dbc.CardBody(
                                [
                                    html.P([
                                        "As is typical with fraud data, the dataset is highly ",
                                        html.B("imbalanced"),
                                        ". Only a tiny fraction of all transactions are fraudulent."
                                    ]),
                                    html.P([
                                        "To fix this, we used a technique called ",
                                        html.B("SMOTE (Synthetic Minority Over-sampling Technique)"),
                                        ". Instead of just copying the fraud examples, SMOTE intelligently creates new, synthetic fraudulent data points that are similar to the existing ones, helping our models learn more effectively."
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
            The dataset includes the following key features:
            - **Step**: The day of the simulation, from 0 to 180 (6 months).
            - **Customer** and **Merchant**: Anonymized IDs for the customer and the merchant.
            - **Age** and **Gender**: Demographic information, categorized into groups.
            - **Category**: The type of purchase (e.g., 'es_travel', 'es_health').
            - **Amount**: The transaction amount.
            - **Fraud**: Our target variable. 1 means fraudulent, 0 means not fraudulent.
            """, className="p-4"
        ),
        html.H5("Dataset Sample (First 10 Rows)"),
        dash_table.DataTable(
            id='table',
            columns=[
                {"name": "age_group" if i == 'age' else i, "id": i, "type": "numeric" if i in ['step', 'amount', 'fraud'] else "text"}
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

# 3. ANALYZE Tab with sub-tabs
analyze_tab = html.Div(
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
                            One of the most significant insights from our data exploration is the difference in transaction amounts. Fraudulent transactions tend to have a much higher amount on average than non-fraudulent ones. Fraudsters often go for high-value targets.
                            """
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="amount-boxplot")),
                        ]),
                        html.P([
                            html.B("Box Plot Insight:"),
                            " The box plot shows the distribution of transaction amounts across different purchase categories. While most categories have a similar amount range, the ",
                            html.B("'es_travel'"),
                            " category stands out with extremely high transaction amounts. This suggests that fraudsters target categories where high-value transactions are common, making their activity less suspicious."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="amount-histogram")),
                        ]),
                        html.P([
                            html.B("Histogram Insight:"),
                            " This graph clearly shows the imbalance in the data. There are far fewer fraudulent transactions, but they are concentrated at much higher amounts, while benign transactions are low-value and very frequent. This is a classic pattern in fraud detection and confirms our hypothesis."
                        ]),
                        html.H5("Fraud by Category and Age", className="mt-4"),
                        html.P("We also explored how fraud is distributed across different purchase categories and age groups."),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="category-fraud-bar"), md=6),
                            dbc.Col(dcc.Graph(id="age-fraud-bar"), md=6),
                        ]),
                        html.P([
                            html.B("Category Fraud Insight:"),
                            " The bar chart shows the percentage of fraudulent transactions within each category. ",
                            html.B("'es_leisure'"),
                            " and ",
                            html.B("'es_travel'"),
                            " have the highest fraud rates, reinforcing the idea that fraudsters target high-value, discretionary spending categories."
                        ]),
                        html.H5("Breakdown by Category"),
                        dcc.Markdown(
                            """
                            - **es_leisure**: Represents transactions for recreational activities, such as entertainment, hobbies, or luxury goods. This category shows a high fraud rate, as these high-value, non-essential purchases are often targeted by fraudsters.
                            - **es_travel**: Transactions related to travel, including flights, hotels, and transportation. This category also has a high fraud rate, likely due to the large ticket sizes, which are attractive to fraudsters.
                            - **es_health**: Transactions for medical or health-related services and products.
                            - **es_hotelservices**: Transactions for hotel stays and related services.
                            - **es_barsandrestaurants**: Payments made at bars and restaurants.
                            - **es_transportation**: Transactions for transportation, such as bus or train tickets.
                            - **es_sportsandoutdoors**: Purchases of sports equipment or outdoor goods.
                            - **es_contents**: Transactions for digital content or media.
                            - **es_fashion**: Purchases of clothing and accessories.
                            - **es_tech**: Transactions for electronics and technology.
                            - **es_home**: Purchases related to home goods and services.
                            - **es_shopping_net**: Online shopping transactions.
                            - **es_others**: A catch-all for transactions that do not fit into other categories.
                            - **es_food**: Transactions for groceries or food items.
                            - **es_service**: Transactions for general services.
                            - **es_shopping**: In-person shopping transactions.
                            """, className="p-4"
                        ),
                        html.P([
                            html.B("Age Fraud Insight:"),
                            " Interestingly, the age group under 18 (category '0') has the highest fraud percentage. This could be due to a number of reasons, such as younger individuals being more susceptible to identity theft or fraudsters intentionally using younger age profiles."
                        ]),
                        html.H5("Breakdown by Age Group"),
                        dcc.Markdown(
                            """
                            - **Age Group 0**: Under 18. This group shows the highest fraud rate, which may be a result of less secure online behavior or the use of stolen credentials on accounts with less rigorous security measures.
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
                            ", and ",
                            html.B("XGBoost"),
                            ". These models are evaluated on a separate 'test' set to ensure they are not just memorizing the training data."
                        ]),
                        html.P(
                            ["To truly evaluate our fraud detection models, we focus on several key metrics beyond simple accuracy:",
                            html.Ul([
                                html.Li([html.B("Precision:"), " Think of Precision as the cost of a false alarm. If our model flags a transaction as fraudulent, high precision means it's very likely to actually be fraudulent. Of all the transactions our model flagged, how many were truly fraudulent? High precision is good to reduce unnecessary investigations."]),
                                html.Li([html.B("Recall:"), " Think of Recall as the cost of a missed fraud. High recall means our model catches most of the actual fraudulent transactions, so we don’t let fraud slip through undetected. Of all fraudulent transactions, how many did our model successfully identify? High recall is crucial to prevent financial losses."]),
                                html.Li([html.B("F1-Score:"), " This is a balance between precision and recall, providing a single metric to compare models. It's the harmonic mean of precision and recall, summarizing both false alarms and missed frauds in one number."]),
                                html.Li([html.B("ROC-AUC:"), " This is a powerful summary metric that measures the model's ability to distinguish between fraudulent and non-fraudulent transactions. A score closer to 1.0 indicates that the model can reliably separate the two classes, making it highly effective for decision-making."])
                            ])
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="model-metrics-bar"), md=12),
                        ]),
                        
                        # New text section to describe model performance with numbers
                        html.P([
                            "Our analysis shows that the ", html.B("XGBoost Classifier"), " performed exceptionally well, achieving a ", html.B("Precision of 0.99"), ", a ", html.B("Recall of 0.99"), ", an ", html.B("F1-Score of 0.99"), ", and a ", html.B("ROC-AUC of 0.99"),
                            ". These results, while incredibly strong, should be considered alongside the full confusion matrix for a complete view of model performance. The ", html.B("K-Neighbors Classifier"), " also performed exceptionally well, with a ", html.B("Precision of 0.98"), ", a ", html.B("Recall of 0.99"), ", an ", html.B("F1-Score of 0.99"), ", and a ", html.B("ROC-AUC of 0.99"),
                            ". The ", html.B("Random Forest Classifier"), " had a lower but still strong performance, with a ", html.B("Precision of 0.97"), ", a ", html.B("Recall of 0.99"), ", an ", html.B("F1-Score of 0.98"), ", and a ", html.B("ROC-AUC of 0.99"), ". The superior performance of XGBoost on some key metrics makes it a leading contender, but the other models also present compelling results."
                        ]),
                        html.Hr(),
                        html.H5("Confusion Matrix & ROC Curve", className="mt-4"),
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
                                html.Li([html.B("False Positives (FP):"), " Incorrectly predicted fraudulent payments (Type I error). These are legitimate transactions flagged as fraud, which can be an inconvenience to customers."]),
                                html.Li([html.B("False Negatives (FN):"), " Incorrectly predicted legitimate payments (Type II error). These are fraudulent transactions that were missed by the model, representing a financial loss for the bank and a security risk for the customer."])
                            ])
                            ]
                        ),
                        html.P([
                            "To provide a more granular view of each model's performance, we can look at the ",
                            html.B("confusion matrix"), " results. The ",
                            html.B("XGBoost Classifier"), " had an accuracy of ",
                            html.B("99.15%"), " with ",
                            html.B("173,997 true positives (TP)"), " and ",
                            html.B("175,490 true negatives (TN)"), ", while only misclassifying ",
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
                            ". The closer the curve is to the top-left corner, the better the model is at distinguishing between the two classes (fraud and non-fraud). The Area Under the Curve (AUC) provides a single metric to summarize the model's performance.",
                        ]),
                        html.P([
                            "The ",
                            html.B("Random Forest Classifier"),
                            " and ",
                            html.B("K-Neighbors Classifier"),
                            " both achieved a perfect ROC curve with an ",
                            html.B("AUC of 1.00"),
                            ", while the ",
                            html.B("XGBoost Classifier"),
                            " was very close with an ",
                            html.B("AUC of 0.99"),
                            ". These results demonstrate the models' excellent ability to differentiate between fraudulent and non-fraudulent transactions. All three models are highly effective at identifying fraud, with the Random Forest and K-Neighbors classifiers performing slightly better in this specific metric."
                        ]),
                        html.Hr(),
                        html.H5("Feature Importance (for tree-based models)", className="mt-4"),
                        html.P("This plot ranks the features based on how much they contributed to the model's prediction."),
                        dcc.Dropdown(
                            id="feature-importance-model-dropdown",
                            options=[
                                {'label': 'Random Forest Classifier', 'value': 'Random Forest Classifier'},
                                {'label': 'XGBoost Classifier', 'value': 'XGBoost Classifier'}
                            ],
                            value='XGBoost Classifier'
                        ),
                        dcc.Graph(id="feature-importance-plot"),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. ACT Tab
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — What to Do Next

    This is the most important section, as it translates our data insights into a business strategy.

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
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(act_tab, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks for Graphs ---
@app.callback(
    Output("amount-boxplot", "figure"),
    Input("amount-histogram", "id") # Dummy input to trigger on load
)
def update_amount_boxplot(dummy):
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
    fig.update_yaxes(range=[0, 1000]) # Set y-axis limit to focus on the main distribution
    return fig

@app.callback(
    Output("amount-histogram", "figure"),
    Input("amount-boxplot", "id") # Dummy input to trigger on load
)
def update_amount_histogram(dummy):
    df_fraud = data[data['fraud'] == 1]
    df_non_fraud = data[data['fraud'] == 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_fraud['amount'], name='Fraudulent', marker_color='red'))
    fig.add_trace(go.Histogram(x=df_non_fraud['amount'], name='Non-Fraudulent', marker_color='blue'))
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
    Output("category-fraud-bar", "figure"),
    Input("age-fraud-bar", "id") # Dummy input to trigger on load
)
def update_category_fraud_bar(dummy):
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
    Output("age-fraud-bar", "figure"),
    Input("category-fraud-bar", "id") # Dummy input to trigger on load
)
def update_age_fraud_bar(dummy):
    fraud_by_age = data.groupby('age')['fraud'].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=fraud_by_age['age'],
        y=fraud_by_age['fraud'] * 100,
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title="Percentage of Fraudulent Transactions by Age",
        xaxis_title="Age Group",
        yaxis_title="Fraud Percentage (%)",
        height=500,
        margin=dict(t=50, b=50)
    )
    return fig

@app.callback(
    Output("model-metrics-bar", "figure"),
    # The Input for this callback is not needed as it's static on load.
    # We use a dummy input to ensure it runs on startup.
    Input('age-fraud-bar', 'id')
)
def update_model_metrics_bar(dummy):
    # Model Metrics Bar Chart
    df_rows = []
    for name, model in model_results.items():
        predictions = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            probabilities = model.decision_function(X_test)

        df_rows.append({
            'Model': name,
            'Precision': precision_score(y_test, predictions, zero_division=0),
            'Recall': recall_score(y_test, predictions, zero_division=0),
            'F1-Score': f1_score(y_test, predictions, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, probabilities),
        })
    metrics_df = pd.DataFrame(df_rows).round(4)
    
    metrics_bar = go.Figure()
    for metric in ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        metrics_bar.add_trace(go.Bar(
            y=metrics_df["Model"],
            x=metrics_df[metric],
            orientation='h',
            name=metric
        ))
    metrics_bar.update_layout(
        barmode='group',
        title="Model Performance Metrics",
        height=450,
        margin=dict(l=150, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return metrics_bar

@app.callback(
    Output("confusion-matrix", "figure"),
    Output("roc-curve", "figure"),
    Input('model-selector-dropdown', 'value')
)
def update_confusion_matrix_roc(selected_model):
    # Retrieve the model from memory, not from disk
    model = model_results[selected_model]

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Non-Fraud (0)", "Predicted Fraud (1)"],
        y=["Actual Non-Fraud (0)", "Actual Fraud (1)"],
        colorscale='blues',
        showscale=False
    )
    cm_fig.update_layout(
        title=f"Confusion Matrix ({selected_model})",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=450,
        margin=dict(t=50, b=50)
    )
    cm_fig.update_yaxes(autorange="reversed")

    # ROC Curve
    roc_fig = go.Figure()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for models without predict_proba
        probabilities = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
    roc_fig.update_layout(
        title=f"ROC Curve ({selected_model})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
        margin=dict(t=50, b=50)
    )
    return cm_fig, roc_fig

@app.callback(
    Output("feature-importance-plot", "figure"),
    Input("feature-importance-model-dropdown", "value")
)
def update_feature_importance(selected_model):
    model = model_results[selected_model]
    
    if hasattr(model, 'feature_importances_'):
        feature_columns = X_test.columns
        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Feature Importances for {selected_model}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"No Feature Importance for {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

if __name__ == "__main__":
    app.run(debug=True)
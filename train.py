# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def pre_process_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Drop columns that don't add predictive value (like static zip codes)
    reduced_data = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    
    # Identify and encode categorical columns
    categorical_columns = reduced_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        reduced_data[col] = reduced_data[col].astype('category')
    
    # Convert categories to numeric codes
    reduced_data[categorical_columns] = reduced_data[categorical_columns].apply(lambda x: x.cat.codes)
    
    # Separate features and target
    X = reduced_data.drop(['fraud'], axis=1)
    y = reduced_data['fraud']
    
    # Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )
    
    return X_train, X_test, y_train, y_test, X.columns

# Pre-process the data
X_train, X_test, y_train, y_test, feature_columns = pre_process_data('dataset/bs140513_032310.csv')

# Define models to train
models_to_train = {
    'K-Neighbors Classifier': KNeighborsClassifier(n_neighbors=5, p=1),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"),
    'XGBoost Classifier': XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, objective="binary:hinge", random_state=42),
}

trained_models = {}
for name, model in models_to_train.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train.values.ravel())
    trained_models[name] = model
    
    # Save the model to a .pkl file
    joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')
    print(f"Model {name} saved to models/{name.replace(' ', '_')}.pkl")

# Save test data and feature names to ensure consistency in the dashboard/evaluation
joblib.dump(X_test, 'data/X_test.pkl')
joblib.dump(y_test, 'data/y_test.pkl')
joblib.dump(feature_columns, 'data/feature_columns.pkl')
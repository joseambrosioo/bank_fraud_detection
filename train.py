# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data_reduced = data.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    col_categorical = data_reduced.select_dtypes(include=['object']).columns
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')
    data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
    X = data_reduced.drop(['fraud'], axis=1)
    y = data_reduced['fraud']
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )
    return X_train, X_test, y_train, y_test, X.columns

X_train, X_test, y_train, y_test, feature_columns = preprocess_data('dataset/bs140513_032310.csv')

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
    # Save the model
    joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')
    print(f"Saved {name} to models/{name.replace(' ', '_')}.pkl")

# You might also want to save your test data to ensure consistency
joblib.dump(X_test, 'data/X_test.pkl')
joblib.dump(y_test, 'data/y_test.pkl')
joblib.dump(feature_columns, 'data/feature_columns.pkl')
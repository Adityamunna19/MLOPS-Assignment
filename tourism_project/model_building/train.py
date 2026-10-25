
from pathlib import Path
import json
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
DEPLOYMENT = ROOT / 'deployment'
DEPLOYMENT.mkdir(exist_ok=True)

train_df = pd.read_csv(ARTIFACTS / 'train.csv')
test_df = pd.read_csv(ARTIFACTS / 'test.csv')
X_train, y_train = train_df.drop(columns=['ProdTaken']), train_df['ProdTaken']
X_test, y_test = test_df.drop(columns=['ProdTaken']), test_df['ProdTaken']

numeric_features = X_train.select_dtypes(include='number').columns.tolist()
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
preprocessor = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features),
])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])
grid = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [8, None],
    'model__min_samples_leaf': [1, 3],
}
search = GridSearchCV(pipeline, grid, scoring='roc_auc', cv=3, n_jobs=-1, refit=True)

mlflow.set_tracking_uri(f'file://{ROOT / "mlruns"}')
mlflow.set_experiment('wellness-tourism-propensity')
with mlflow.start_run():
    search.fit(X_train, y_train)
    proba = search.predict_proba(X_test)[:, 1]
    pred = search.predict(X_test)
    metrics = {
        'roc_auc': round(float(roc_auc_score(y_test, proba)), 4),
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'best_params': search.best_params_,
        'classification_report': classification_report(y_test, pred, output_dict=True),
    }
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics({'roc_auc': metrics['roc_auc'], 'accuracy': metrics['accuracy']})
    mlflow.sklearn.log_model(search.best_estimator_, 'model')

joblib.dump(search.best_estimator_, DEPLOYMENT / 'wellness_propensity_model.joblib')
(ARTIFACTS / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print('Best model saved to deployment/wellness_propensity_model.joblib')

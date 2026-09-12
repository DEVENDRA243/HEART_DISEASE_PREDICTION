import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_csv("heart_disease_combined.csv")
    df = df.drop_duplicates()
    if 'source' in df.columns:
        df = df.drop('source', axis=1)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    continuous_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    nominal_features = ['cp', 'restecg']

    print("Imputing...")
    cont_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()

    X_train_imputed[continuous_features] = cont_imputer.fit_transform(X_train[continuous_features])
    X_test_imputed[continuous_features] = cont_imputer.transform(X_test[continuous_features])

    X_train_imputed[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
    X_test_imputed[categorical_features] = cat_imputer.transform(X_test[categorical_features])

    # Save Imputers
    joblib.dump(cont_imputer, "cont_imputer.joblib")
    joblib.dump(cat_imputer, "cat_imputer.joblib")

    print("Encoding...")
    X_train_imputed[nominal_features] = X_train_imputed[nominal_features].astype(str)
    X_test_imputed[nominal_features] = X_test_imputed[nominal_features].astype(str)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Fit encoder
    encoder.fit(X_train_imputed[nominal_features])
    joblib.dump(encoder, "encoder.joblib")

    encoded_train_cols = encoder.transform(X_train_imputed[nominal_features])
    encoded_test_cols = encoder.transform(X_test_imputed[nominal_features])
    
    encoded_feature_names = encoder.get_feature_names_out(nominal_features)
    
    encoded_train_df = pd.DataFrame(encoded_train_cols, columns=encoded_feature_names, index=X_train.index)
    encoded_test_df = pd.DataFrame(encoded_test_cols, columns=encoded_feature_names, index=X_test.index)

    X_train_encoded = pd.concat([X_train_imputed.drop(nominal_features, axis=1), encoded_train_df], axis=1)
    X_test_encoded = pd.concat([X_test_imputed.drop(nominal_features, axis=1), encoded_test_df], axis=1)

    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
    joblib.dump(scaler, "scaler.joblib")

    # To save column names for the app to know exactly the feature order
    joblib.dump(list(X_train_encoded.columns), "feature_names.joblib")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc'}
    metrics = {}

    # 1. SVM
    print("Training SVM...")
    param_grid_svm = {
        "C": [0.1, 1, 10, 50, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf", "linear", "poly"]
    }
    grid_svm = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), param_grid_svm, cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1)
    grid_svm.fit(X_train_scaled, y_train)
    best_svm = grid_svm.best_estimator_
    joblib.dump(best_svm, "svm.joblib")
    
    y_pred_svm = best_svm.predict(X_test_scaled)
    y_prob_svm = best_svm.predict_proba(X_test_scaled)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    metrics['SVM'] = {
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm),
        'Recall': recall_score(y_test, y_pred_svm),
        'F1-Score': f1_score(y_test, y_pred_svm),
        'ROC-AUC': roc_auc_score(y_test, y_prob_svm),
        'CM': confusion_matrix(y_test, y_pred_svm).tolist(),
        'FPR': fpr_svm.tolist(),
        'TPR': tpr_svm.tolist()
    }

    # 2. Logistic Regression
    print("Training Logistic Regression...")
    param_grid_lr = {
        "C": [0.001, 0.01, 0.1, 1, 10, 50, 100],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [1000]
    }
    grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1)
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_
    joblib.dump(best_lr, "lr.joblib")

    y_pred_lr = best_lr.predict(X_test_scaled)
    y_prob_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    metrics['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'F1-Score': f1_score(y_test, y_pred_lr),
        'ROC-AUC': roc_auc_score(y_test, y_prob_lr),
        'CM': confusion_matrix(y_test, y_pred_lr).tolist(),
        'FPR': fpr_lr.tolist(),
        'TPR': tpr_lr.tolist()
    }

    # 3. Gradient Boosting
    print("Training Gradient Boosting...")
    param_grid_gb = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0]
    }
    grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=cv, scoring=scoring, refit='Accuracy', n_jobs=-1)
    grid_gb.fit(X_train_encoded, y_train)
    best_gb = grid_gb.best_estimator_
    joblib.dump(best_gb, "gb.joblib")

    y_pred_gb = best_gb.predict(X_test_encoded)
    y_prob_gb = best_gb.predict_proba(X_test_encoded)[:, 1]
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
    metrics['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, y_pred_gb),
        'Precision': precision_score(y_test, y_pred_gb),
        'Recall': recall_score(y_test, y_pred_gb),
        'F1-Score': f1_score(y_test, y_pred_gb),
        'ROC-AUC': roc_auc_score(y_test, y_prob_gb),
        'CM': confusion_matrix(y_test, y_pred_gb).tolist(),
        'FPR': fpr_gb.tolist(),
        'TPR': tpr_gb.tolist()
    }

    # Save metrics
    print("Saving metrics...")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # SHAP Explainer
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(best_gb)
    joblib.dump(explainer, "gb_explainer.joblib")

    # Global SHAP Summary Plot
    shap_values = explainer.shap_values(X_train_encoded)
    plt.figure()
    shap.summary_plot(shap_values, X_train_encoded, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()

    print("Training complete! All artifacts saved.")

if __name__ == "__main__":
    main()

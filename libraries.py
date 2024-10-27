import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import optuna
import matplotlib.pyplot as plt

class ModelOptimization:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial):
        classifier_name = trial.suggest_categorical("classifier", ["RandomForest", "LogisticRegression"])
        if classifier_name == "RandomForest":
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
            rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 200)
            rf_max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None])
            rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
            rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 4)
            base_classifier = RandomForestClassifier(
                max_depth=rf_max_depth, 
                n_estimators=rf_n_estimators, 
                max_features=rf_max_features,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=42
            )
        else:
            lr_C = trial.suggest_float("lr_C", 1e-10, 1e10, log=True)
            lr_penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2", "elasticnet"])
            lr_solver = 'saga' if lr_penalty == 'elasticnet' else trial.suggest_categorical("lr_solver", ["liblinear", "saga"])
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if lr_penalty == "elasticnet" else None
            base_classifier = LogisticRegression(
                C=lr_C, 
                penalty=lr_penalty, 
                solver=lr_solver, 
                l1_ratio=l1_ratio, 
                max_iter=10000, 
                random_state=42
            )

        calibrated_classifier = CalibratedClassifierCV(base_classifier, method='sigmoid', cv=3)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(calibrated_classifier, self.X_train, self.y_train, cv=skf, scoring="roc_auc").mean()
        return score

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        return self.best_params

class ModelTraining:
    def __init__(self, best_params, X_train, y_train, X_val, y_val):
        self.best_params = best_params
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_model(self):
        if self.best_params["classifier"] == "RandomForest":
            base_classifier = RandomForestClassifier(
                max_depth=self.best_params["rf_max_depth"],
                n_estimators=self.best_params["rf_n_estimators"],
                max_features=self.best_params["rf_max_features"],
                min_samples_split=self.best_params["rf_min_samples_split"],
                min_samples_leaf=self.best_params["rf_min_samples_leaf"],
                random_state=42
            )
        else:
            base_classifier = LogisticRegression(
                C=self.best_params["lr_C"], 
                penalty=self.best_params["lr_penalty"],
                solver=self.best_params["lr_solver"], 
                l1_ratio=self.best_params["l1_ratio"] if self.best_params["lr_penalty"] == "elasticnet" else None,
                max_iter=10000, 
                random_state=42
            )

        self.calibrated_classifier = CalibratedClassifierCV(base_classifier, method='sigmoid', cv=3)
        self.calibrated_classifier.fit(self.X_train, self.y_train)
        return self.calibrated_classifier

    def validate_model(self):
        y_val_pred = self.calibrated_classifier.predict_proba(self.X_val)[:, 1]
        roc_auc = roc_auc_score(self.y_val, y_val_pred)
        print(f"Validation ROC-AUC: {roc_auc:.2f}")

    

class SubmissionPreparation:
    def __init__(self, model, X_test_final, test_ids):
        self.model = model
        self.X_test_final = X_test_final
        self.test_ids = test_ids

    def create_submission(self, filename='submission.csv'):
        y_test_pred = self.model.predict_proba(self.X_test_final)[:, 1]
        submission = pd.DataFrame({'id': self.test_ids, 'target': y_test_pred})
        submission.to_csv(filename, index=False)
        print(f'Submission file {filename} created.')
        
        
        
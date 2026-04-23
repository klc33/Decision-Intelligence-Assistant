#!/usr/bin/env python
"""
Train ML classifier for ticket priority with train/validation/test split.

This script:
1. Loads the cleaned labeled dataset.
2. Splits into train (70%), validation (15%), test (15%) – stratified.
3. Extracts features and builds a preprocessing pipeline (imputation + scaling).
4. Uses 3-fold cross-validation on the training set to tune hyperparameters.
5. Selects the best model based on validation set performance.
6. Evaluates final model on the held-out test set.
7. Saves the pipeline and metrics.

Usage:
    python training/train.py
"""

import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

from features import extract_features, get_feature_names, get_numeric_feature_names

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = Path("data/processed/tickets_with_labels.csv")
OUTPUT_DIR = Path("training/outputs")
MODEL_PATH_TRAINING = OUTPUT_DIR / "models" / "priority_classifier.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
BACKEND_MODEL_PATH = Path("backend/app/models/priority_classifier.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.15       # 15% for final test
VALIDATION_SIZE = 0.15 # 15% for validation (of the original data)
TRAIN_SIZE = 0.70      # 70% for training
N_JOBS = -1            # Use all CPU cores
CV_FOLDS = 3           # 3-fold cross-validation


# ============================================================================
# Data Loading and Preparation
# ============================================================================
def load_and_prepare_data():
    """Load cleaned data and prepare features/labels."""
    print("📂 Loading cleaned data...")
    if not DATA_PATH.exists():
        print(f"❌ Data not found at {DATA_PATH}")
        print("   Please run Phase 1 first: python scripts/prepare_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} rows")

    print("🔧 Extracting features...")
    X = extract_features(df, text_column='text')
    y = df['priority'].map({'urgent': 1, 'normal': 0})

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Label distribution:\n{y.value_counts()}")
    return X, y, get_feature_names()


def split_data(X, y):
    """
    Split data into train, validation, and test sets with stratification.

    First split: train+val (85%) and test (15%)
    Second split: train (70% of original) and validation (15% of original)
    """
    # Split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split remaining into train and validation
    # validation size relative to temp = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    val_relative = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"\n📊 Data split:")
    print(f"   Train:      {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Preprocessing Pipeline
# ============================================================================
def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build a column transformer that:
    - Imputes missing numeric values with median
    - Scales numeric features
    - Leaves binary features unchanged
    """
    numeric_features = get_numeric_feature_names()
    # Keep only those present in the data
    numeric_features = [f for f in numeric_features if f in X_train.columns]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'  # binary features remain as-is
    )
    return preprocessor


# ============================================================================
# Model Training with Cross-Validation
# ============================================================================
def train_models(X_train, y_train, X_val, y_val, preprocessor):
    """
    Train multiple models using 3-fold CV on the training set for hyperparameter tuning.
    Then evaluate the best model on the validation set.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__class_weight': [None, 'balanced']
        },
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        }
    }
    if XGB_AVAILABLE:
        param_grids['XGBoost'] = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [6, 10],
            'classifier__learning_rate': [0.1, 0.3]
        }

    best_model = None
    best_val_score = 0
    best_name = None
    results = {}

    print("\n🤖 Training models with 3-fold CV on training set...")
    for name, classifier in models.items():
        print(f"\n--- {name} ---")
        start_time = time.time()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        # Grid search with 3-fold CV on training data
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=CV_FOLDS,
            scoring='f1',
            n_jobs=N_JOBS,
            verbose=1
        )
        grid.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = grid.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred)

        train_time = time.time() - start_time

        results[name] = {
            'best_params': grid.best_params_,
            'best_cv_score': float(grid.best_score_),
            'validation_f1': float(val_f1),
            'train_time_sec': train_time
        }

        print(f"   Best params: {grid.best_params_}")
        print(f"   CV F1 (mean): {grid.best_score_:.4f}")
        print(f"   Validation F1: {val_f1:.4f}")
        print(f"   Training time: {train_time:.1f}s")

        if val_f1 > best_val_score:
            best_val_score = val_f1
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n🏆 Best model based on validation set: {best_name} (Validation F1: {best_val_score:.4f})")
    return best_model, best_name, results


# ============================================================================
# Final Evaluation on Test Set
# ============================================================================
def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the final pipeline on the held-out test set."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred))
    }

    print("\n📊 Test Set Performance (final evaluation):")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['normal', 'urgent']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:\n{cm}")

    return metrics, cm


# ============================================================================
# Save Model and Metadata
# ============================================================================
def save_model_and_metadata(pipeline, feature_names, model_name, cv_results, test_metrics):
    """Save the model pipeline and all metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH_TRAINING.parent.mkdir(parents=True, exist_ok=True)
    BACKEND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'numeric_features': get_numeric_feature_names(),
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics
    }

    joblib.dump(model_package, MODEL_PATH_TRAINING)
    print(f"\n💾 Model saved to {MODEL_PATH_TRAINING}")

    joblib.dump(model_package, BACKEND_MODEL_PATH)
    print(f"💾 Model copied to {BACKEND_MODEL_PATH}")

    # Save metrics as JSON
    all_metrics = {
        'model_name': model_name,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'feature_count': len(feature_names),
        'data_split': {
            'train_size': TRAIN_SIZE,
            'validation_size': VALIDATION_SIZE,
            'test_size': TEST_SIZE
        }
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"📄 Metrics saved to {METRICS_PATH}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Phase 2: ML Model Training (Train/Val/Test Split + 3-Fold CV)")
    print("=" * 60)

    # 1. Load data and extract features
    X, y, feature_names = load_and_prepare_data()

    # 2. Train/validation/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Build preprocessor (fit only on training data to avoid leakage)
    preprocessor = build_preprocessor(X_train)

    # 4. Train models with CV on training set, select best on validation set
    best_pipeline, best_name, cv_results = train_models(
        X_train, y_train, X_val, y_val, preprocessor
    )

    # 5. Final evaluation on test set
    test_metrics, cm = evaluate_model(best_pipeline, X_test, y_test)

    # 6. Save everything
    save_model_and_metadata(best_pipeline, feature_names, best_name, cv_results, test_metrics)

    print("\n✅ Phase 2 complete! Model ready for backend inference.")


if __name__ == "__main__":
    main()
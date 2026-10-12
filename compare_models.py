# compare_models.py
# College Mini-Project: Social Media / Device Addiction Prediction
# Compares 6 Machine Learning algorithms on a stratified sample of the dataset.

import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Machine learning classifiers to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==============================================================================
# SAMPLE CONFIGURATION
# The full training dataset contains over 690,000 rows.
# Algorithms like SVM (O(n^2) to O(n^3) complexity) and KNN take excessive time
# on 690k rows. Therefore, we use a representative stratified sample of 50,000 rows
# to fairly compare all 6 models within a reasonable amount of time.
# ==============================================================================
SAMPLE_SIZE = 50000
RANDOM_STATE = 42

# List of numerical and categorical feature names
NUMERIC_FEATURES = [
    'age',
    'daily_screen_time_hours',
    'social_media_hours',
    'gaming_hours',
    'work_study_hours',
    'sleep_hours',
    'notifications_per_day',
    'app_opens_per_day',
    'weekend_screen_time'
]

CATEGORICAL_FEATURES = [
    'gender',
    'stress_level',
    'academic_work_impact'
]

TARGET = 'addicted_label'


def load_and_sample_data(filepath, sample_size, random_state):
    """
    Step 1: Load the full training data and draw a stratified random sample.
    Stratified sampling ensures the class balance (addicted vs. not addicted) is preserved.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Full dataset shape: {df.shape}")

    # Draw a stratified random sample based on the target column
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[TARGET],
        random_state=random_state
    )
    print(f"Sampled dataset shape: {sampled_df.shape}")
    return sampled_df


def create_preprocessor():
    """
    Step 2: Create a ColumnTransformer for preprocessing.
    - Numeric features: Median imputation (robust to outliers) + Standard scaling.
    - Categorical features: Most-frequent imputation + One-Hot Encoding.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def get_models(random_state):
    """
    Step 3: Define the 6 candidate classification algorithms.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Naive Bayes (GaussianNB)': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (SVC)': SVC(probability=True, random_state=random_state)
    }
    return models


def evaluate_models(models, preprocessor, X_train, X_val, y_train, y_val):
    """
    Step 4: Train each model pipeline and evaluate on the validation set.
    Metrics evaluated: ROC AUC (primary), Accuracy, Precision, Recall, F1-Score.
    """
    results = []

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        # Build end-to-end Pipeline combining preprocessing and classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Train model pipeline
        pipeline.fit(X_train, y_train)
        train_duration = time.time() - start_time

        # Predict probabilities and discrete classes
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)

        # Calculate metrics
        roc_auc = roc_auc_score(y_val, y_prob)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        print(f"Finished {name} in {train_duration:.2f}s | ROC AUC: {roc_auc:.4f}")

        results.append({
            'Model': name,
            'ROC AUC': round(roc_auc, 4),
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4)
        })

    return pd.DataFrame(results)


def main():
    # 1. Load and sample data
    data = load_and_sample_data('train.csv', SAMPLE_SIZE, RANDOM_STATE)

    # 2. Separate features (X) and target (y) - 'id' column is excluded
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    # 3. Stratified 80/20 train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"Train split size: {X_train.shape[0]} samples")
    print(f"Validation split size: {X_val.shape[0]} samples")

    # 4. Build preprocessor and models
    preprocessor = create_preprocessor()
    models = get_models(RANDOM_STATE)

    # 5. Evaluate all models
    results_df = evaluate_models(models, preprocessor, X_train, X_val, y_train, y_val)

    # 6. Sort results table by ROC AUC descending (primary metric)
    results_df = results_df.sort_values(by='ROC AUC', ascending=False).reset_index(drop=True)

    # 7. Print and save results
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS (Sorted by ROC AUC)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    results_file = 'comparison_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved comparison results to '{results_file}'.")

    # 8. Report the best model and viva explanation
    best_model_name = results_df.iloc[0]['Model']
    best_auc = results_df.iloc[0]['ROC AUC']
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name} (ROC AUC = {best_auc:.4f})")
    print("=" * 70)
    print(
        f"Plain-English Reason:\n"
        f"1. {best_model_name} achieved the highest ROC AUC score ({best_auc:.4f}), which measures how well\n"
        f"   the model separates addicted users from non-addicted users across all classification thresholds.\n"
        f"2. Unlike plain accuracy, ROC AUC is unaffected by class imbalance (~71% positive in this dataset)\n"
        f"   and confirms superior diagnostic discrimination for the mini-project application."
    )


if __name__ == '__main__':
    main()

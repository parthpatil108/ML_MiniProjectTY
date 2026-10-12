# Smartphone & Social Media Addiction Prediction (TYBtech ML Mini-Project)

A simple, readable, and end-to-end Machine Learning project designed for second-year B.Tech students. This project predicts whether an individual is at risk of device/social media addiction based on daily lifestyle and digital usage habits.

---

## 1. Project Overview

- **Target Variable**: `addicted_label`
  - `1`: Addicted
  - `0`: Not Addicted
- **Features Used (12 total)**:
  - **9 Numeric Features**: `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time`.
  - **3 Categorical Features**: `gender` (`Female`, `Male`, `Other`), `stress_level` (`Low`, `Medium`, `High`), `academic_work_impact` (`No`, `Yes`).
  - *Note*: `id` is an identifier and is strictly excluded from model training.
- **Data Preprocessing**:
  - **Numeric Features**: Handled using `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Median imputation is robust against outliers, and scaling ensures distance/gradient-based models function properly.
  - **Categorical Features**: Handled using `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Imputation is used instead of dropping rows so no valuable sample data is lost.

---

## 2. Project File Structure

```text
miniproject/
│
├── train.csv                 # Training dataset (691,369 rows)
├── test.csv                  # Test dataset (unlabeled, 296,302 rows)
├── sample_submission.csv     # Sample submission format
├── requirements.txt          # Pinned library dependencies
│
├── compare_models.py         # Step 1: Benchmark 6 ML models on a 50k stratified sample
├── comparison_results.csv    # Benchmark metrics table
│
├── train_final_model.py      # Step 2: Dynamically train winner on full training split & save
├── model.joblib              # Saved scikit-learn Pipeline (< 100 MB, viva-safe)
├── model_metadata.json       # Feature names, categories, and validation scores
│
├── app.py                    # Step 3: Desktop Tkinter Graphical User Interface
├── AddictionPredictor.spec   # PyInstaller build specification file
├── dist/
│   └── AddictionPredictor.exe# Step 4: Standalone Windows executable
└── README.md                 # Project documentation and viva guide
```

---

## 3. How to Run Each Script in Order

### Prerequisites
Install all required libraries using the pinned versions:
```bash
pip install -r requirements.txt
```

### Step 1: Model Comparison
Run `compare_models.py` to compare 6 standard scikit-learn algorithms on a stratified random sample of 50,000 rows (80/20 train/validation split):
```bash
python compare_models.py
```
- Algorithms tested: Logistic Regression, Decision Tree, Random Forest, Naive Bayes (GaussianNB), K-Nearest Neighbors, SVM (SVC).
- Outputs a sorted table and saves `comparison_results.csv`.

### Step 2: Final Model Training
Run `train_final_model.py` to automatically select the top-performing model from `comparison_results.csv`, train it on the main dataset split, and save the pipeline:
```bash
python train_final_model.py
```
- Trains on 553,095 training samples and validates on 138,274 held-out samples.
- Limits tree depth (`max_depth=15, min_samples_leaf=5`) to prevent overfitting and keep the serialized model compact (~26.7 MB, well below the 100 MB threshold).
- Generates `model.joblib` and `model_metadata.json`.

### Step 3: Desktop GUI Application
Run `app.py` to open the graphical user interface:
```bash
python app.py
```
- Enter numeric habits (or leave them empty to use median imputation).
- Select categorical values from dropdowns.
- Click **Predict** to view the classification result and addiction risk probability.
- Click **Clear** to reset all input fields.

### Step 4: Run the Standalone Windows Executable
You can run the pre-built executable without needing Python or external libraries:
```bash
.\dist\AddictionPredictor.exe
```

---

## 4. Model Comparison Results

All 6 algorithms were benchmarked using a stratified 80/20 validation split (`random_state=42`) on 50,000 samples:

| Model | ROC AUC (Primary) | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9321** | **85.79%** | **89.31%** | **90.84%** | **0.9007** |
| **SVM (SVC)** | 0.9195 | 84.82% | 88.40% | 90.47% | 0.8942 |
| **Logistic Regression** | 0.9123 | 83.63% | 86.47% | 91.19% | 0.8877 |
| **Naive Bayes (GaussianNB)** | 0.9018 | 83.08% | 90.90% | 84.62% | 0.8765 |
| **K-Nearest Neighbors** | 0.8714 | 82.13% | 86.23% | 89.03% | 0.8761 |
| **Decision Tree** | 0.7746 | 81.23% | 86.99 | 86.47% | 0.8673 |

### Why Random Forest is the Winner:
1. **Highest ROC AUC (0.9321)**: Random Forest demonstrated the highest discrimination ability across all classification thresholds.
2. **Balanced Performance**: It achieved the highest overall Accuracy (85.79%) and F1 Score (0.9007), accurately capturing non-linear interactions among digital habits.
3. **Full Dataset Validation**: When trained on the 553,095 training samples with tuned depth, the final validation ROC AUC reached **0.9386** and Accuracy reached **86.55%**.

---

## 5. Standalone .exe Build Command

The standalone Windows executable was compiled using **PyInstaller**:

```bash
python -m PyInstaller --noconfirm --onefile --windowed --add-data "model.joblib;." --add-data "model_metadata.json;." --collect-submodules sklearn --collect-data sklearn --name AddictionPredictor app.py
```

### Explanation of Flags:
- `--onefile`: Bundles the Python interpreter, runtime libraries, and assets into a single executable (`AddictionPredictor.exe`).
- `--windowed`: Suppresses the background command prompt window so only the GUI appears.
- `--add-data "model.joblib;."`: Packages the trained pipeline inside the bundle root directory.
- `--add-data "model_metadata.json;."`: Packages feature names and dropdown options inside the bundle root directory.
- `--collect-submodules sklearn`: Ensures dynamically imported scikit-learn modules are included.
- `sys._MEIPASS`: In `app.py`, the `get_resource_path()` function checks `sys._MEIPASS` when running inside the PyInstaller temporary extraction folder and falls back to the current directory when running as standard Python code.

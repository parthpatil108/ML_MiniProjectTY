# app.py
# College Mini-Project: Social Media / Device Addiction Prediction
# Simple and readable desktop application built using Tkinter.

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, messagebox


def get_resource_path(relative_path):
    """
    Step 1: Resource path helper.
    Returns the absolute path to a file. Works both during normal Python execution
    and inside a PyInstaller standalone executable (using sys._MEIPASS).
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def load_model_and_metadata():
    """
    Step 2: Load the saved pipeline (preprocessing + model) and metadata.
    """
    model_path = get_resource_path('model.joblib')
    metadata_path = get_resource_path('model_metadata.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run train_final_model.py first.")

    pipeline = joblib.load(model_path)

    metadata = {
        'model_name': 'Random Forest',
        'val_roc_auc': 0.9386,
        'numeric_features': [
            'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time'
        ],
        'categorical_features': ['gender', 'stress_level', 'academic_work_impact'],
        'categorical_options': {
            'gender': ['Female', 'Male', 'Other'],
            'stress_level': ['Low', 'Medium', 'High'],
            'academic_work_impact': ['No', 'Yes']
        }
    }

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return pipeline, metadata


# Friendly display labels for features
FEATURE_LABELS = {
    'age': 'Age (years):',
    'daily_screen_time_hours': 'Daily Screen Time (hours):',
    'social_media_hours': 'Social Media (hours):',
    'gaming_hours': 'Gaming (hours):',
    'work_study_hours': 'Work / Study (hours):',
    'sleep_hours': 'Sleep (hours):',
    'notifications_per_day': 'Notifications / Day:',
    'app_opens_per_day': 'App Opens / Day:',
    'weekend_screen_time': 'Weekend Screen Time (hours):',
    'gender': 'Gender:',
    'stress_level': 'Stress Level:',
    'academic_work_impact': 'Academic/Work Impact:'
}


class AddictionPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Addiction Predictor - College ML Mini-Project")
        self.root.geometry("620x760")
        self.root.minsize(580, 700)

        # Load trained ML pipeline and metadata
        self.pipeline, self.metadata = load_model_and_metadata()
        self.model_name = self.metadata.get('model_name', 'Random Forest')
        self.auc_score = self.metadata.get('val_roc_auc', 0.9386)

        # Input storage dictionaries
        self.numeric_entries = {}
        self.categorical_combos = {}

        self.setup_ui()

    def setup_ui(self):
        """
        Step 3: Build the user interface components.
        """
        # Header banner
        header_frame = tk.Frame(self.root, bg="#1E3A8A", pady=14)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text="Device / Social Media Addiction Predictor",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#1E3A8A"
        )
        title_lbl.pack()

        subtitle_lbl = tk.Label(
            header_frame,
            text=f"College ML Mini-Project | Powered by {self.model_name} (ROC AUC: {self.auc_score:.4f})",
            font=("Arial", 10),
            fg="#93C5FD",
            bg="#1E3A8A"
        )
        subtitle_lbl.pack(pady=2)

        # Scrollable / Container frame
        container = tk.Frame(self.root, padx=20, pady=10)
        container.pack(fill="both", expand=True)

        instructions = tk.Label(
            container,
            text="Enter user metrics below. Leave numeric fields blank to use automatic median imputation.",
            font=("Arial", 9, "italic"),
            fg="#4B5563"
        )
        instructions.pack(pady=(0, 10))

        # Form fields container
        form_frame = tk.LabelFrame(container, text=" User Input Features ", font=("Arial", 11, "bold"), padx=15, pady=10)
        form_frame.pack(fill="x", expand=False)

        # Numeric fields
        row_idx = 0
        for num_feat in self.metadata['numeric_features']:
            lbl_text = FEATURE_LABELS.get(num_feat, num_feat.replace('_', ' ').title() + ":")
            lbl = tk.Label(form_frame, text=lbl_text, font=("Arial", 10), anchor="w")
            lbl.grid(row=row_idx, column=0, sticky="w", pady=4, padx=(0, 10))

            entry = tk.Entry(form_frame, font=("Arial", 10), width=22)
            entry.grid(row=row_idx, column=1, sticky="e", pady=4)
            self.numeric_entries[num_feat] = entry
            row_idx += 1

        # Separator line
        sep = ttk.Separator(form_frame, orient="horizontal")
        sep.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=8)
        row_idx += 1

        # Categorical dropdown fields
        for cat_feat in self.metadata['categorical_features']:
            lbl_text = FEATURE_LABELS.get(cat_feat, cat_feat.replace('_', ' ').title() + ":")
            lbl = tk.Label(form_frame, text=lbl_text, font=("Arial", 10), anchor="w")
            lbl.grid(row=row_idx, column=0, sticky="w", pady=4, padx=(0, 10))

            options = self.metadata['categorical_options'].get(cat_feat, [])
            combo = ttk.Combobox(form_frame, values=options, font=("Arial", 10), width=20, state="readonly")
            if options:
                combo.current(0)
            combo.grid(row=row_idx, column=1, sticky="e", pady=4)
            self.categorical_combos[cat_feat] = combo
            row_idx += 1

        # Buttons frame (Predict & Clear)
        btn_frame = tk.Frame(container, pady=12)
        btn_frame.pack()

        predict_btn = tk.Button(
            btn_frame,
            text="Predict Addiction",
            font=("Arial", 11, "bold"),
            bg="#2563EB",
            fg="white",
            padx=18,
            pady=6,
            command=self.predict
        )
        predict_btn.grid(row=0, column=0, padx=10)

        clear_btn = tk.Button(
            btn_frame,
            text="Clear Form",
            font=("Arial", 11),
            bg="#E5E7EB",
            fg="#1F2937",
            padx=18,
            pady=6,
            command=self.clear_form
        )
        clear_btn.grid(row=0, column=1, padx=10)

        # Prediction results card
        self.result_frame = tk.Frame(container, relief="groove", borderwidth=2, padx=15, pady=12, bg="#F9FAFB")
        self.result_frame.pack(fill="x", pady=10)

        self.result_heading = tk.Label(
            self.result_frame,
            text="Prediction Result:",
            font=("Arial", 11, "bold"),
            bg="#F9FAFB",
            fg="#374151"
        )
        self.result_heading.pack()

        self.result_label = tk.Label(
            self.result_frame,
            text="Please fill features and click 'Predict Addiction'.",
            font=("Arial", 12),
            bg="#F9FAFB",
            fg="#6B7280"
        )
        self.result_label.pack(pady=4)

        self.prob_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 10),
            bg="#F9FAFB",
            fg="#4B5563"
        )
        self.prob_label.pack()

        # Footer
        footer_lbl = tk.Label(
            self.root,
            text=f"Chosen Model: {self.model_name} | Trained on full preprocessed dataset | Threshold: 0.50",
            font=("Arial", 9),
            fg="#6B7280",
            pady=8
        )
        footer_lbl.pack(side="bottom")

    def predict(self):
        """
        Step 4: Validate inputs, build data row, and run the pipeline.
        - Blank numeric fields are converted to NaN (imputed by pipeline median).
        - Non-numeric inputs raise a friendly validation error.
        """
        row_data = {}

        # Validate numeric inputs
        for num_feat, entry in self.numeric_entries.items():
            val_text = entry.get().strip()
            if val_text == "":
                row_data[num_feat] = np.nan
            else:
                try:
                    num_val = float(val_text)
                    if num_val < 0:
                        messagebox.showwarning(
                            "Invalid Input",
                            f"Value for '{FEATURE_LABELS.get(num_feat, num_feat)}' cannot be negative."
                        )
                        return
                    row_data[num_feat] = num_val
                except ValueError:
                    messagebox.showerror(
                        "Number Format Error",
                        f"Please enter a valid numeric number for:\n'{FEATURE_LABELS.get(num_feat, num_feat)}'\n"
                        f"(Or leave it empty for automatic median imputation)."
                    )
                    return

        # Get categorical inputs
        for cat_feat, combo in self.categorical_combos.items():
            row_data[cat_feat] = combo.get()

        # Build DataFrame with the exact feature order
        all_cols = self.metadata['numeric_features'] + self.metadata['categorical_features']
        input_df = pd.DataFrame([row_data], columns=all_cols)

        try:
            # Predict probability of class 1 (Addicted)
            prob_addicted = self.pipeline.predict_proba(input_df)[0, 1]
            prob_percent = prob_addicted * 100

            # Classification using standard 0.5 threshold
            if prob_addicted >= 0.50:
                result_text = "ADDICTED"
                card_bg = "#FEE2E2"  # Light red
                text_fg = "#991B1B"  # Dark red
            else:
                result_text = "NOT ADDICTED"
                card_bg = "#DCFCE7"  # Light green
                text_fg = "#166534"  # Dark green

            # Update UI result display
            self.result_frame.config(bg=card_bg)
            self.result_heading.config(bg=card_bg, fg=text_fg)
            self.result_label.config(
                text=f"Predicted Class: {result_text}",
                font=("Arial", 14, "bold"),
                bg=card_bg,
                fg=text_fg
            )
            self.prob_label.config(
                text=f"Estimated Addiction Probability: {prob_percent:.1f}% (Threshold: 50.0%)",
                bg=card_bg,
                fg=text_fg
            )

        except Exception as err:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{err}")

    def clear_form(self):
        """
        Step 5: Reset all entry fields, dropdowns, and results.
        """
        for entry in self.numeric_entries.values():
            entry.delete(0, tk.END)

        for cat_feat, combo in self.categorical_combos.items():
            options = self.metadata['categorical_options'].get(cat_feat, [])
            if options:
                combo.current(0)

        # Reset results card
        self.result_frame.config(bg="#F9FAFB")
        self.result_heading.config(bg="#F9FAFB", fg="#374151")
        self.result_label.config(
            text="Please fill features and click 'Predict Addiction'.",
            font=("Arial", 12),
            bg="#F9FAFB",
            fg="#6B7280"
        )
        self.prob_label.config(text="", bg="#F9FAFB")


def main():
    # If invoked with --test-predict, run automated prediction verification without GUI
    if len(sys.argv) > 1 and sys.argv[1] == '--test-predict':
        pipeline, metadata = load_model_and_metadata()
        sample_dict = {
            'age': 22.0,
            'daily_screen_time_hours': 9.5,
            'social_media_hours': 5.0,
            'gaming_hours': 2.5,
            'work_study_hours': 3.0,
            'sleep_hours': 6.0,
            'notifications_per_day': 130.0,
            'app_opens_per_day': 90.0,
            'weekend_screen_time': 11.0,
            'gender': 'Male',
            'stress_level': 'High',
            'academic_work_impact': 'Yes'
        }
        all_cols = metadata['numeric_features'] + metadata['categorical_features']
        df_test = pd.DataFrame([sample_dict], columns=all_cols)
        prob = float(pipeline.predict_proba(df_test)[0, 1])
        pred = int(pipeline.predict(df_test)[0])
        pred_label = "Addicted" if pred == 1 else "Not Addicted"
        
        output_line = f"CLASS={pred_label} | PROB={prob:.6f}"
        print(output_line)
        with open("exe_test_output.txt", "w") as f:
            f.write(output_line + "\n")
        return

    root = tk.Tk()
    app = AddictionPredictorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

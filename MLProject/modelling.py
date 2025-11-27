import pandas as pd
import time
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import dagshub

# --- SETUP DAGSHUB OTOMATIS ---
dagshub.init(repo_owner='farhanhanifazhary', repo_name='Heart-Failure-Tracking', mlflow=True)

# --- 2. LOAD DATA ---
df = pd.read_csv('heart_failure_clinical_records_dataset_preprocessing.csv')

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAINING DENGAN LOGGING ADVANCE ---
print("Memulai Training...")

# Matikan autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Autolog") as run:
    
    # 1. Training & Hitung Waktu (Manual Metric 1)
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    
    # 2. Evaluasi Tambahan
    # Autolog sudah mencatat Accuracy/F1 standar.
    # Kita tambahkan metrik KHUSUS yang autolog kadang lewatkan (Syarat Advance).
    y_pred = model.predict(X_test)
    
    # Hitung Specificity (Manual Metric 2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    print(f"Training Duration: {duration}s")
    print(f"Specificity: {specificity}")

    # 3. LOGGING TAMBAHAN (Hybrid)
    # Autolog sudah log model & metrik dasar.
    # Kita CUKUP log metrik tambahan saja di sini.
    mlflow.log_metric("training_duration", duration)
    mlflow.log_metric("specificity", specificity)

    # Kunci untuk Docker
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print("Selesai!")
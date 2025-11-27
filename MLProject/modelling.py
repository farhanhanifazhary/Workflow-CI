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
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="RandomForest_Advance_Manual"):
    
    # A. Parameter Model
    n_estimators = 100
    max_depth = 10
    
    # Log Parameter
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # B. Training & Hitung Waktu
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # C. Prediksi & Evaluasi
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Hitung Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Time: {training_duration:.4f}s")
    
    # D. LOGGING KE DAGSHUB
    # Metrik Wajib
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Metrik Tambahan (Syarat Advance)
    mlflow.log_metric("training_duration", training_duration)
    mlflow.log_metric("specificity", specificity)
    
    # Simpan Model
    mlflow.sklearn.log_model(model, "model_random_forest")
    
    print("Selesai!")
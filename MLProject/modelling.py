import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub
import os

# --- DEBUGGING PATH ---
print(f"📂 Current Working Directory: {os.getcwd()}")
print(f"📂 Isi folder saat ini: {os.listdir()}")

# --- SETUP ---
try:
    print("🔌 Menghubungkan ke DagsHub...")
    dagshub.init(repo_owner='farhanhanifazhary', repo_name='Heart-Failure-Tracking', mlflow=True)
    
    print("💾 Loading Data...")
    df = pd.read_csv('heart_failure_clinical_records_dataset_preprocessing.csv')
    
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🚀 Memulai Training...")
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="RandomForest_Docker") as run:
        # 1. Training
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Model Trained.")
        
        # 2. Logging
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        
        # 3. Save Model
        mlflow.sklearn.log_model(model, "model_random_forest")
        print("✅ Model Logged to MLflow.")
        
        # 4. SAVE RUN ID (CRITICAL STEP)
        run_id = run.info.run_id
        print(f"🆔 Run ID ditemukan: {run_id}")
        
        output_file = "run_id.txt"
        with open(output_file, "w") as f:
            f.write(run_id)
        
        # Verifikasi langsung apakah file tertulis
        if os.path.exists(output_file):
            print(f"🎉 SUKSES: File {output_file} berhasil dibuat di {os.getcwd()}")
        else:
            print(f"❌ GAGAL: File {output_file} TIDAK ditemukan setelah penulisan!")

except Exception as e:
    print(f"🔥 ERROR FATAL TERJADI: {e}")
    # Raise error agar GitHub Actions tahu ini gagal (Merah)
    raise e
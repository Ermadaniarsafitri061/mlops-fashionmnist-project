# MLOps Fashion-MNIST — MLflow + Optuna + DVC

Implementasi tugas praktikum Minggu 6 menggunakan Fashion-MNIST dan CNN kecil.

## 1. Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Jalankan MLflow Tracking Server
Terminal 1:
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
Buka http://127.0.0.1:5000. Biarkan terminal ini tetap berjalan.

## 3. Download dataset dan catat versi dengan DVC
Terminal 2:
```powershell
python src/prepare_data.py
dvc init
dvc add data/FashionMNIST
git add .dvc .gitignore data/FashionMNIST.dvc
```
Folder data bisa cukup besar; jangan commit isi dataset ke Git. Jika diminta menyimpan data ke remote, konfigurasi remote DVC secara terpisah. Script training mencatat SHA-256 dataset ke setiap run MLflow.

## 4. Jalankan eksperimen
Jalankan berurutan dari root project:
```powershell
python src/experiment.py baseline
python src/experiment.py tune --trials 20
python src/experiment.py seeds
python src/experiment.py ablation
python src/experiment.py register
python src/experiment.py predict
```
Untuk pengujian cepat sebelum eksperimen penuh, coba `python src/experiment.py baseline --epochs 1`. Untuk tugas final, pastikan tuning minimal 20 trial selesai.

## 5. Deliverables
- Repositori Git berisi kode, requirements, file `.dvc`, dan laporan.
- Screenshot MLflow UI: baseline, nested runs/compare view, seed runs, ablation, registry alias `champion`.
- Laporan singkat: hipotesis, hasil aktual, mean ± std dari tiga seed, hasil ablation, kesimpulan.
- Jangan mengarang angka hasil. Ambil metrik aktual dari MLflow UI atau output terminal.

## Catatan
- Dataset: Fashion-MNIST (gambar grayscale 28×28, 10 kelas). Script memakai subset agar ringan.
- Hyperparameter search menggunakan Optuna dan nested runs pada MLflow.
- Metrik validasi dicatat per epoch; test set hanya dipakai pada prediksi contoh akhir (untuk menghindari pemilihan model berdasarkan test).
- Jika registry alias gagal karena versi MLflow lama, upgrade MLflow dengan `pip install -U mlflow`.

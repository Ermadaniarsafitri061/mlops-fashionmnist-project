# Laporan Praktikum MLOps — Fashion-MNIST

**Nama:** [isi]  
**NIM:** [isi]  
**Kelas:** [isi]

## 1. Hipotesis
- H1: CNN baseline mempelajari pola Fashion-MNIST dan mengungguli tebakan acak.
- H2: Learning rate, batch size, jumlah channel, dan dropout memengaruhi validation accuracy.
- H3: Augmentasi dan dropout memengaruhi generalisasi; efeknya diuji melalui ablation.
- H4: Tiga seed menunjukkan variasi performa yang tidak terlihat dari satu run saja.

## 2. Dataset dan lingkungan
- Dataset: Fashion-MNIST subset.
- Ukuran subset train yang digunakan: [isi aktual]
- Ukuran subset test: [isi aktual]
- Perangkat: [CPU/GPU]
- Versi Python/PyTorch/MLflow: [isi]
- Hash SHA-256: [salin dari parameter `dataset_sha256` di MLflow]

## 3. Hasil baseline dan tuning
- Baseline validation accuracy: [hasil aktual]
- Jumlah trial Optuna selesai: [minimal 20]
- Konfigurasi terbaik: [hasil aktual]
- Screenshot MLflow Compare view: [tempel]

## 4. Hasil tiga seed
| Seed | Validation accuracy |
|---|---:|
| 42 | [aktual] |
| 123 | [aktual] |
| 2026 | [aktual] |
| Mean ± sample std | [aktual] |

## 5. Ablation study
| Konfigurasi | Augmentasi | Dropout | Validation accuracy |
|---|---|---|---:|
| Full model | Ya | Ya | [aktual] |
| Tanpa augmentasi | Tidak | Ya | [aktual] |
| Tanpa dropout | Ya | Tidak | [aktual] |

## 6. Model Registry dan prediksi
Nama model: `fashion-mnist-cnn-classifier`  
Alias: `champion`  
Bukti versi model dan hasil prediksi: [tempel screenshot]

## 7. Kesimpulan
Tuliskan kesimpulan berdasarkan hasil yang benar-benar diperoleh. Jangan isi angka sebelum skrip selesai dijalankan.

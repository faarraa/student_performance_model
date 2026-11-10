# Student Performance Prediction (PyTorch)

Aplikasi Streamlit untuk memprediksi kategori performa rendah berdasarkan lima fitur kategorikal dari dataset Students Performance in Exams.

## Target
- `0`: Tidak Rendah
- `1`: Performa Rendah, jika rata-rata math, reading, dan writing score kurang dari 60.

## Fitur input
- gender
- race/ethnicity
- parental level of education
- lunch
- test preparation course

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training model `.pth`
Letakkan `StudentsPerformance.csv` di folder project, lalu jalankan:
```bash
python train_model.py
```
Script juga mencoba mengambil dataset dari sumber publik, lalu fallback ke KaggleHub jika pengambilan pertama gagal. Hasil training disimpan ke `model/student_performance_model.pth`.

## Menjalankan aplikasi
```bash
python -m streamlit run app.py
```

## Catatan
Model `.pth` dibuat dari ANN PyTorch. Jalankan training terlebih dahulu sebelum membuka aplikasi jika file model belum tersedia. Jangan mengganti file `.joblib` menjadi `.pth` hanya dengan rename; keduanya memiliki format yang berbeda.

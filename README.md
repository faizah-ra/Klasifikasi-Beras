# 🌾 Rice Variety Classification using Convolutional Neural Network

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Proyek ini bertujuan untuk mengklasifikasikan lima varietas beras menggunakan algoritma *Convolutional Neural Network (CNN)*. Sistem dibangun dalam kerangka klasifikasi citra, lengkap dengan preprocessing data, pelatihan model, evaluasi performa, dan ekspor model ke dalam tiga format: **SavedModel**, **TensorFlow Lite**, dan **TensorFlow.js** untuk mendukung deployment lintas platform.

📌 Proyek ini merupakan bagian dari submission **Belajar Fundamental Deep Learning** di Dicoding.  
🎖️ **Rating Submission: 4/5 (Bintang Empat)**  
📁 Submission ID: `4205562`  
📅 Tanggal Kirim: `4 Mei 2025`

---

## 🧑‍💼 Peran dan Tanggung Jawab System Analyst

Sebagai System Analyst dalam proyek ini, saya bertanggung jawab untuk:

- 📌 **Menganalisis kebutuhan sistem klasifikasi gambar** untuk otomasi identifikasi varietas beras.
- 🧩 **Menyusun arsitektur pemrosesan data dan pemodelan CNN** secara modular dan terdokumentasi.
- 📑 **Menyiapkan dokumentasi sistem** (struktur folder, proses pelatihan, evaluasi).
- 📊 **Menganalisis performa sistem dan memberikan rekomendasi pengembangan** berdasarkan hasil evaluasi dan umpan balik reviewer.

---

## 🎯 Tujuan Proyek

- Mengembangkan model CNN untuk mengklasifikasikan 5 varietas beras:
  - Arborio
  - Basmati
  - Ipsala
  - Jasmine
  - Karacadag
- Meningkatkan akurasi klasifikasi hingga mendekati 100%
- Mengekspor model untuk keperluan deployment web dan mobile

---

## 🗂 Dataset

Dataset yang digunakan adalah:
📦 [Rice Image Dataset – Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  
Total: 15.000 gambar (3.000 gambar per kelas)  
Lisensi: CC0 1.0 Public Domain

---

## ⚙️ Teknologi dan Tools

- **Bahasa Pemrograman**: Python
- **Framework DL**: TensorFlow, Keras
- **Preprocessing**: OpenCV, Keras ImageDataGenerator
- **Deployment Model**: SavedModel, TF-Lite, TFJS
- **Visualisasi**: Matplotlib, Seaborn

---

## 🧠 Arsitektur Model

- Model Sequential CNN
- Layer utama: Conv2D → MaxPooling → Dropout
- Optimizer: Adam
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Data augmentation: flip, rotate, zoom

---
---

## 🧾 Ringkasan Proses Sistem

| Langkah Sistem            | Deskripsi                                                             |
|---------------------------|----------------------------------------------------------------------|
| Data Ingestion            | Mengambil data gambar varietas beras dari Kaggle                    |
| Preprocessing             | Resize, augmentasi, normalisasi                                      |
| Modeling                  | CNN (Conv2D, MaxPooling, Dropout) dengan optimasi Adam              |
| Evaluasi                  | Akurasi, F1-score, confusion matrix, visualisasi loss/accuracy      |
| Deployment Preparation    | Ekspor ke SavedModel, TFLite, dan TFJS                               |
| Inference                 | Prediksi gambar baru melalui model terlatih                         |

---

## 📊 Evaluasi Performa

| Metrik                | Hasil        |
|-----------------------|--------------|
|  Akurasi data latih   | ~99.91%      |
| Akurasi Validasi      | ~99.47%      |
| F1-score per Kelas    | ≥ 99%        |
| Overfitting           | Minimal      |

Visualisasi dan metrik evaluasi tersedia dalam bentuk:
- Kurva akurasi & loss
- Confusion matrix
- Klasifikasi per kelas

---

## 📦 Output Model

- `saved_model/` — format asli TensorFlow
- `tflite/` — untuk perangkat mobile
- `tfjs_model/` — untuk deployment web

---

## 📂 Struktur Folder

```
submission/
├───tfjs_model/
│   ├───group1-shard1of1.bin
│   └───model.json
├───tflite/
│   ├───model.tflite
│   └───label.txt
├───saved_model/
│   ├───saved_model.pb
│   └───variables/
├───notebook.ipynb
├───README.md
└───requirements.txt
```

---

## 📝 Penggunaan

### Jalankan Notebook
```bash
jupyter notebook notebook.ipynb
```

### Ekspor Model (dari notebook)
```python
# SavedModel
model.save('saved_model')

# TensorFlow Lite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()
with open('tflite/model.tflite', 'wb') as f: f.write(tflite_model)

# Label
labels = list(train_generator.class_indices.keys())
with open('tflite/label.txt', 'w') as f: f.write('\n'.join(labels))

# TensorFlow.js
!tensorflowjs_converter --input_format=keras saved_model tfjs_model
```

### Inference Gambar Baru
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('saved_model')
img = image.load_img('path/to/img.jpg', target_size=(250,250))
arr = image.img_to_array(img)/255.0
pred = model.predict(np.expand_dims(arr, 0))
idx = np.argmax(pred[0])
class_names = list(train_generator.class_indices.keys())
print(f"Prediksi: {class_names[idx]} (confidence {pred[0][idx]:.4f})")
```

---

## 📊 Hasil Pelatihan

**Best Validation Accuracy:** 99.47%  
**Test Set Performance:**
| Kelas      | Precision | Recall |
|------------|-----------|--------|
| Arborio    | 0.99      | 0.99   |
| Basmati    | 0.99      | 0.99   |
| Ipsala     | 1.00      | 1.00   |
| Jasmine    | 0.99      | 0.99   |
| Karacadag  | 0.99      | 0.99   |

---

## 📌 Insight dan Rekomendasi
  - Tambahkan preprocessing untuk penyesuaian ukuran gambar
  - Validasi dengan data dari dunia nyata
  - Lakukan kompresi model agar cocok untuk device edge
  - Implementasi inferensi realtime berbasis web/mobile

---

## 👩‍💻 Tentang Penulis
Faizah Rizki Auliawati
📍 Mahasiswa Informatika, Machine Learning & System Analysis Enthusiast
🎓 Dicoding Certified — Belajar Fundamental Deep Learning
📬 frauliawati@gmail.com
🔗 [GitHub](https://github.com/faizah-ra)

---

## 📄 Lisensi

Proyek ini berlisensi MIT. Lihat `LICENSE` untuk detail.

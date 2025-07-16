
# 🌾 Klasifikasi Varietas Beras

Repositori ini berisi pipeline lengkap untuk mengklasifikasikan lima varietas beras menggunakan Convolutional Neural Network (CNN). Proyek ini mencakup langkah-langkah dari persiapan data, pelatihan model, evaluasi, hingga ekspor model dalam berbagai format: **SavedModel**, **TensorFlow Lite**, dan **TensorFlow.js** untuk deployment multiplatform.

---

## 📋 Ikhtisar Proyek

### 🎯 Tujuan
Membangun model CNN untuk mengklasifikasikan gambar beras ke dalam lima varietas:
- Arborio
- Basmati
- Jasmine
- Ipsala
- Karacadag

### 📦 Dataset
Dataset terdiri dari **15.000 gambar** (3.000 per kelas) dari:
> [Rice Image Dataset – Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)  
(Lisensi: CC0 1.0 Public Domain)

### 🧠 Metode
- Arsitektur: CNN dengan Conv2D, MaxPooling, dan Dropout
- Optimasi: Adam + ReduceLROnPlateau
- Evaluasi: Akurasi, confusion matrix, classification report

### 📈 Performa Model
- Akurasi pelatihan: ~99.91%
- Akurasi validasi: ~99.47%
- F1-score test set: ~99% di semua kelas
- Overfitting minimal: training loss 0.0035 vs validation loss 0.0174

---

## ⚙️ Fitur Utama

### ✅ Persiapan & Augmentasi Data
- Sampling acak: 3.000 gambar/kelas
- Split: Train/Validation/Test (80/10/10)
- Augmentasi real-time: rotasi, zoom, flip horizontal

### 🧱 Arsitektur Model
- CNN berlapis (Conv2D → MaxPooling → Dropout)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Penyeimbangan kelas otomatis dengan `class_weight`

### 📊 Evaluasi & Visualisasi
- Visualisasi kurva akurasi dan loss
- Confusion matrix dan classification report
- Contoh prediksi benar & salah

### 📤 Ekspor Model
- ✅ SavedModel (`saved_model/`)
- ✅ TensorFlow Lite (`tflite/`)
- ✅ TensorFlow.js (`tfjs_model/`)

---

## 🚀 Cara Memulai

### 🧰 Prasyarat
- Python 3.7+
- pip

### 🛠 Instalasi
```bash
git clone https://github.com/username/rice-classification.git
cd rice-classification/submission
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
```

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

## 💡 Saran Pengembangan
- Validasi dengan data baru dari dunia nyata
- Analisis kesalahan prediksi (30–50 kasus)
- Kompresi model (TFLite quantization)
- Cross-validation (k=5)
- Monitoring performa di lingkungan produksi
- Integrasi feedback pengguna untuk retraining

---

## 👤 Penulis

**Faizah Rizki Auliawati**  
📧 frauliawati@gmail.com  
🆔 Dicoding ID: MC009D5X2457

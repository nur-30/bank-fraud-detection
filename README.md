# Kasus A: Deteksi Penipuan Pembukaan Rekening Bank (BAF – varian Base)

**Hasil utama (data validasi, month 3): TPR = 58,3% pada FPR 5%** (AUC 0,914). Artinya, dengan hanya ≤ 5% nasabah jujur yang tertandai, model menangkap ±58% penipu.

## Isi repo
`notebook.ipynb` (jalankan atas ke bawah) · `fraud_model.joblib` (model tersimpan) · `fraud_utils.py` (fungsi bantu) · `requirements.txt`
Cara menjalankan: `pip install -r requirements.txt`, taruh `Base.csv` (Kaggle, varian Base) di folder yang sama, lalu *Run All*. Data tidak diunggah karena ±213 MB.

## Metode
Data: baris dengan `month` 0–3 (548.000 baris, rasio penipuan ≈ 1%). Nilai `-1` pada 5 kolom (mis. `prev_address_months_count`) berarti "tidak ada data", diubah menjadi `NaN` plus kolom indikator. Kategorik dipakai sebagai tipe `category`. `month` dan `device_fraud_count` (konstan) dibuang.
Model: **LightGBM** (gradient boosting, 441 pohon pada pemilihan awal, ×1,15 saat latih final), dibandingkan dengan baseline Regresi Logistik berbobot kelas.

## Alasan memilih LightGBM
Data tabular dengan campuran numerik/kategorik, banyak nilai hilang, interaksi antar fitur, dan kelas sangat tidak seimbang. Boosting menangani NaN secara native dan menangkap interaksi nonlinier tanpa rekayasa fitur panjang. Hasilnya unggul atas baseline di semua metrik (di bawah). Cukup ringan: model 0,8 MB, dilatih dalam beberapa menit di CPU.

## Cara membagi data
Pembagian **berdasarkan waktu**, bukan acak, karena di produksi model dilatih dari masa lalu dan dipakai pada pengajuan mendatang; pembagian acak membuat skor terlalu optimis.
- Month 0–1 latih, month 2 validasi-dalam (early stopping & pemilihan jumlah pohon)
- **Latih final: month 0–2. Validasi yang dilaporkan: month 3** (tidak dipakai untuk memilih apa pun kecuali menghitung ambang metrik)

## Skor di data validasi (month 3, 150.936 baris, 1.392 penipu)
| Model | AUC | PR-AUC | **TPR @ FPR 5%** |
|---|---|---|---|
| Regresi Logistik | 0,902 | 0,135 | 53,7% |
| **LightGBM** | **0,914** | **0,157** | **58,3%** |

## Catatan jujur
- Ambang untuk FPR 5% dihitung pada data validasi (definisi metrik TPR@FPR 5%). Uji stabilitas: ambang yang ditetapkan dari month 2 ternyata menghasilkan FPR ≈ 8,9% di month 3 (distribusi skor bergeser antar bulan). Jadi ambang perlu dikalibrasi ulang berkala di produksi.
- Satu split waktu saja; belum ada interval kepercayaan. Jumlah penipu di validasi 1.392, sehingga TPR wajar berfluktuasi beberapa poin.
- Fitur sensitif (`customer_age`, `income`) ikut dipakai; sebelum produksi perlu telaah keadilan/regulasi.

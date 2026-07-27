# Setup Working Directory
getwd()
setwd("C:/Project Data Analyst/Project 2 APS Failure at Scania Trucks")

# =====================================================================
# PROJECT: PREDICTIVE MAINTENANCE (APS FAILURE AT SCANIA TRUCKS)
# OBJECTIVE: MINIMIZE MAINTENANCE BUSINESS COST VIA THRESHOLD OPTIMIZATION
# =====================================================================

# 1. IMPORT LIBRARY
# Memuat pustaka yang dibutuhkan untuk manipulasi data dan pemodelan
library(dplyr)
library(readr)
library(randomForest)

# =====================================================================
# 2. DATA INGESTION & INITIAL EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

# Load dataset dan konversi string "na" menjadi format standard NA
scania_train <- read_csv("aps_failure_training_set.csv", na = c("na", "NA", ""))

# Verifikasi dimensi data (Ekspektasi: 60000 baris, 171 kolom)
dim(scania_train)

# Analisis distribusi kelas target (Pengecekan Imbalanced Data)
table(scania_train$class)

# =====================================================================
# 3. DATA PRE-PROCESSING: HANDLING MISSING VALUES
# =====================================================================

# Kalkulasi total missing values (NA) pada level dataset
sum(is.na(scania_train))

# Kalkulasi persentase missing values per fitur/kolom
persentase_na <- colMeans(is.na(scania_train)) * 100

# Identifikasi fitur dengan tingkat missing values > 50%
kolom_rusak_parah <- sort(persentase_na[persentase_na > 50], decreasing = TRUE)

# Evaluasi fitur yang tidak memenuhi standar kualitas data
print("Daftar fitur dengan persentase missing data > 50%:")
print(kolom_rusak_parah)
print(paste("Total fitur yang akan didrop:", length(kolom_rusak_parah)))

# Ekstraksi nama fitur untuk tahap eksklusi
kolom_dibuang <- names(kolom_rusak_parah)

# Drop fitur dengan missing values tinggi menggunakan dplyr
scania_clean <- scania_train %>% select(-all_of(kolom_dibuang))

# Verifikasi dimensi pasca-penghapusan fitur (Ekspektasi kolom berkurang)
dim(scania_clean)

# Imputasi sisa missing values pada fitur numerik menggunakan nilai Median
# Nilai median dipilih karena lebih robust terhadap pencilan (outliers)
scania_imputed <- scania_clean %>% 
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Verifikasi akhir validitas imputasi data
sum(is.na(scania_imputed))

# =====================================================================
# 4. DATA PREPARATION FOR MODELING
# =====================================================================

# Konversi variabel target menjadi tipe factor untuk pemodelan klasifikasi
scania_imputed$class <- as.factor(scania_imputed$class)

# Penetapan seed untuk memastikan reproduktibilitas hasil (reproducibility)
set.seed(123)

# Splitting dataset: 80% Data Latih (Train) dan 20% Data Uji (Test)
index_belajar <- sample(1:nrow(scania_imputed), 0.8 * nrow(scania_imputed))

data_train <- scania_imputed[index_belajar, ]
data_test <- scania_imputed[-index_belajar, ]

# Verifikasi dimensi pemisahan data
dim(data_train)
dim(data_test)

# =====================================================================
# 5. BASELINE MODEL: LOGISTIC REGRESSION
# =====================================================================

# Training model regresi logistik sebagai baseline klasifikasi awal
model_logreg <- glm(class ~ ., data = data_train, family = "binomial")

# Prediksi probabilitas kelas pada data uji
prediksi_prob <- predict(model_logreg, newdata = data_test, type = "response")

# Klasifikasi variabel target menggunakan decision threshold default (0.5)
tebakan_kelas <- ifelse(prediksi_prob > 0.5, "pos", "neg")

# Evaluasi performa klasifikasi model baseline (Confusion Matrix)
print("--- EVALUASI PERFORMA: LOGISTIC REGRESSION (BASELINE) ---")
table(Tebakan = tebakan_kelas, Asli = data_test$class)

# =====================================================================
# 6. ADVANCED MODELING: RANDOM FOREST
# =====================================================================

# Training model Random Forest dengan hyperparameter standard
model_rf <- randomForest(class ~ ., data = data_train, ntree = 100, importance = TRUE)

# Prediksi dan evaluasi kelas pada data uji
tebakan_rf <- predict(model_rf, newdata = data_test)
print("--- EVALUASI PERFORMA: RANDOM FOREST STANDARD ---")
print(table(Tebakan = tebakan_rf, Asli = data_test$class))

# Analisis Feature Importance (Identifikasi Top 10 Sensor Kritis)
varImpPlot(model_rf, n.var = 10, main = "Top 10 Feature Importance (Scania APS Sensors)")

# =====================================================================
# 7. PENANGANAN IMBALANCED DATA: CLASS WEIGHTING
# =====================================================================

# Training ulang model dengan memberikan penalti (class weight)
# Kelas minoritas ('pos') diberi bobot 50x lebih signifikan dibanding kelas mayoritas
model_rf_berat <- randomForest(class ~ ., data = data_train, 
                               ntree = 100, 
                               classwt = c(neg = 1, pos = 50), 
                               importance = TRUE)

# Prediksi dan evaluasi model dengan penyesuaian bobot kelas
tebakan_rf_berat <- predict(model_rf_berat, newdata = data_test)
print("--- EVALUASI PERFORMA: RANDOM FOREST (CLASS WEIGHT) ---")
table(Tebakan = tebakan_rf_berat, Asli = data_test$class)

# =====================================================================
# 8. BUSINESS COST METRIC EVALUATION
# Parameter Kerugian Finansial Spesifik Domain:
# - False Positive (Alarm Palsu) = Biaya inspeksi mekanik ($10)
# - False Negative (Kebobolan) = Biaya breakdown truk di lapangan ($500)
# =====================================================================

# Perhitungan agregasi kerugian untuk Model Standard
cost_model_1 <- (11 * 10) + (77 * 500)

# Perhitungan agregasi kerugian untuk Model Class Weight
cost_model_2 <- (13 * 10) + (119 * 500)

print("--- ANALISIS KERUGIAN FINANSIAL ---")
print(paste("Total Business Cost - Model Standard: $", cost_model_1))
print(paste("Total Business Cost - Model Class Weight: $", cost_model_2))
# Insight Analisis: Implementasi class weighting mendegradasi keseimbangan model
# dan menaikkan potensi kerugian finansial perusahaan.

# =====================================================================
# 9. OPTIMASI RISIKO: THRESHOLD ADJUSTMENT (CUTOFF)
# =====================================================================

# Modifikasi decision threshold untuk memprioritaskan deteksi kelas 'pos'
# Penyesuaian batas klasifikasi: Prediksi 'pos' aktif pada probabilitas minimal 5%
set.seed(123)
model_rf_cutoff <- randomForest(class ~ ., data = data_train, 
                                ntree = 100, 
                                cutoff = c(0.95, 0.05),
                                importance = TRUE)

# Prediksi dan evaluasi menggunakan model dengan penyesuaian threshold
tebakan_rf_cutoff <- predict(model_rf_cutoff, newdata = data_test)

print("--- EVALUASI PERFORMA: RANDOM FOREST (CUTOFF 5%) ---")
table(Tebakan = tebakan_rf_cutoff, Asli = data_test$class)

# Ekstraksi komponen Confusion Matrix untuk kalkulasi Business Cost
cm_3 <- table(Tebakan = tebakan_rf_cutoff, Asli = data_test$class)
fp_3 <- cm_3["pos", "neg"]
fn_3 <- cm_3["neg", "pos"]

# Kalkulasi performa finansial solusi threshold awal
cost_model_3 <- (fp_3 * 10) + (fn_3 * 500)
print(paste("Total Business Cost - Model Cutoff 5% (Initial Solution): $", cost_model_3))

# =====================================================================
# 10. THRESHOLD TUNING (GLOBAL MINIMUM COST OPTIMIZATION)
# =====================================================================

# Inisialisasi parameter iterasi threshold probabilitas (1% hingga 10%)
cutoff_range <- seq(0.01, 0.10, by = 0.01)
biaya_list <- numeric(length(cutoff_range))

# Iterasi komputasi Business Cost pada berbagai variasi titik cutoff
for(i in 1:length(cutoff_range)) {
  c_pos <- cutoff_range[i]
  c_neg <- 1 - c_pos
  
  # Prediksi kelas berdasarkan threshold iteratif
  pred_temp <- predict(model_rf_cutoff, newdata = data_test, cutoff = c(c_neg, c_pos))
  cm_temp <- table(Tebakan = pred_temp, Asli = data_test$class)
  
  # Ekstraksi dinamis untuk metrik False Positive dan False Negative
  fp_temp <- ifelse("pos" %in% rownames(cm_temp) & "neg" %in% colnames(cm_temp), cm_temp["pos", "neg"], 0)
  fn_temp <- ifelse("neg" %in% rownames(cm_temp) & "pos" %in% colnames(cm_temp), cm_temp["neg", "pos"], 0)
  
  # Kalkulasi nilai fungsi kerugian
  biaya_list[i] <- (fp_temp * 10) + (fn_temp * 500)
}

# Kompilasi hasil evaluasi optimasi
hasil_tuning <- data.frame(Cutoff_Probabilitas = cutoff_range, Total_Kerugian = biaya_list)
print("--- HASIL OPTIMASI THRESHOLD (MINIMUM BUSINESS COST) ---")
print(hasil_tuning)

# Visualisasi kurva optimasi fungsi biaya
plot(hasil_tuning$Cutoff_Probabilitas, hasil_tuning$Total_Kerugian, type = "b", 
     col = "darkred", lwd = 2, pch = 19,
     xlab = "Persentase Cutoff (Probabilitas Kelas Positif)", 
     ylab = "Total Kerugian Bisnis / Business Cost ($)",
     main = "Kurva Optimasi Biaya Bisnis (Pencarian Global Minimum)")
grid()

# =====================================================================
# 11. DATA EXPORT UNTUK PIPELINE REPORTING
# =====================================================================

# Menyimpan luaran (output) akhir untuk proses integrasi visualisasi (Dashboarding)
hasil_akhir <- data.frame(
  Aktual = data_test$class,
  Prediksi_Model = tebakan_rf_cutoff
)
write_csv(hasil_akhir, "Hasil_Prediksi_Scania_Final.csv")

print("Analisis Selesai! Data prediksi berhasil diekspor ke CSV.")
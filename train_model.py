import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Membaca dataset
df = pd.read_csv('iris.csv')

# 2. Memisahkan Fitur (X) dan Label (y)
# Sesuai filemu, kolom target bernama 'species'
X = df.drop('species', axis=1) 
y = df['species']              

# 3. Membagi Data (80% Latih, 20% Uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Membuat dan Melatih Model KNN 
# Modul tugas biasanya menggunakan k ganjil, kita pakai k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 5. Mengevaluasi Akurasi Model
y_pred = knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi Model KNN: {akurasi * 100:.2f}%")

# 6. Menyimpan Model agar bisa digunakan untuk Web Flask nanti
with open('model_knn.pkl', 'wb') as file:
    pickle.dump(knn, file)
print("Model berhasil disimpan sebagai 'model_knn.pkl'")
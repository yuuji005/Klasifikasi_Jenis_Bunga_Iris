from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Memuat model KNN yang sudah disimpan
with open('model_knn.pkl', 'rb') as file:
    model = pickle.load(file)

# Dictionary untuk menyimpan data gambar dan fakta menarik tiap bunga
INFO_BUNGA = {
    'setosa': {
        'nama_lengkap': 'Iris Setosa',
        'gambar': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg',
        'fakta': 'Iris setosa sering dijuluki "Bristle-pointed iris" karena kelopak bagian dalamnya mengecil hingga menyerupai duri/rambut kaku. Spesies ini sangat tangguh dan mampu bertahan hidup di iklim dingin ekstrem seperti wilayah Artik.'
    },
    'versicolor': {
        'nama_lengkap': 'Iris Versicolor',
        'gambar': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
        'fakta': 'Dikenal dengan nama "Blue Flag", Iris versicolor memiliki kombinasi warna ungu-biru yang sangat cantik. Menariknya, bunga ini merupakan lambang resmi dari Provinsi Quebec di Kanada dan sering tumbuh subur di area lahan basah atau rawa.'
    },
    'virginica': {
        'nama_lengkap': 'Iris Virginica',
        'gambar': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg',
        'fakta': 'Dikenal sebagai "Virginia Iris", spesies ini memiliki ukuran kelopak dan mahkota yang paling besar dan kokoh di antara ketiga jenis Iris. Bunga ini sangat menyukai habitat air tawar dan sering ditemukan di sepanjang pantai timur Amerika Serikat.'
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    detail_bunga = None
    
    if request.method == 'POST':
        try:
            # Ambil data dari slider form web
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Masukkan ke array numpy 2D untuk prediksi
            data_baru = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Prediksi menggunakan model KNN
            hasil_prediksi = model.predict(data_baru)
            prediction = hasil_prediksi[0].lower() # Memastikan teks berhuruf kecil agar cocok dengan key dict
            
            # Ambil detail gambar dan fakta berdasarkan hasil prediksi
            detail_bunga = INFO_BUNGA.get(prediction)
            
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, detail=detail_bunga)

app = app
if __name__ == '__main__':
    app.run(debug=True)
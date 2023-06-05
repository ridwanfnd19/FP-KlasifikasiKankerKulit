'''
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 4
	Tim Deployment
	2023
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
# from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
Flatten, Dense, Activation, Dropout,LeakyReLU
from PIL import Image
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 4
classes = ["Actnic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis", "Dermatofibroma", "Melanoma", "Melanoctic Nevi", "Vascular"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename) # type: ignore
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = cv2.imread('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			image_fromarray = Image.fromarray(test_image, 'RGB')
			test_image_resized = image_fromarray.resize((100, 100))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255)
			test_image_x       = np.array([image_array])
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = classes[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model()
	model.load_weights("model.h5")

	# Run Flask di Google Colab menggunakan ngrok
	# run_with_ngrok(app)
	app.run()
	
	


# Cargamos las librerias necesarias
import os
import cv2
from flask import render_template, request
import matplotlib.image as matimg
from app.face_recognized import faceRecognitionPipeline

# ruta de carpeta donde se almacenan los imagenes cargadas
UPLOAD_FOLDER = 'static/upload'
PREDICT_FOLDER = 'static/predict'
PREDICT_IMAGE_FILE_NAME = 'prediction_image.jpg'

# funciones para llamar a las vistas html
def index():
    return render_template('')
def app():
    return render_template('')

# logica del programa de reconocimiento de genero
def genderapp():
    if request.method=='POST':
        f=request.files['image_name']
        filename=f.filename
        
        # almacenar la imagen
        path=os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        
        # Obtener la predicción
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename=PREDICT_IMAGE_FILE_NAME
        cv2.imwrite(f'./{PREDICT_FOLDER}/{pred_filename}', pred_image)
        
        # Generar el report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi']
            eigen_image = obj['eig_img'].reshape(100, 100)
            gender_name = obj['predicition_name']
            score = round(obj['score']*100,2)
            # colocar nombre y guardar en archivo en la carpeta predict
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./{PREDICT_FOLDER}/{gray_image_name}',
                           gray_image, 
                           cmap='gray')
            matimg.imsave(f'./{PREDICT_FOLDER}/{eig_image_name}',
                           eigen_image, 
                           cmap='gray')
            # guardar el reporte
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
            # petición POST que genera los resultados de la imagen
            return render_template('',
                                   fileupload=True,report=report)
        #petición GET que muestra la vista para subir la imagen
        return render_template('',
                               fileupload=False)
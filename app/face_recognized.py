# Cargamos las librerias necesarias
import numpy as np
import sklearn
import pickle
import cv2

# Colores que se utilizan
MALE_BORDER = (255,255,0)
WOMEN_BORDER = (255,0,255)
BACKGROUND_FONT = (255,255,255)

# Colocamos el modelo cascada y nuestros modelos
model_internet = cv2.CascadeClassifier('') # reconocimiento frontal Internet
model_svm = pickle.load(open('', mode='rb')) #modelo de aprendizaje de maquina
dicc_pca = pickle.load(open('', mode='rb')) # Dicccionario
model_pca = dicc_pca['pca'] # modelo pca
mean_face_arr = dicc_pca['mean_face']  # media de las caras

def faceRecognitionPipeline(filename, path=True):
    # Primer paso leer la imagen
    if path:
        img = cv2.imread(filename)
    else:
        img = filename
    # Segundo paso convertir la imagen BGR a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Tercer paso cortar solo las caras de la imagen
    faces=model_internet.detectMultiScale(gray,1.5,3)
    predictions=[]
    for x,y,w,h in faces:
        roi=gray[y:y+h,x:x+w]
    # Cuarto paso normalizamos para no tener valores fuera de rango
        roi = roi / 255
    # Quinto paso cambiamos el tamaño de las imagenes (100;100)
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),
                                    cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100,100),
                                    cv2.INTER_CUBIC)
    # Sexto paso convertir una imagen 2D a 1D
        roi_reshape = roi_resize.reshape(1, 10000)
    # Septimo paso obtenemos la media de la nueva cara con la media de nuestro modelo
        roi_mean = roi_reshape - mean_face_arr
    # Octavo paso obtener los valores propios de la imagen
        eigen_image = model_pca.transform(roi_mean)
    # Noveno paso visualización de los valores propios
        eig_img = model_pca.inverse_transform(eigen_image)
    # Decimo paso obtenemos la predicción
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
    # Decimo primer paso generamos el reporte
        text= "%s : %d"%(results[0], prob_score_max*100)
        # Definimos el color base del resultado
        if results[0] == "masculino":
            color = MALE_BORDER
        else:
            color = WOMEN_BORDER

        cv2.rectangle(img, (x,y),(x+w,y+h),color,2)
        cv2.rectangle(img, (x,y-40),(x+w,y),color,-1)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN,3,BACKGROUND_FONT,5)
        output = {
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name':results[0],
            'score': prob_score_max
        }
        predictions.append(output)
    return img, predictions

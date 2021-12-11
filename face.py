from typing import OrderedDict
import numpy as np
import cv2
import os
import pickle

# Caminho onde esse arquivo .py esta salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cascade_path = BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)
print(face_cascade)
cap = cv2.VideoCapture(0)
#Modelo de reconhecimento criado em face_train.py
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")

#Importar ids
labels = {}
with open("labels.pickle","rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()} #Inverter o dicionario

while(True):
    # Captura frames por frame
    ret, frame = cap.read()

    # Converte os frames em cinza para a detecção com cascade
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detecta faces no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    print(faces)
    for (x, y , w , h) in faces:
        # Regiao de interesse (face)
        face_detect_gray = gray[y:y+h,x:x+w]
        face_detect_color = frame[y:y+h,x:x+w]

        # Reconhecimento de face
        id_, conf = recognizer.predict(face_detect_gray)
        if conf >= 90:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255) #Branco
            stroke = 2
            cv2.putText(frame,name, (x,y), font, 1, color, stroke, cv2.LINE_AA) #Adicionar o nome da pessoa
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = ""
            color = (255,255,255) #Branco
            stroke = 2
            cv2.putText(frame,name, (x,y), font, 1, color, stroke, cv2.LINE_AA) #Desconhecido

        img_item = "last_person.png"
        cv2.imwrite(img_item,frame)


        # Retangulo em volta do rosto
        color_frame = (255, 0, 0) #Azul
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color_frame, stroke)

    # Mostra a camera
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


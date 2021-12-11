import os
from PIL import Image
import numpy as np
import cv2
import pickle

# Caminho onde esse arquivo .py esta salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

cascade_path = BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

# Encontra imagens jpeg e png no diretorio images
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpeg") or file.endswith("png"):
            # Caminho da imagem
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # Criando um id para cada label das imagens
            if label not in label_ids:  
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            pil_image = Image.open(path).convert("L") # Abrindo a imagem e convertendo em cinza
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") # Transformando em um numpy array
            # Detecta faces no frame
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) 
            print(faces)
            for (x,y,w,h) in faces:
                face_detect = image_array[y:y+h, x:x+w]
                #Adicionando as faces das pessoas escolhidas para treino
                x_train.append(face_detect)
                y_labels.append(id_)

# Criar um arquivo com os ids das pessoas
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids, f)

#Treinar o modelo de reconhecimento facial
recognizer.train(x_train, np.array(y_labels))
#Salvar em um arquivo yml
recognizer.save("train.yml")
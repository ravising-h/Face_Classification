import cv2
import numpy as np
import os
from PIL import Image
from keras_vggface.vggface import VGGFace
import sys


path_of_haar_caascade = 'haarcascade_frontalface_default.xml'
path_of_faces = "faces/"


face_cascade = cv2.CascadeClassifier(path_of_haar_caascade)
vgg_features = VGGFace(model = 'resnet50',include_top=False, input_shape=(224, 224, 3), pooling='avg')

def face_feature_extrac(croped):
    feature = np.zeros((len(croped),2048))
    index = 0
    for  face in range(croped.shape[0]):
        feature[index] = vgg_features.predict(cv2.resize(croped[face],(224,224)).reshape(1,224,224,3))
        index += 1
    return np.array(feature)

def detect_face(img_name,return_face_feature = True):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3)
    if (len(faces) == 0):
        print("No Faces Found!")
        sys.exit()
    frame_crop = list()
    for face in faces:
        (x, y, w, h) = face
        frame_crop.append(img[y:y+w, x:x+h])
    if return_face_feature is True:
        return face_feature_extrac(np.array(frame_crop))
    else:
        return np.array(frame_crop)

def making_dataset(path_of_faces):
    face_name = np.array(os.listdir(path_of_faces))
    face_feature = np.zeros((face_name.shape[0],2048))
    index = 0
    for face in face_name:
        face_feature[index] = detect_face(path_of_faces + face)
        index += 1
    np.save("face_name.npy", face_name)
    np.save("face_feature.npy",face_feature)
making_dataset(path_of_faces)
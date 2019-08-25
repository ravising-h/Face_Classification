import cv2
import numpy as np
import os
from PIL import Image
from keras_vggface.vggface import VGGFace
from scipy import spatial

path_of_haar_caascade = 'haarcascade_frontalface_default.xml'
path_of_faces = "faces/"
path_of_test =  "test/"
path_of_result  = "result/pick.txt"
path_of_face_name = "face_name.npy"
path_of_face_feature = "face_feature.npy"


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
        with open("result/pick.txt","w") as f:
            f.write("No Face In the Frame")
        #exit(1)
    frame_crop = list()
    for face in faces:
        (x, y, w, h) = face
        frame_crop.append(img[y:y+w, x:x+h])
    if return_face_feature is True:
        return face_feature_extrac(np.array(frame_crop))
    else:
        return np.array(frame_crop)


def face_compare(face_features,test_feature):
    result = list()
    for face_feature in face_features:
        result.append(1 - spatial.distance.cosine(test_feature, face_feature))
    return result

def test(test_image,path_of_test,face_name, face_feature):
    test_feature = detect_face(path_of_test + test_image)
    no_of_faces = test_feature.shape[0]
    index, comparision = 0, list()
    for test in range(no_of_faces):
        comparision.append(face_compare(face_feature,test_feature[test]))
    return comparision


def faces_classification(comparision,face_name):
    faces = ["None"] * len(comparision)
    ind = 0
    for face in comparision:
        max_ = 0
        index = 0

        for known_face in face:

            if known_face >0.75 and  known_face > max_:
                max_ = known_face
                faces[ind] = face_name[index][0:-4]
            index += 1
        ind += 1
    return faces


def face_recognition(test_image = os.listdir(path_of_test)[-1]):
    
    face_name = np.load(path_of_face_name)
    face_feature = np.load(path_of_face_feature)
    comparision = test(test_image,path_of_test,face_name,face_feature)
    result = faces_classification(comparision,face_name)
    print(result)
    if result == "None": 
        with open(path_of_result,"w") as f:
            f.write("recognized no one")
    else:
        with open(path_of_result,"w") as f:
            f.write("None")
            for i in result:
                if i != "None":

                    f.write(i)


                    
face_recognition()






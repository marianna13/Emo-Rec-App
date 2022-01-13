import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2


def extract_face(image, size=(48,48)):
    array_image = np.array(image)
    gray = cv2.cvtColor(array_image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for (x, y, w, h) in faces:
        image = image.crop((x, y, x+w, y+h)).convert('L')
    
    return image.resize(size)

def predict(image):
    np_image = img_to_array(image)
    np_image = np.expand_dims(np_image, axis=0)
    my_model = load_model("model\model.h5")
    y_prob = my_model.predict(np_image)
    classes = np.argmax(y_prob,axis=1)
    labels = ['sad', 'happy', 'neutral']
    pred = labels[classes[0]]
    return pred


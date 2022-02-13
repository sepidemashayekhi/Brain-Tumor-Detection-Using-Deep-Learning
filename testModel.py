import cv2
from keras.models import load_model
from tensorflow.keras.utils import normalize
import numpy as np 

def Preprocess(image):
    image=cv2.resize(image,(200,200))
    image=np.reshape(image,(1,200,200,3))
    image=normalize(image,axis=1) 
    print(" Finish process")  
    return image

model=load_model('my_model .h5')
print('Model load ....')

imgpath='G:\my_project\Brain-Tumor-Detection-Using-Deep-Learning\pred\pred0.jpg'
image=cv2.imread('G:\my_project\Brain-Tumor-Detection-Using-Deep-Learning\pred\pred0.jpg')
image=Preprocess(image)


result=model.predict(image)

print(result)

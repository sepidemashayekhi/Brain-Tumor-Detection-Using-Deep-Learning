import streamlit as st
from TumorDetection import TumorDetection
import numpy as np
from tensorflow.keras.utils import  normalize
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image
from keras.models import load_model

def Preprocess(image):
    image=cv2.resize(image,(200,200))
    image=np.reshape(image,(1,200,200,3))
    image=normalize(image,axis=1)
    print(" Finish process")
    return image
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.markdown(" # Hello dear user ")
st.text("Enter MRI image of your brain ")
image_file=st.file_uploader("image ",type=["png","jpg","jpeg"])
model=load_model('my_model .h5')

if image_file:
    image = load_image(image_file)
    st.image(image)
    image_array= img_to_array(image)
    image_pre=Preprocess(image_array)

run=st.button("Process")
if run:
    result = model.predict(image_pre)
    if result>=0.5:
        st.text("Unfortunately you have a brain tumor")
        TumorD=TumorDetection()
        imagePre=TumorD.preprocess(image_array)
        meanStd=cv2.meanStdDev(imagePre)
        Thresh=meanStd[0][0]+meanStd[1][0]
        TumorD.cannyThreshold(image_array)
        thresh=TumorD.detection(imagePre,Thresh[0])
        st.image(image_array)
    
    elif result<0.5:
        st.text("Fortunately your brain is healthy")
    
    
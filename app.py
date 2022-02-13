import streamlit as st
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
    image = img_to_array(image)
    image=Preprocess(image)

run=st.button("Process")
if run:
    result = model.predict(image)
    if result>=0.5:
        st.text("Unfortunately you have a brain tumor")
    elif result<0.5:
        st.text("Fortunately your brain is healthy")










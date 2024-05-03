import streamlit as st
from keras.preprocessing import image
import numpy as np
import os
import cv2
from skimage.feature import local_binary_pattern
from keras.models import load_model
import joblib
from PIL import Image

# Streamlit UI
st.title('Rice Leaf Disease Prediction')

# Load the pre-trained Keras model
model_path = 'rice_leaf_model_pretrained.keras'
keras_model = load_model(model_path)

# Load the SVM model trained on LBP features
svm_model_lbp = joblib.load("svm_rice_leaf_model_lbp.joblib")

# Function for predicting with Keras model
def predict_with_keras(image_data):
    target_size=(224, 224)
    img = image.array_to_img(image_data)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predictions = keras_model.predict(img_array)
    return predictions

# Function for predicting with SVM model using LBP features
def predict_with_svmlbp(image_data):
    category_order = ['tungro', 'blast', 'brownspot', 'sheathblight', 'bacterialblight']
    img_gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    lbp_features = local_binary_pattern(img_gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 10), range=(0, 9))
    prediction_index = svm_model_lbp.predict(hist.reshape(1, -1))[0]
    prediction = category_order[prediction_index]
    return prediction



# Model selection
model_option = st.selectbox('Select Model', ['Keras Model', 'SVM Model (LBP)'])

# Image upload
uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Prediction
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_data = np.array(img)

    if st.button('Predict'):
        if model_option == 'Keras Model':
            prediction = predict_with_keras(img_data)
        elif model_option == 'SVM Model (LBP)':
            prediction = predict_with_svmlbp(img_data)

        st.write('Prediction:', prediction)

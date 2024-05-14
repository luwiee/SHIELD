import streamlit as st
from keras.preprocessing import image
import numpy as np
import os
import cv2
from skimage.feature import local_binary_pattern
import joblib
from PIL import Image

import tensorflow as tf

# Streamlit UI
st.title('Rice Leaf Disease Prediction')

# Load the SVM model trained on LBP features
svm_model_lbp = joblib.load("svm_rice_leaf_model_lbp.joblib")
# Load the pre-trained Keras model
model_path = 'rice_leaf_model_pretrained.h5'
keras_model = tf.keras.models.load_model(model_path)


category_order = ['tungro', 'blast', 'brownspot', 'sheathblight', 'bacterialblight']
category_order_cnn = ['brownspot', 'tungro', 'blast', 'bacterialblight', 'sheathblight']
# Function for predicting with Keras model
def predict_with_keras(image_data):
    target_size = (224, 224)
    
    # Ensure image has 3 channels
    if image_data.shape[2] == 4:
        image_data = image_data[:, :, :3]
        
    img = image.array_to_img(image_data)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predictions = keras_model.predict(img_array)[0]  # Get probabilities for each class
    sorted_indices = np.argsort(predictions)[::-1]  # Sort probabilities in descending order
    predictions_sorted = [(category_order_cnn[i], predictions[i]*100) for i in sorted_indices]  # Convert to category labels and percentages
    return predictions_sorted

# Function for predicting with SVM model using LBP features
def predict_with_svmlbp(image_data):
    img_gray = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    lbp_features = local_binary_pattern(img_gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 10), range=(0, 9))
    prediction_index = svm_model_lbp.predict(hist.reshape(1, -1))[0]
    prediction = category_order[prediction_index]
    return prediction

# Function to set selected image as uploaded_image
def set_uploaded_image(image_path):
    img = Image.open(image_path)
    st.session_state.uploaded_image = np.array(img)

# Model selection
model_option = st.selectbox('Select Model', ['Keras Model Pre-trained', 'SVM Model (LBP)'])

# Initialize uploaded_image in session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
    
# Image upload
uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Gallery
gallery_folder = 'validation'  # Folder containing validation images
if os.path.exists(gallery_folder):
    gallery_images = os.listdir(gallery_folder)
    if gallery_images:
        st.subheader('Gallery')
        num_columns = 5
        num_images = len(gallery_images)
        rows = num_images // num_columns + int(num_images % num_columns > 0)
        for i in range(rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < num_images:
                    image_name = gallery_images[index]
                    image_path = os.path.join(gallery_folder, image_name)
                    img = Image.open(image_path)
                    # Resize image to 1x1 aspect ratio
                    img = img.resize((100, 100))
                    if cols[j].button(image_name):
                        set_uploaded_image(image_path)
                    cols[j].image(img, caption=image_name, use_column_width=True)

# Prediction
if uploaded_image is not None or st.session_state.uploaded_image is not None:
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.session_state.uploaded_image = np.array(img)
    else:
        img = Image.fromarray(st.session_state.uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        if model_option == 'Keras Model Pre-trained':
            predictions = predict_with_keras(st.session_state.uploaded_image)
            for category, percentage in predictions:
                st.write(f"The image is {percentage:.2f}% likely to be {category}.")
        elif model_option == 'SVM Model (LBP)':
            predictions = predict_with_svmlbp(st.session_state.uploaded_image)
            st.write(f"The image is likely to be {predictions}.")
        
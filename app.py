import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('CAT VS DOG_cnn_model.h5')  # Make sure this is the path to your saved model

# Define function for prediction
def predict_class(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title("Cat vs Dog Classifier")

st.write("Upload a picture of a cat or dog, and we'll predict which one it is!")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Show uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction
    prediction = predict_class(uploaded_image)

    # Display prediction result
    if prediction[0] > 0.5:
        st.write("Prediction: **Dog**")
    else:
        st.write("Prediction: **Cat**")

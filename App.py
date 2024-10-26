import keras
import streamlit as st
from PIL import Image
import numpy as np

model = keras.models.load_model('model/model.h5')
class_names = ['Angry', 'Other', 'Sad', 'Happy']

st.title("Pet Expression Classifier")
st.write("Upload an image of a pet to classify its expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((244, 244))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    st.write(f"Predicted Expression: **{predicted_class}**")

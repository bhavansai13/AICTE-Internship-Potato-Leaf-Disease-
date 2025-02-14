import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Function to encode image to base64
def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load and encode background image
bg_image = "sai33.png"
bg_base64 = get_base64(bg_image)

# Apply background image using markdown and CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
    }}
    h1 {{
        color: black; /* Title color */
        text-align: center;
    }}
    .file-uploader {{"
        text-align: center;
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 10px;
        margin: auto;
        width: 50%;
    }}
    .upload-text {{
        text-align: center;
        font-size: 18px;
        color: yellow; /* Change color for "Please drop the image below" */
        font-weight: bold;
    }}
    .choose-image {{
        color: blue !important; /* Change color for "Choose an image" */
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Potato Plant Disease System")
app_mode = st.sidebar.selectbox('Select page', ['Disease Recognition'])

if app_mode == 'Disease Recognition':
    st.markdown("<h1 style='text-align: center;'>Potato Plant Disease Detection System</h1>", unsafe_allow_html=True)
    
    
    test_image = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button('Predict'):
        st.write("<p style='font-size: 20px; font-weight: bold;'>Prediction was:</p>", unsafe_allow_html=True)

        result_index = model_prediction(test_image)
        class_name = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__Healthy']

        st.markdown( 
            f"<p style='color: blue; font-size: 18px; font-weight: bold; background-color: rgba(255, 255, 255, 0.5); padding: 10px; border-radius: 5px;'>"
            f"Model is predicting it is a {class_name[result_index]}</p>",
            unsafe_allow_html=True
    )

    
        

# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np 
import seaborn as sns
import tensorflow as tf
from PIL import Image

# Setting page config
st.set_page_config(page_title="Food Predictor",page_icon=":hamburger:", layout="wide")

# Loading the model (make sure to load it once)
@st.cache_resource(show_spinner=False) # show spinner hide the "Running load_model()" when you opening the app first time
def load_model():
    model = tf.keras.models.load_model("efficientnetb0_fine_tuned_101_classes_mixed_precision.keras")
    return model

model = load_model()

# Class_name from the food
# this class is taken from the dataset after importing it and it used here for the testing purpose (so, we don't need to import data again and again)
class_names = ['apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare','beet_salad','beignets','bibimbap','bread_pudding','breakfast_burrito',
               'bruschetta','caesar_salad','cannoli','caprese_salad','carrot_cake','ceviche','cheesecake','cheese_plate','chicken_curry','chicken_quesadilla',
               'chicken_wings','chocolate_cake','chocolate_mousse','churros','clam_chowder','club_sandwich','crab_cakes','creme_brulee','croque_madame',
               'cup_cakes','deviled_eggs','donuts','dumplings','edamame','eggs_benedict','escargots','falafel','filet_mignon','fish_and_chips','foie_gras',
               'french_fries','french_onion_soup','french_toast','fried_calamari','fried_rice','frozen_yogurt','garlic_bread','gnocchi','greek_salad',
               'grilled_cheese_sandwich','grilled_salmon','guacamole','gyoza','hamburger','hot_and_sour_soup','hot_dog','huevos_rancheros','hummus',
               'ice_cream','lasagna','lobster_bisque','lobster_roll_sandwich','macaroni_and_cheese','macarons','miso_soup','mussels','nachos','omelette',
               'onion_rings','oysters','pad_thai','paella','pancakes','panna_cotta','peking_duck','pho','pizza','pork_chop','poutine','prime_rib',
               'pulled_pork_sandwich','ramen','ravioli','red_velvet_cake','risotto','samosa','sashimi','scallops','seaweed_salad','shrimp_and_grits',
               'spaghetti_bolognese','spaghetti_carbonara','spring_rolls','steak','strawberry_shortcake','sushi','tacos','takoyaki','tiramisu','tuna_tartare','waffles']

# Functions for image preparation and prediction

# 1. Image preparation
def load_and_prep(image, image_shape=224, scale=True):
    # Ensure the image is RGB not (RGBA transparent PNG image have 4 color channels rather than 3)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    img = tf.convert_to_tensor(np.array(image))
    img = tf.image.resize(img, [image_shape, image_shape])
    if scale:
        return img/255.0
    else:
        return img

# 2. Image prediction
def prediction(image):
    prep_img = load_and_prep(image, scale=False) # scale false because we are using the EfficientNetB0 it already has the rescale builtin
    pred = model.predict(tf.expand_dims(prep_img, axis=0))
    pred_class = class_names[pred.argmax()]
    return pred_class, pred.max()

# Web app 

st.write("""
# 🍔 Food Image Prediction (Deep Learning)
Welcome to the Food Image Predictor! This web app uses a deep learning model (EfficientNetB0) to identify different food items from images. With 81% accuracy, it can classify 101 types of delicious dishes like pizza, sushi, tacos, and more.

✨ How it works:

* Upload an image of any food item (JPG, JPEG, PNG).
* The model will analyze the image and predict the food category.
* You'll get the predicted dish name along with the confidence score.
* 🍽️ Try it now – Upload a food image and see what the model thinks!""")

# file upload
st.write("---")
st.write('### Choose an image...')
uploaded_file = st.file_uploader(label="", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with st.container():
        left,right = st.columns(2,border=True)
        with left:
            st.subheader("Uploaded Image")
            st.image(image,use_container_width=True)
        with right:
            # Show loading GIF
            with st.spinner("Processing the image...", show_time=True):
                # Make prediction
                pred_class, pred_prob = prediction(image)
                pred_prob = pred_prob * 100
            st.subheader("Prediction")
            st.success("Done")
            st.markdown(f"<h2>Prediction : <span style='color:green'>{pred_class}</span> and <br>Prediction Probability : <span style='color:green'>{pred_prob:.2f}%</span></h2>", unsafe_allow_html=True)
                
        st.write("")



# Intro of Mine
personal_image = "personal_img.png"
st.write("---")
st.header("About Me")
with st.container():
    # creating column in the container
    left_column, right_column = st.columns((0.2,0.7),border=True)
    # Left column
    with left_column:
        st.image(personal_image)
        st.markdown("<h3 align='center'>Abhishek</h3>",unsafe_allow_html=True)
    with right_column:
        st.markdown("""<h1 align="left">Hi 👋, I'm Abhishek</h1>
                    <p style='font-size:larger'>A Computer Science student at the United College of Institutions and Technology.<br> Currently in my final year, I’m passionate about Data science and aspire to make a mark in this field.<br>
                    🔍 My interests lie in solving technical challenges, conducting research, and mastering data management tools like SQL. I’m well-versed in numpy, pandas, matplotlib, machine learning, and statistics.</p>
                    <h3 align="left">Connect with me:</h3>
                    <p align="left">
                    <a href="https://twitter.com/abhishek208" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="abhishek208" height="30" width="40" /></a>
                    <a href="https://linkedin.com/in/https://www.linkedin.com/in/abhishek-kumar-44150723b/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="https://www.linkedin.com/in/abhishek-kumar-44150723b/" height="30" width="40" /></a>
                    </p>
                    <h3 align="left">Languages and Tools:</h3>
                    <p align="left"> <a href="https://www.cprogramming.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/c/c-original.svg" alt="c" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.mysql.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg" alt="mysql" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.postgresql.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original-wordmark.svg" alt="postgresql" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>
                    """,unsafe_allow_html=True)
        
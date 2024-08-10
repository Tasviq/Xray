import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Define the path to your saved model file
MODEL_PATH = 'chest_xray.h5'

# Function to load the trained model
def load_pneumonia_model():
    global model
    model = load_model(MODEL_PATH)

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    return img_data

# Function to predict pneumonia
def predict_pneumonia(image_path):
    img_data = preprocess_image(image_path)
    classes = model.predict(img_data)
    return classes

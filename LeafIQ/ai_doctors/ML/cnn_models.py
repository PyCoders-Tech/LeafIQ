import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import os


def predict_cnn_result(tree_name, test_type, image_path):
    if tree_name == 'potato':
        return potato_model(test_type, image_path)
    
    elif tree_name == 'rice':
        return rice_model(test_type, image_path)
    
    elif tree_name == 'corn':
        return corn_model(test_type, image_path)

    return None
    

def extract_steps_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            lines = file_content.split('\n')
            steps = []
            for line in lines:
                line = line.strip()
                if line:
                    steps.append(line)

            return steps

    except FileNotFoundError as e:
        raise e
        return None  # Return None if the file does not exist


def image_load(img_path):

    img = image.load_img('.'+img_path, target_size=(64, 64))
    

    if img is None:
        print('Error: Image not loaded.')
        raise ValueError(f"Failed to load image from path: {img_path}")
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def potato_model(test_name, image_path):
    img_array = image_load(image_path)
    #done
    if test_name == 'leaf':
        loaded_model = tf.keras.models.load_model('./ai_doctors/ML/models/potato/Potato_leaf_prediction_model.h5')
        predictions = loaded_model.predict(img_array)
        # For example, if your model uses one-hot encoding and you have a class mapping:
        class_mapping = {0: 'Potato Early_Blight', 1: 'Potato Late_Blight',
                         2: 'Potato Healthy'}  # Replace with your class mapping

        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Get the human-readable label
        predicted_class = class_mapping.get(predicted_class_index, 'Unknown')

        # Print the prediction
        print(f"Predicted Class: {predicted_class}")
        # Specify the path to your text file
        
        if predicted_class == "Potato Early_Blight":
            file_path = './ai_doctors/ML/instructions/potato/leaf/Potato Early_Blight.txt'
        elif predicted_class == "Potato Late_Blight":
            file_path = './ai_doctors/ML/instructions/potato/leaf/Potato Late_Blight.txt'
        else:
            file_path = './ai_doctors/ML/instructions/potato/leaf/Potate Healthy.txt'
        # Call the function to extract steps from the file
        instructions = extract_steps_from_file(file_path)

        return [predicted_class, instructions]
    elif test_name == 'skin':
        pass


def rice_model(test_name, image_path):
    img_array = image_load(image_path)
    #done
    if test_name == 'leaf':
        loaded_model = tf.keras.models.load_model('./ai_doctors/ML/models/rice/Rice_leaf_prediction_model.h5')
        predictions = loaded_model.predict(img_array)
        # For example, if your model uses one-hot encoding and you have a class mapping:
        class_mapping = {0: 'Rice__Brown_Spot', 1: 'Rice__Neck_Blast', 2: 'Rich__Healthy', 3: 'Rice__Leaf_Blast'}   # Replace with your class mapping

        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Get the human-readable label
        predicted_class = class_mapping.get(predicted_class_index, 'Unknown')

        # Print the prediction
        print(f"Predicted Class: {predicted_class}")
        # Specify the path to your text file
        
        if predicted_class == "Rice__Brown_Spot":
            file_path = './ai_doctors/ML/instructions/rice/leaf/Rice__Brown_Spot.txt'
        elif predicted_class == "Rice__Neck_Blast":
            file_path = './ai_doctors/ML/instructions/rice/leaf/Rice__Neck_Blast.txt'
        elif predicted_class == 'Rich__Healthy':
            file_path = './ai_doctors/ML/instructions/rice/leaf/Rich__Healthy.txt'
        else:
            file_path = './ai_doctors/ML/instructions/rice/leaf/Rice__Leaf_Blast.txt'
        # Call the function to extract steps from the file
        instructions = extract_steps_from_file(file_path)

        return [predicted_class, instructions]
    elif test_name == 'skin':
        pass


def corn_model(test_name, image_path):
    img_array = image_load(image_path)
    #done
    if test_name == 'leaf':
        loaded_model = tf.keras.models.load_model('./ai_doctors/ML/models/corn/Corn_leaf_prediction_model.h5')
        predictions = loaded_model.predict(img_array)
        # For example, if your model uses one-hot encoding and you have a class mapping:
        class_mapping = {0: 'Corn__Common_Rust', 1: 'Corn__Gray_Leaf_Spot', 2: 'Corn___Healthy', 3: 'Corn__Northern_Leaf_blight'}   # Replace with your class mapping

        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Get the human-readable label
        predicted_class = class_mapping.get(predicted_class_index, 'Unknown')

        # Print the prediction
        print(f"Predicted Class: {predicted_class}")
        # Specify the path to your text file
        
        if predicted_class == "Corn__Common_Rust":
            file_path = './ai_doctors/ML/instructions/corn/leaf/Corn__Common_Rust.txt'
        elif predicted_class == "Corn__Gray_Leaf_Spot":
            file_path = './ai_doctors/ML/instructions/corn/leaf/Corn__Gray_Leaf_Spot.txt'
        elif predicted_class == 'Corn___Healthy':
            file_path = './ai_doctors/ML/instructions/corn/leaf/Corn___Healthy.txt'
        else:
            file_path = './ai_doctors/ML/instructions/corn/leaf/Corn__Northern_Leaf_blight.txt'
        # Call the function to extract steps from the file
        instructions = extract_steps_from_file(file_path)

        return [predicted_class, instructions]
    elif test_name == 'skin':
        pass

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2


def predict_cnn_result(tree_name, test_type, image_path):
    if tree_name == 'potato':
        return potato_model(test_type, image_path)
    return 'test'
    

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
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    if img is None:
        print('Error: Image not loaded.')
        return None
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


def potato_model(test_name, image_path):
    img_array = image_load(image_path)
    if test_name == 'leaf':
        loaded_model = tf.keras.models.load_model('./ML models/potato/Potato_leaf_prediction_model.h5')
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
            file_path = 'instructions/potato/leaf/Potato Early_Blight.txt'
        elif predicted_class == "Potato Late_Blight":
            file_path = 'instructions/potato/leaf/Potato Late_Blight.txt'
        else:
            file_path = 'instructions/potato/leaf/Potato Healthy.txt'
        # Call the function to extract steps from the file
        instructions = extract_steps_from_file(file_path)

        return predicted_class, instructions
    elif test_name == 'skin':
        pass
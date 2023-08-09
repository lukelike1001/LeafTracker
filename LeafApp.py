# import library dependencies, tested on python 3.11.2
import gradio as gr
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# load the pre-trained model from the appropriate file path
def predict_plant(path):
    model = tf.keras.models.load_model('saved_model/my_model')

    # redefine values from the model
    img_height = img_width = 180
    class_names = ['bear_oak', 'boxelder', 'eastern_poison_ivy',
                   'eastern_poison_oak', 'fragrant_sumac',
                   'jack_in_the_pulpit', 'poison_sumac',
                   'virginia_creeper', 'western_poison_ivy',
                   'western_poison_oak']
    
    # load the image into a variable
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )

    # convert the image into a tensor and create a batch for testing
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # find the confidence probability for each plant
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    confidences = {class_names[i]: float(score[i]) for i in range(len(class_names))}
    return confidences

# add a title and description to the model
title = "Leaftracker Interactive Model"
description = """Leaftracker is an image classification model that differentiates toxic plants from their
                 non-toxic look-alikes. Built on TensorFlow, this interactive model has been ported to
                 Hugging Face as a web application. For further documentation, check out the Github
                 repository at https://github.com/lukelike1001/LeafTracker, and the project's info
                 page at https://lukelike1001.github.io/leaf.html."""

# launch the app
app = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    flagging_options=["incorrect", "other"],
    title=title,
    description=description,
    examples=[
        os.path.join(os.path.dirname(__file__), "tpc-imgs/bear_oak/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/boxelder/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/poison_sumac/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/western_poison_oak/000.jpg"),
    ],
)
app.launch()
import gradio as gr
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# Load the old model using a folder then convert it into a h5 file
folder_model = tf.keras.models.load_model('saved_model/', compile=False)
folder_model.compile(optimizer="adam",
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=["accuracy"])
folder_model.save("leaf_model.keras")

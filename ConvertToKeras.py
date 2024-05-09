import gradio as gr
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# Load the old model using a folder then convert it into a h5 file
folder_model = tf.keras.models.load_model('saved_model/')
folder_model.save("leaf_model.keras")

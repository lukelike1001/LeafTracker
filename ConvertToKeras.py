import tensorflow as tf
import keras

# Originally TensorFlow 2.13.0 Keras 2.13.1
print(tf.__version__)
print(keras.__version__)

# Load the old model using the folder
inference_layer = keras.layers.TFSMLayer('saved_model/', call_endpoint='serving_default')

# Specify the model architecture using an inference layer
model = tf.keras.Sequential([
    inference_layer,
])

# compile the model using the same parameters when first training the model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# save the model as a .keras file (in line with Keras 3 specifications)
model.save("leaf_model.keras")

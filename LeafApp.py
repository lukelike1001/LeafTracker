# import library dependencies, tested on python 3.11.2
import io
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

    # find the top three likeliest plants based on probabilities
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    top_three = np.array(score).argsort()[-3:][::-1]
    numpy_array = score.numpy()

    # convert the folder names into English words then return the three likeliest probabilities
    output = []
    for i in top_three:
        words = class_names[i].split("_")
        name = " ".join([word.capitalize() for word in words])
        output.append(
            "This image likely belongs to {} with {:.2f}% confidence."
            .format(name, 100 * numpy_array[i])
        )
    return "\n".join(output)

app = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    flagging_options=["blurry", "incorrect", "other"],
    examples=[
        os.path.join(os.path.dirname(__file__), "tpc-imgs/bear_oak/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/boxelder/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/poison_sumac/000.jpg"),
        os.path.join(os.path.dirname(__file__), "tpc-imgs/western_poison_oak/000.jpg"),
    ],
)
app.launch()
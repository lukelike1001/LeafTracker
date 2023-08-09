import gradio as gr

def greet_user(name):
    return "Hello " + name + "! Welcome to Gradio!"

app = gr.Interface(fn = greet_user, inputs="text", outputs="text")
app.launch()
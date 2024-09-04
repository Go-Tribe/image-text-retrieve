import gradio as gr
import tempfile
import os
import numpy as np
from PIL import Image

from image_retrieve import ImageRetrieve

image_retrieve = ImageRetrieve()

def search_text(text):
    results = image_retrieve.text2image(text)
    # Assuming text2image returns a list of image paths
    return results

def search_image(image):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(np.uint8(image))
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_filename = temp_file.name
        pil_image.save(temp_filename)
    
    print(temp_filename)
    # Use the temporary file for image search
    results = image_retrieve.image2image(temp_filename)
    # Clean up the temporary file
    os.unlink(temp_filename)
    
    return results

with gr.Blocks() as demo:
    gr.Markdown("# Image Search Engine")
    gr.Markdown("Search images by text using this demo.")
    
    with gr.Tab("Search Text"):
        with gr.Column():
            text_input = gr.Textbox(label="Query Text", info="Enter text to search for related images")
            text_output = gr.Gallery(label="Search Results", columns=3)
            text_button = gr.Button("Search")

    with gr.Tab("Search Image"):
        with gr.Row():
            image_input = gr.Image(label="Query Image")
            image_output = gr.Gallery(label="Search Results", columns=3)
        image_button = gr.Button("Search")

    text_button.click(search_text, inputs=text_input, outputs=text_output)
    image_button.click(search_image, inputs=image_input, outputs=image_output)



if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=8000)

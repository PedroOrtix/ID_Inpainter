import numpy as np
import gradio as gr
from typing import Dict

from src.signature_inpaint.utils.inference import remove_masked_lama

def dummy_fn(image_dict: Dict[str, np.ndarray]):
    return image_dict["background"]

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint with LaMa")
    with gr.Tab("Signature Inpainting"):
        with gr.Tab("Delete"):
            with gr.Row():
                image_input = gr.ImageMask(type="pil", layers=False)
            inpaint_button = gr.Button("Inpaint")
            image_view = gr.Image(interactive=True, type="pil", label="Image Signatute Template", show_download_button=True)
        with gr.Tab("Add"):
            with gr.Row():
                image_input = gr.Image(type="pil", interactive=True, scale = 2)
                with gr.Column(min_width=200):
                    # add text box for the 4 coordinates
                    x1 = gr.Textbox(label="X1")
                    y1 = gr.Textbox(label="Y1")
                    x2 = gr.Textbox(label="X2")
                    y2 = gr.Textbox(label="Y2")

            image_output = gr.Image(type="pil", interactive=False, scale = 0)

            with gr.Row():
                add_button = gr.Button("Add")

    # fn = remove_masked_lama
    inpaint_button.click(fn=remove_masked_lama, 
                    inputs=[image_input],
                    outputs=[image_input])
    
    image_input.change(fn=dummy_fn,
                       inputs=[image_input],
                    outputs=[image_view])


demo.launch(debug=True, show_error=True)

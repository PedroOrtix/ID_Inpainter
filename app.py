import numpy as np
import gradio as gr
from typing import Dict

from src.signature_inpaint.utils.inference import remove_masked_lama

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint with LaMa")
    with gr.Tab("Signature Inpainting"):
        with gr.Tab("Delete"):
            with gr.Row():
                image_delete = gr.ImageMask(type="pil", layers=False)
            with gr.Row():
                inpaint_button = gr.Button("Inpaint")
                send_add_button = gr.Button("Send to Add")
            image_delete_view = gr.Image(interactive=False, type="pil", label="Image Signatute Template", show_download_button=True)
        with gr.Tab("Add"):
            with gr.Row():
                image_add = gr.Image(type="pil", interactive=False, scale = 2, inputs=image_delete_view)
                with gr.Column(min_width=200):
                    # add text box for the 4 coordinates
                    x1 = gr.Textbox(label="X1")
                    y1 = gr.Textbox(label="Y1")
                    x2 = gr.Textbox(label="X2")
                    y2 = gr.Textbox(label="Y2")

            image_add_view = gr.Image(type="pil", interactive=False, scale = 0)

            with gr.Row():
                add_button = gr.Button("Add")

    # fn = remove_masked_lama
    inpaint_button.click(fn=remove_masked_lama, 
                        inputs=[image_delete],
                        outputs=[image_delete_view])
    
    send_add_button.click(fn=lambda x: x,
                        inputs=[image_delete_view],
                        outputs=[image_add])

demo.launch(debug=True, show_error=True)

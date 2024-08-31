import numpy as np
import gradio as gr
from typing import Dict
from PIL import Image, ImageDraw

from src.signature_inpaint.utils.inference import (remove_masked_lama,
                                                    detect_image_with_prompt,
                                                    segment_image_with_prompt
                                                    )

from src.signature_inpaint.utils.extra import (update_bounding_box,
                                            cut_image,
                                            update_and_cut,
                                            process_and_add_template)

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint with LaMa")
    with gr.Tab("Signature Inpainting"):
        with gr.Tab("Deleting phase"):
            image_delete = gr.ImageMask(type="pil", layers=False)
            with gr.Row():
                delete_button = gr.Button("Delete")
                send_add_button = gr.Button("Send to Add")
            image_delete_view = gr.Image(interactive=False, type="pil", label="Image Signature Template", show_download_button=True)
        with gr.Tab("Adding phase"):
            with gr.Row():
                with gr.Column(min_width=200):
                    image_add = gr.Image(type="pil", interactive=False, scale = 2, inputs=image_delete_view)
                with gr.Column(min_width=200):
                    # add text box for the 4 coordinates
                    x1 = gr.Number(label="X1")
                    y1 = gr.Number(label="Y1")
                    x2 = gr.Number(label="X2")
                    y2 = gr.Number(label="Y2")
                    set_bounding_box = gr.Button("Set Bounding Box")
                    template_image = gr.Image(type="pil", interactive=True, scale = 2, label="Template Image")
        with gr.Tab("Inpainting phase"):
            with gr.Row():
                sig_inpaint_1 = gr.Image(type="pil", interactive=True, scale=1, label="Original Signature")
                sig_segmented_1 = gr.Image(interactive=False, type="pil", label="Signature Segmented", show_download_button=True)
                with gr.Column(scale=0):
                    button_inpaint_1 = gr.Button("segment")
                    button_clear_1 = gr.Button("clear")
            with gr.Row():
                sig_inpaint_2 = gr.Image(type="pil", interactive=True, scale=1, label="Signature to Inpaint")
                sig_segmented_2 = gr.Image(interactive=False, type="pil", label="Signature Segmented", show_download_button=True)
                with gr.Column(scale=0):
                    button_inpaint_2 = gr.Button("segment")
                    button_clear_2 = gr.Button("clear")
            
            inpaint_button = gr.Button("Let's Inpaint")
            img_inpaint = gr.Image(type="pil", interactive=False, scale=1)
                

    # DELETE PHASE
    delete_button.click(fn=remove_masked_lama, 
                        inputs=[image_delete],
                        outputs=[image_delete_view])
    
    send_add_button.click(fn=lambda x: x,
                        inputs=[image_delete_view],
                        outputs=[image_add]).then(fn=detect_image_with_prompt,
                                                inputs=[image_delete],
                                                outputs=[x1, y1, x2, y2])

    # ADD PHASE
    set_bounding_box.click(fn=update_and_cut,
                            inputs=[image_delete_view, x1, y1, x2, y2],
                            outputs=[image_add, template_image])

demo.launch(debug=True, show_error=True)

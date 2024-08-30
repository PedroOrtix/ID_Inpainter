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
        with gr.Tab("Delete"):
            with gr.Row():
                image_delete = gr.ImageMask(type="pil", layers=False)
            with gr.Row():
                inpaint_button = gr.Button("Inpaint")
                send_add_button = gr.Button("Send to Add")
            image_delete_view = gr.Image(interactive=False, type="pil", label="Image Signature Template", show_download_button=True)
        with gr.Tab("Add"):
            with gr.Row():
                with gr.Column(min_width=200):
                    image_add = gr.Image(type="pil", interactive=False, scale = 2, inputs=image_delete_view)
                    image_add_view = gr.Image(interactive=False, type="pil", label="Image Signature Result", show_download_button=True)
                    original_segmented = gr.Image(interactive=False, type="pil", label="Original Segmented", show_download_button=True)
                    template_segmented = gr.Image(interactive=False, type="pil", label="Template Segmented", show_download_button=True)
                with gr.Column(min_width=200):
                    # add text box for the 4 coordinates
                    x1 = gr.Number(label="X1")
                    y1 = gr.Number(label="Y1")
                    x2 = gr.Number(label="X2")
                    y2 = gr.Number(label="Y2")
                    set_bounding_box = gr.Button("Set Bounding Box")
                    template_image = gr.Image(type="pil", interactive=True, scale = 2, label="Template Image")
                    add_button = gr.Button("Add")
                

    # fn = remove_masked_lama
    inpaint_button.click(fn=remove_masked_lama, 
                        inputs=[image_delete],
                        outputs=[image_delete_view])
    
    send_add_button.click(fn=lambda x: x,
                        inputs=[image_delete_view],
                        outputs=[image_add]).then(fn=detect_image_with_prompt,
                                                inputs=[image_delete],
                                                outputs=[x1, y1, x2, y2])

    set_bounding_box.click(fn=update_and_cut,
                            inputs=[image_delete_view, x1, y1, x2, y2],
                            outputs=[image_add, template_image])

    

    add_button.click(fn=process_and_add_template,
                    inputs=[image_delete, template_image, x1, y1, x2, y2],
                    outputs=[image_add_view, original_segmented, template_segmented])

demo.launch(debug=True, show_error=True)

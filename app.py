import numpy as np
import gradio as gr
from gradio_image_annotation import image_annotator
from gradio_image_prompter import ImagePrompter

from src.signature_inpaint.utils.inference import (remove_masked_lama, segment_image_with_points,
                                                    # detect_image_with_prompt,
                                                    # segment_image_with_prompt
                                                    )

from src.signature_inpaint.utils.signature_processing import process_and_send_to_inpaint

# from src.signature_inpaint.utils.bounding_box import update_and_cut
from src.signature_inpaint.utils.image_processing import cut_image

def crop(annotations):
    if annotations["boxes"]:
        box = annotations["boxes"][0]
        return annotations["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]
        ]
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint with LaMa")
    with gr.Tab("Signature Inpainting"):
        with gr.Tab("Deleting phase"):
            image_delete = gr.ImageMask(type="pil", layers=False)
            with gr.Row():
                delete_button = gr.Button("Delete")
                send_add_button = gr.Button("Send to Add")
                send_to_inpaint_button = gr.Button("Send to Inpaint")  # Nuevo botón
            image_delete_view = gr.Image(interactive=False, type="pil", label="Image Signature Template", show_download_button=True)
        with gr.Tab("Adding phase"):
            with gr.Row():
                with gr.Column(min_width=200):
                    func_aux = lambda x: x
                    annotator_crop = image_annotator(
                        image_type="numpy",
                        disable_edit_boxes=True,
                        single_box=True,
                    )
                with gr.Column(min_width=200):
                    crop_button = gr.Button("Recortar")
                    template_image = gr.Image(type="pil", interactive=True, scale=2, label="Template Image")
        with gr.Tab("Inpainting phase"):
            with gr.Row():
                sig_inpaint_1 = ImagePrompter(label="Original Signature")
                sig_segmented_1 = gr.Image(interactive=False, type="pil", label="Signature Segmented", show_download_button=True)
                with gr.Column(scale=0):
                    button_inpaint_1 = gr.Button("segment")
                    button_clear_1 = gr.Button("clear")
            with gr.Row():
                sig_inpaint_2 = ImagePrompter(label="Signature to Inpaint")
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
    
    # SEND TO ADD PHASE

    send_add_button.click(fn=lambda x: {"image": x, "boxes": []},
                        inputs=[image_delete_view],
                        outputs=[annotator_crop])
    
    # SEND TO INPAINT PHASE
    send_to_inpaint_button.click(
        fn=lambda x: process_and_send_to_inpaint(x),
        inputs=[image_delete],
        outputs=[sig_inpaint_1]
    )

    # ADD PHASE
    crop_button.click(fn=crop,
                    inputs=[annotator_crop],
                    outputs=[template_image])
    
    # INPAINT PHASE
    button_clear_1.click(fn=lambda x: {"image": x, "points": []},
                        inputs=[sig_inpaint_1],
                        outputs=[sig_inpaint_1])
    
    button_clear_2.click(fn=lambda x: {"image": x, "points": []},
                        inputs=[sig_inpaint_2],
                        outputs=[sig_inpaint_2])
    
    button_inpaint_1.click(fn=segment_image_with_points,
                        inputs=[sig_inpaint_1],
                        outputs=[sig_segmented_1])

    button_inpaint_2.click(fn=segment_image_with_points,
                        inputs=[sig_inpaint_2],
                        outputs=[sig_segmented_2])

demo.launch(debug=True, show_error=True)

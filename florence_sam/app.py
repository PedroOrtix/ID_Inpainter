import os
from typing import Tuple, Optional

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE
from utils.sam import load_sam_image_model, run_sam_inference

MARKDOWN = """
# Florence2 + SAM2 🔥

This demo integrates Florence2 and SAM2 by creating a two-stage inference pipeline. In 
the first stage, Florence2 performs tasks such as object detection, open-vocabulary 
object detection, image captioning, or phrase grounding. In the second stage, SAM2 
performs object segmentation on the image.
"""

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


def on_mode_dropdown_change(text):
    return [
        gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
    ]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    mode_dropdown, image_input, text_input
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None

    if not text_input:
        gr.Info("Please enter a text prompt.")
        return None, None

    texts = [prompt.strip() for prompt in text_input.split(",")]
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
    return annotate_image(image_input, detections), None

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Tab("Image"):
        image_processing_mode_dropdown_component = gr.Dropdown(
            choices=IMAGE_INFERENCE_MODES,
            value=IMAGE_INFERENCE_MODES[0],
            label="Mode",
            info="Select a mode to use.",
            interactive=True
        )
        with gr.Row():
            with gr.Column():
                image_processing_image_input_component = gr.Image(
                    type='pil', label='Upload image')
                image_processing_text_input_component = gr.Textbox(
                    label='Text prompt',
                    placeholder='Enter comma separated text prompts')
                image_processing_submit_button_component = gr.Button(
                    value='Submit', variant='primary')
            with gr.Column():
                image_processing_image_output_component = gr.Image(
                    type='pil', label='Image output')

    image_processing_submit_button_component.click(
        fn=process_image,
        inputs=[
            image_processing_mode_dropdown_component,
            image_processing_image_input_component,
            image_processing_text_input_component
        ],
        outputs=[
            image_processing_image_output_component
        ]
    )

    image_processing_text_input_component.submit(
        fn=process_image,
        inputs=[
            image_processing_mode_dropdown_component,
            image_processing_image_input_component,
            image_processing_text_input_component
        ],
        outputs=[
            image_processing_image_output_component
        ]
    )

demo.launch(debug=False, show_error=True, share=True)
import numpy as np
import gradio as gr

from src.signature_inpaint.utils.inference import remove_masked_lama

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint with LaMa")
    with gr.Tab("Signature Inpainting"):
        with gr.Row():
            image_input = gr.ImageMask(type="pil")
            # image_output = gr.Image(interactive=True, show_download_button=True, type="pil")
        run_button = gr.Button("Inpaint")
        image_view = gr.Image(interactive=False, type="pil", label="Result")


    run_button.click(fn=remove_masked_lama, 
                    inputs=[image_input],
                    outputs=[image_view])


demo.launch(debug=True, show_error=True)

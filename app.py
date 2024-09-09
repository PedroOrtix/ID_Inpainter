import numpy as np
import gradio as gr
from PIL import Image
from gradio_image_annotation import image_annotator
from gradio_image_prompter import ImagePrompter
from paddleocr import PaddleOCR
from src.signature_inpaint.utils.inference import remove_masked_lama, segment_image_with_points
from src.signature_inpaint.utils.signature_processing import procesar_y_fusionar_firma_template
from src.text_inpaint import ocr
from src.text_inpaint.utils import separar_cadenas
from src.text_inpaint.run import simple_inpaint

# from src.signature_inpaint.utils.bounding_box import update_and_cut

def crop(annotations):
    if annotations["boxes"]:
        box = annotations["boxes"][0]
        return annotations["image"][
            box["ymin"]:box["ymax"],
            box["xmin"]:box["xmax"]
        ]
    return None

def show_points(input_data):
    points = input_data.get("points", [])
    return str([point[:2] for point in points])

def extraer_entidades(image):
    if image is not None:
        # Inicializar el modelo OCR Paddle
        model = PaddleOCR(use_angle_cls=True, lang='es')  # 'es' para español
        
        # Realizar OCR en la imagen
        print("Extrayendo entidades...")
        bounds = model.ocr(np.array(image))
        bounds = ocr.convert_paddle_to_easyocr(bounds)
        print("Entidades extraídas")
        # Extraer las palabras detectadas
        lista_elementos_detectados = [bound[1] for bound in bounds]
        
        # Separar las cadenas en palabras y alfanuméricos
        dict_elems = separar_cadenas(lista_elementos_detectados)
        print(dict_elems)
        return dict_elems["alpha"], dict_elems["numeric"]
    return [], []

def mostrar_entidades(image, tipo):
    palabras, alfanumericos = extraer_entidades(image)
    if tipo == "Palabras":
        return ", ".join(palabras)
    else:
        return ", ".join(alfanumericos)

def recortar_imagen(image, entidad, height, width):
    # es el principo de la funcion process_image de run.py

    #saquemos los bounds de la imagen
    model = PaddleOCR(use_angle_cls=True, lang='es')  # 'es' para español
    bounds = model.ocr(np.array(image))
    bounds = ocr.convert_paddle_to_easyocr(bounds)

    palabra_a_reemplazar = entidad

    #recortamos la imagen
    img_pil, coordenadas_originales = ocr.recortar_imagen(bounds, palabra_a_reemplazar, np.array(image), height, width)
    img_pil = Image.fromarray(img_pil).convert('RGB')
    if 2*width <= 512:
        img_pil = ocr.juntar_imagenes_vertical(img_pil, img_pil)

    # dimesion de la imagen nueva con la adición vertical para futura restauración del padding
    # dim_img_pil = img_pil.size # comentado por ahora
    
    img_pil = ocr.rellenar_imagen_uniformemente(img_pil, dimensiones_objetivo=(512, 512))

    #mostramos la imagen recortada
    return img_pil, coordenadas_originales

def process_image_gradio(imagen_recortada, 
                         palabra_a_reemplazar, 
                         palabra_reemplazo,
                         height,
                         width,
                         slider_step,
                         slider_guidance,
                         slider_batch,
                         show_plot,
                         save_intermediate_images,
                         coordenadas_originales,
                         bounds_resized):
    
    model = PaddleOCR(use_angle_cls=True, lang='es')  # 'es' para español
    bounds_resized = model.ocr(np.array(imagen_recortada))
    bounds_resized = ocr.convert_paddle_to_easyocr(bounds_resized)

    
    right_bounds = [[bound for bound in bounds_resized if bound[1] == palabra_a_reemplazar][0]]

    if palabra_a_reemplazar.isalpha() and len(palabra_a_reemplazar) < len(palabra_reemplazo.strip()):
        right_bounds[0] = list(right_bounds[0])
        right_bounds[0][0] = ocr.recalcular_cuadricula_rotada(right_bounds[0][0], palabra_a_reemplazar, palabra_reemplazo)

    # Step 5: Perform inpainting to replace the word in the image
    modified_images, composed_prompt = simple_inpaint(imagen_recortada,
                                                    right_bounds,
                                                    [palabra_reemplazo],
                                                    slider_step=slider_step,
                                                    slider_guidance=slider_guidance,
                                                    slider_batch=slider_batch)
    # Reconstruir parcialmente las imágenes modificadas
    imagenes_reconstruidas = []
    coodinates = np.array(right_bounds[0][0], dtype = int).flatten()
    for img_modificada in modified_images:
        img_reconstruida = ocr.reemplazar_parte_imagen(imagen_recortada,
                                                        img_modificada,
                                                        coordinates=coodinates,
                                                        adjust_temp=False)
        
        #return to the originar size before the padding
        img_reconstruida, _ = ocr.recortar_imagen_uniformemente(img_reconstruida)
        # take the half of the image horizontally of img_recortad_mod
        img_reconstruida = img_reconstruida.crop((0, 0, img_reconstruida.width, img_reconstruida.height//2))

        imagenes_reconstruidas.append(img_reconstruida)

    # superponer la imagen 512x512 modificada a la original
    lista_imagenes_template =  [np.array(imagen_recortada)[:, :, :3] for _ in range(len(imagenes_reconstruidas))]
    x_min, y_min, x_max, y_max = coordenadas_originales

    for img_reconstruida, i in enumerate(imagenes_reconstruidas):
        lista_imagenes_template[i][y_min:y_max, x_min:x_max] = np.array(img_reconstruida)[:, :, :3]


    return lista_imagenes_template, right_bounds, coordenadas_originales
    
# Función auxiliar para mostrar las imágenes y coordenadas en la interfaz Gradio
def mostrar_resultados(modified_images, right_bounds, coordenadas_originales):
    images = [gr.Image.update(value=img) for img in modified_images]
    right_bounds_str = str(right_bounds)
    coordenadas_originales_str = str(coordenadas_originales)
    return images + [right_bounds_str, coordenadas_originales_str]

with gr.Blocks() as demo:
    gr.Markdown("# Inpaint of Document")
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
                sig_inpaint_1 = ImagePrompter(label="Signature to Inpaint", type="pil", scale=2)
                sig_segmented_1 = gr.Image(interactive=True, type="pil", label="Signature Segmented", show_download_button=True)
                
            button_inpaint_1 = gr.Button("segment")
            points_textbox_1 = gr.Textbox(label="Puntos seleccionados", interactive=False)
            
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

    # ADD PHASE
    crop_button.click(fn=crop,
                    inputs=[annotator_crop],
                    outputs=[template_image])

    button_inpaint_1.click(fn=segment_image_with_points,
                        inputs=[sig_inpaint_1],
                        outputs=[sig_segmented_1]).then(fn=show_points,
                                                        inputs=[sig_inpaint_1],
                                                        outputs=[points_textbox_1])
    
    inpaint_button.click(fn=procesar_y_fusionar_firma_template,
                        inputs=[image_delete, sig_inpaint_1, sig_segmented_1, annotator_crop],
                        outputs=[img_inpaint])
    

    with gr.Tab("Text Inpainting"):
        # Añadir un componente gr.State para almacenar las coordenadas originales
        coordenadas_originales_state = gr.State()
        bounds_resized_state = gr.State()

        with gr.Row():
            input_image = gr.Image(type="pil", label="Imagen de entrada")
            with gr.Column():
                tipo_entidad = gr.Radio(["Palabras", "Alfanuméricos"], label="Tipo de entidad", value="Palabras")
                entidades_textbox = gr.Textbox(label="Entidades detectadas", interactive=False, lines=3)
                entidad_seleccionada = gr.Textbox(label="Entidad a reemplazar", placeholder="Ingrese la entidad aquí")
        with gr.Row():
            extraer_button = gr.Button("Extraer entidades")
            recortar_button = gr.Button("Recortar imagen")

        with gr.Row():
            crop_image = gr.Image(type="pil", label="Imagen recortada")
            with gr.Column():
                tipo_entidad_2 = gr.Radio(["Palabras", "Alfanuméricos"], label="Tipo de entidad", value="Palabras")
                entidades_textbox_2 = gr.Textbox(label="Entidades detectadas", interactive=False, lines=3)
                entidad_seleccionada_2 = gr.Textbox(label="Entidad a reemplazar", placeholder="Ingrese la entidad aquí")
                entidad_reemplazo = gr.Textbox(label="Palabra de reemplazo", placeholder="Ingrese la palabra de reemplazo aquí")
                # Nuevos componentes para los parámetros de process_image
                height = gr.Slider(minimum=32, maximum=1024, value=512, step=32, label="Altura", interactive=True)
                width = gr.Slider(minimum=32, maximum=1024, value=512, step=32, label="Ancho", interactive=True)
                slider_step = gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Slider Step", interactive=True)
                slider_guidance = gr.Slider(minimum=0.1, maximum=10, value=2, step=0.1, label="Slider Guidance", interactive=True)
                slider_batch = gr.Slider(minimum=1, maximum=16, value=6, step=1, label="Slider Batch", interactive=True)
                show_plot = gr.Checkbox(label="Mostrar gráfico", value=False, interactive=True)
                save_intermediate_images = gr.Checkbox(label="Guardar imágenes intermedias", value=True, interactive=True)
        with gr.Row():
            extraer_button_2 = gr.Button("Extraer entidades de imagen recortada")
            get_candidates_button = gr.Button("Get candidates")

        extraer_button.click(fn=mostrar_entidades, 
                        inputs=[input_image, tipo_entidad], 
                        outputs=[entidades_textbox])

        tipo_entidad.change(fn=mostrar_entidades,
                        inputs=[input_image, tipo_entidad],
                        outputs=[entidades_textbox])

        recortar_button.click(
            fn=recortar_imagen,
            inputs=[input_image, entidad_seleccionada, height, width],
            outputs=[crop_image, coordenadas_originales_state]
        )

        # Nuevo botón para extraer entidades de la imagen recortada
        extraer_button_2.click(fn=mostrar_entidades, 
                        inputs=[crop_image, tipo_entidad_2], 
                        outputs=[entidades_textbox_2])

        tipo_entidad_2.change(fn=mostrar_entidades,
                        inputs=[crop_image, tipo_entidad_2],
                        outputs=[entidades_textbox_2])

        with gr.Row():
            process_button = gr.Button("Procesar imagen")
            results_gallery = gr.Gallery(label="Resultados del inpainting", height="auto")
            right_bounds_textbox = gr.Textbox(label="Right Bounds", interactive=False)
            coordenadas_originales_textbox = gr.Textbox(label="Coordenadas Originales", interactive=False)

        process_button.click(
            fn=process_image_gradio,
            inputs=[
                crop_image,
                entidad_seleccionada_2,
                entidad_reemplazo,
                height,
                width,
                slider_step,
                slider_guidance,
                slider_batch,
                show_plot,
                save_intermediate_images,
                coordenadas_originales_state,  # Añadir esto
                bounds_resized_state  # Añadir este parámetro
            ],
            outputs=[results_gallery, right_bounds_textbox, coordenadas_originales_textbox]
        )

demo.launch(debug=True, show_error=True)
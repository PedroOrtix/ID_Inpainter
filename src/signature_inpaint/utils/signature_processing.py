# signature_processing.py
from PIL import Image
import numpy as np
from .image_processing import preprocess_images, smooth_edges, paste_image
from .color_transfer import transfer_color
from .image_processing import cut_image
from .inference import detect_image_with_prompt

def procesar_y_fusionar_firma_template(original_signature: Image.Image, template_image: Image.Image, mask_original: Image.Image, mask_template: Image.Image, x1: int, y1: int, x2: int, y2: int, image_delete: Image.Image):
    """
    Procesa la firma original y el template, transfiere el color, suaviza los bordes y añade el resultado a la imagen original.

    Args:
        original_signature (PIL.Image.Image): Imagen de la firma original.
        template_image (PIL.Image.Image): Imagen del template.
        mask_original (PIL.Image.Image): Máscara de la firma original.
        mask_template (PIL.Image.Image): Máscara del template.
        x1, y1, x2, y2 (int): Coordenadas donde añadir el template procesado.
        image_delete (PIL.Image.Image): Imagen original donde se añadirá el template procesado.

    Returns:
        numpy.ndarray: Imagen final con el template procesado añadido.
    """
    original_rgb, template_rgb, original_lab, template_lab, mask_original_np, mask_template_np = preprocess_images(
        original_signature, template_image, mask_original, mask_template
    )
    
    colored_template = transfer_color(original_lab, template_lab, mask_original_np, mask_template_np)
    
    result = template_rgb * (1 - mask_template_np[:,:,np.newaxis]) + colored_template * mask_template_np[:,:,np.newaxis]
    
    result = smooth_edges(result, mask_template_np)
    
    colored_template = Image.fromarray(result.astype(np.uint8))
    final_result = paste_image(image_delete, colored_template, x1, y1, x2, y2)
    
    return final_result

def process_and_send_to_inpaint(image_dict):
    """
    Procesa una imagen para detectar y recortar una firma, y prepara el resultado para la fase de inpainting.

    Args:
        image_dict (dict): Un diccionario que contiene la imagen de fondo en la clave 'background'.

    Returns:
        dict: Un diccionario con la imagen recortada y una lista vacía de puntos.
            - 'image' (PIL.Image.Image): La imagen de la firma recortada.
            - 'points' (list): Una lista vacía para futuros puntos de referencia.

    Este proceso incluye:
    1. Detectar las coordenadas de la firma en la imagen.
    2. Recortar la imagen usando las coordenadas detectadas.
    3. Preparar un diccionario con la imagen recortada y una lista vacía de puntos.
    """
    # Detectar las coordenadas de la firma
    x1, y1, x2, y2 = detect_image_with_prompt(image_dict)
    
    # Recortar la imagen usando las coordenadas detectadas
    background_image = image_dict["background"]
    cropped_image = cut_image(background_image, x1, y1, x2, y2)
    
    # Crear el diccionario con la imagen recortada y una lista vacía de puntos
    result = {
        "image": cropped_image,
        "points": []
    }
    
    return result
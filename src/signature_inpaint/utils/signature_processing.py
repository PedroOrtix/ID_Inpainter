# signature_processing.py
from PIL import Image
import numpy as np
import cv2
from .image_processing import cut_image
from .inference import detect_image_with_prompt
from .inference import detect_image_with_prompt
import colortrans

def procesar_y_fusionar_firma_template(original_signature, signature_to_inpaint, signature_to_inpaint_mask, annotator_crop):
    """
    Procesa la firma original y la firma a inpaintar, transfiere el estilo y añade el resultado a la imagen original.

    Args:
        original_signature (dict): Diccionario con la imagen de la firma original y sus puntos.
        signature_to_inpaint (dict): Diccionario con la imagen de la firma a inpaintar y sus puntos.
        signature_to_inpaint_mask (PIL.Image.Image): Máscara de la firma a inpaintar.
        annotator_crop (dict): Diccionario con la imagen de fondo y su máscara. (perteneciente a la fase de adding)

    Returns:
        PIL.Image.Image: Imagen final con la firma procesada añadida.
    """
    # Extraer las imágenes de los diccionarios
    x1_original, y1_original, x2_original, y2_original = detect_image_with_prompt(original_signature["background"])
    original_img = cut_image(original_signature["background"], x1_original, y1_original, x2_original, y2_original)

    to_inpaint_img = signature_to_inpaint['image']
    background_img = annotator_crop['image']
    x1 = annotator_crop['boxes'][0]["xmin"]
    y1 = annotator_crop['boxes'][0]["ymin"]
    x2 = annotator_crop['boxes'][0]["xmax"]
    y2 = annotator_crop['boxes'][0]["ymax"]

    # Convertir las imágenes a arrays de numpy si no lo son ya
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img)
    if isinstance(to_inpaint_img, Image.Image):
        to_inpaint_img = np.array(to_inpaint_img)
    if isinstance(background_img, Image.Image):
        background_img = np.array(background_img)
    if isinstance(signature_to_inpaint_mask, Image.Image):
        signature_to_inpaint_mask = np.array(signature_to_inpaint_mask)

    # Hacer una transferencia de color
    output_pccm = colortrans.transfer_lhm(to_inpaint_img, original_img)

    # Obtener las dimensiones del área de recorte
    crop_height = y2 - y1
    crop_width = x2 - x1

    # Redimensionar las imágenes y la máscara al tamaño del área de recorte
    output_pccm = cv2.resize(output_pccm, (crop_width, crop_height))
    signature_to_inpaint_mask = cv2.resize(signature_to_inpaint_mask, (crop_width, crop_height))

    # Segmentar la imagen LHM utilizando la máscara del template
    output_lhm_segmentado = np.zeros_like(output_pccm)
    output_lhm_segmentado[signature_to_inpaint_mask > 0] = output_pccm[signature_to_inpaint_mask > 0]

    # Crear una copia del fondo
    resultado = background_img.copy()

    # Pegar la firma procesada y segmentada en el área de recorte
    resultado[y1:y2, x1:x2][signature_to_inpaint_mask > 0] = output_lhm_segmentado[signature_to_inpaint_mask > 0]

    return resultado

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
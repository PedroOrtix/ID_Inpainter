# image_processing.py
import numpy as np
import cv2
from PIL import Image

def smooth_edges(result, mask_template_np):
    """
    Suaviza los bordes de la imagen resultante usando la máscara del template.

    Args:
        result (numpy.ndarray): Imagen resultante.
        mask_template_np (numpy.ndarray): Máscara del template.

    Returns:
        numpy.ndarray: Imagen con bordes suavizados.
    """
    kernel = np.ones((3,3), np.float32) / 9
    mask_dilated = cv2.dilate(mask_template_np.astype(np.uint8), kernel, iterations=1)
    mask_edge = mask_dilated - mask_template_np
    result_blurred = cv2.filter2D(result.astype(np.float32), -1, kernel)
    result = np.where(mask_edge[:,:,np.newaxis] > 0, result_blurred, result)
    return result

def paste_image(original_image, template_image, x1, y1, x2, y2):
    """
    Pega una imagen template sobre una imagen original en las coordenadas especificadas.

    Args:
        original_image (PIL.Image.Image o numpy.ndarray): Imagen original.
        template_image (PIL.Image.Image o numpy.ndarray): Imagen a pegar.
        x1, y1, x2, y2 (int): Coordenadas donde pegar la imagen template.

    Returns:
        numpy.ndarray: Imagen resultante después de pegar el template.
    """
    if original_image is None or template_image is None:
        return None
    
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)
    if isinstance(template_image, np.ndarray):
        template_image = Image.fromarray(template_image)
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    template_image = template_image.resize((int(x2 - x1), int(y2 - y1)))
    
    result_image = original_image.copy()
    result_image.paste(template_image, (int(x1), int(y1)))
    
    return np.array(result_image)

def cut_image(image, x1, y1, x2, y2):
    """
    Recorta una sección de la imagen basada en las coordenadas dadas.

    Args:
        image (PIL.Image.Image o numpy.ndarray): Imagen a recortar.
        x1, y1, x2, y2 (int): Coordenadas del área a recortar.

    Returns:
        PIL.Image.Image: Imagen recortada.
    """
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image
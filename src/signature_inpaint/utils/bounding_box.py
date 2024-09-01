# bounding_box.py
from PIL import Image, ImageDraw
import numpy as np
from .image_processing import cut_image

def update_bounding_box(image, x1, y1, x2, y2):
    """
    Dibuja un rectángulo verde en la imagen para representar un bounding box.

    Args:
        image (PIL.Image.Image o numpy.ndarray): Imagen a actualizar.
        x1, y1, x2, y2 (int): Coordenadas del bounding box.

    Returns:
        numpy.ndarray: Imagen con el bounding box dibujado.
    """
    if image is None:
        return None
    if isinstance(image, Image.Image):
        img = image.copy()
    else:
        img = Image.fromarray(np.array(image))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    return np.array(img)

def update_and_cut(image, x1, y1, x2, y2):
    """
    Actualiza el bounding box en la imagen y recorta la sección especificada.

    Args:
        image (PIL.Image.Image o numpy.ndarray): Imagen a procesar.
        x1, y1, x2, y2 (int): Coordenadas del área a procesar.

    Returns:
        tuple: (imagen actualizada con bounding box, imagen recortada)
    """
    updated_image = update_bounding_box(image, x1, y1, x2, y2)
    cropped_image = cut_image(image, x1, y1, x2, y2)
    return updated_image, cropped_image
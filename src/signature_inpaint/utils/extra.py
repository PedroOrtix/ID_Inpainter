import datetime
import os
import shutil
import uuid
from PIL import Image, ImageDraw
import numpy as np


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def delete_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    try:
        shutil.rmtree(directory_path)
    except PermissionError:
        raise PermissionError(
            f"Permission denied: Unable to delete '{directory_path}'.")


def generate_unique_name():
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4()
    return f"{current_datetime}_{unique_id}"

def update_bounding_box(image, x1, y1, x2, y2):
        if image is None:
            return None
        if isinstance(image, Image.Image):
            img = image.copy()
        else:
            img = Image.fromarray(np.array(image))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        return np.array(img)

def cut_image(image, x1, y1, x2, y2):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    
    # Asegurarse de que las coordenadas estén en el orden correcto
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Recortar la imagen
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def paste_image(original_image, template_image, x1, y1, x2, y2):
    if original_image is None or template_image is None:
        return None
    
    # Convertir a objetos PIL Image si no lo son ya
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)
    if isinstance(template_image, np.ndarray):
        template_image = Image.fromarray(template_image)
    
    # Asegurarse de que las coordenadas estén en el orden correcto
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Redimensionar el template_image si es necesario
    template_image = template_image.resize((int(x2 - x1), int(y2 - y1)))
    
    # Crear una copia de la imagen original
    result_image = original_image.copy()
    
    # Pegar el template_image en la posición correcta
    result_image.paste(template_image, (int(x1), int(y1)))
    
    return np.array(result_image)

def add_template_to_image(original_image, template_image, x1, y1, x2, y2):
    return paste_image(original_image, template_image, x1, y1, x2, y2)
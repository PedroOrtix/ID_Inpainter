import datetime
import os
import shutil
from typing import Dict
import uuid
from PIL import Image, ImageDraw
import numpy as np

from .inference import segment_image_with_prompt


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

def update_and_cut(image, x1, y1, x2, y2):
        updated_image = update_bounding_box(image, x1, y1, x2, y2)
        cropped_image = cut_image(image, x1, y1, x2, y2)
        return updated_image, cropped_image

def add_template_to_image(original_image, template_image, x1, y1, x2, y2):
    return paste_image(original_image, template_image, x1, y1, x2, y2)

def process_and_add_template(image_delete: Dict[str, Image.Image], template_image: Image.Image, x1, y1, x2, y2):
    # esto en un futuro podria ser mejorado para que no se tenga que hacer esto
    image_delete = image_delete["background"]
    # Recortar la firma original
    original_signature = cut_image(image_delete, x1, y1, x2, y2)
        
    # Segmentar la firma original
    _, _, mask_original = segment_image_with_prompt(original_signature, "signature")
        
    # Segmentar la firma del template
    _, _, mask_template = segment_image_with_prompt(template_image, "signature")
        
    # Calcular la media RGB de la firma original segmentada
    original_rgb = np.array(original_signature)
    mask_original_np = np.array(mask_original) / 255
    masked_original = original_rgb * mask_original_np[:,:,np.newaxis]
    mean_rgb = np.mean(masked_original[masked_original != 0], axis=0)
        
    # Aplicar el color medio solo a los píxeles de la firma en el template
    template_rgb = np.array(template_image)
    mask_template_np = np.array(mask_template) / 255
        
    # Crear una máscara de 3 canales
    mask_3channel = np.repeat(mask_template_np[:, :, np.newaxis], 3, axis=2)
        
    # Aplicar el color medio solo a los píxeles de la firma
    colored_signature = mean_rgb * mask_3channel
    template_rgb = template_rgb * (1 - mask_3channel) + colored_signature
        
    colored_template = Image.fromarray(template_rgb.astype(np.uint8))
        
    # Añadir el template coloreado a la imagen original
    result = add_template_to_image(image_delete, colored_template, x1, y1, x2, y2)
        
    return result, mask_original, mask_template
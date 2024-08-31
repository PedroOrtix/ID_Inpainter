import datetime
import os
import shutil
from typing import Dict
import uuid
from PIL import Image, ImageDraw
import numpy as np
import cv2

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

def preprocess_images(original_signature, template_image, mask_original, mask_template):
    # Convertir máscaras a numpy arrays
    mask_original_np = np.array(mask_original) / 255
    mask_template_np = np.array(mask_template) / 255
    
    # Convertir imágenes a numpy arrays
    original_rgb = np.array(original_signature)
    template_rgb = np.array(template_image)
    
    # Aplicar máscaras
    original_masked = original_rgb * mask_original_np[:,:,np.newaxis]
    template_masked = template_rgb * mask_template_np[:,:,np.newaxis]
    
    # Convertir a LAB solo las partes segmentadas
    original_lab = cv2.cvtColor(original_masked.astype(np.uint8), cv2.COLOR_RGB2LAB)
    template_lab = cv2.cvtColor(template_masked.astype(np.uint8), cv2.COLOR_RGB2LAB)
    
    return original_rgb, template_rgb, original_lab, template_lab, mask_original_np, mask_template_np

def transfer_color(original_lab, template_lab, mask_original_np, mask_template_np):
    # Transferir color solo en los canales A y B
    for i in range(3):
        mean_i = np.mean(original_lab[:,:,i][mask_original_np > 0])
        std_i = np.std(original_lab[:,:,i][mask_original_np > 0])
        template_lab[:,:,i] = np.where(
            mask_template_np > 0,
            (template_lab[:,:,i] - np.mean(template_lab[:,:,i][mask_template_np > 0])) / np.std(template_lab[:,:,i][mask_template_np > 0]) * std_i + mean_i,
            template_lab[:,:,i]
        )
    
    # Convertir de vuelta a RGB
    colored_template = cv2.cvtColor(template_lab, cv2.COLOR_LAB2RGB)
    return colored_template

def smooth_edges(result, mask_template_np):
    kernel = np.ones((3,3), np.float32) / 9
    mask_dilated = cv2.dilate(mask_template_np.astype(np.uint8), kernel, iterations=1)
    mask_edge = mask_dilated - mask_template_np
    result_blurred = cv2.filter2D(result.astype(np.float32), -1, kernel)
    result = np.where(mask_edge[:,:,np.newaxis] > 0, result_blurred, result)
    return result

def process_and_add_template(image_delete: Dict[str, Image.Image], template_image: Image.Image, x1, y1, x2, y2):
    image_delete = image_delete["background"]
    original_signature = cut_image(image_delete, x1, y1, x2, y2)
    
    # Segmentar las firmas
    _, _, mask_original = segment_image_with_prompt(original_signature, "signature")
    _, _, mask_template = segment_image_with_prompt(template_image, "signature")
    
    # Preprocesar imágenes
    original_rgb, template_rgb, original_lab, template_lab, mask_original_np, mask_template_np = preprocess_images(
        original_signature, template_image, mask_original, mask_template
    )
    
    # Transferir color y convertir de vuelta a RGB
    colored_template = transfer_color(original_lab, template_lab, mask_original_np, mask_template_np)
    
    # Combinar la firma coloreada con el fondo original
    result = template_rgb * (1 - mask_template_np[:,:,np.newaxis]) + colored_template * mask_template_np[:,:,np.newaxis]
    
    # Suavizar bordes
    result = smooth_edges(result, mask_template_np)
    
    # Convertir a Image y añadir a la imagen original
    colored_template = Image.fromarray(result.astype(np.uint8))
    final_result = add_template_to_image(image_delete, colored_template, x1, y1, x2, y2)
    
    return final_result, mask_original, mask_template
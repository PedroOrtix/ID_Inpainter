import torch
import numpy as np
from PIL import Image
import supervision as sv
from typing import Tuple, List, Dict

from .florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from .sam import load_sam_image_model, run_sam_inference
from simple_lama_inpainting import SimpleLama

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

# Inicializar SimpleLama (esto debería hacerse una sola vez, fuera de la función)
simple_lama = SimpleLama()

def process_image_with_prompt(img_path: str, prompt: str) -> Tuple[List[Dict[str, np.ndarray]], Image.Image, Image.Image]:
    # Cargar la imagen
    image = Image.open(img_path).convert("RGB")
    
    # Ejecutar la inferencia de Florence
    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image,
        task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
        text=prompt
    )
    
    # Convertir los resultados a detecciones de Supervision
    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=result,
        resolution_wh=image.size
    )
    
    # Ejecutar la inferencia de SAM
    detections = run_sam_inference(SAM_IMAGE_MODEL, image, detections)
    
    # Preparar los resultados
    mask_coordinates = []
    for xyxy, mask in zip(detections.xyxy, detections.mask):
        mask_coordinates.append({
            "bbox": xyxy.tolist(),
            "mask": mask.astype(np.uint8)
        })
    
    # Crear la imagen enmascarada
    masked_image = image.copy()
    mask_image = Image.new("L", image.size, 0)
    for coord in mask_coordinates:
        mask_array = coord["mask"]
        mask = Image.fromarray(mask_array * 255)
        masked_image.paste(image, mask=mask)
        mask_image.paste(255, mask=mask)
    
    return mask_coordinates, masked_image, mask_image

def remove_masked_element(original_image: Image.Image, mask_image: Image.Image) -> Image.Image:
    # Asegurarse de que la máscara esté en modo 'L' (escala de grises)
    mask_image = mask_image.convert('L')
    
    # Realizar el inpainting con SimpleLama
    result = simple_lama(original_image, mask_image)
    
    return result

# Ejemplo de uso:
# mask_coords, masked_img, mask_img = process_image_with_prompt("ruta/a/la/imagen.jpg", "gato")
# image_without_element = remove_masked_element(Image.open("ruta/a/la/imagen.jpg"), mask_img)
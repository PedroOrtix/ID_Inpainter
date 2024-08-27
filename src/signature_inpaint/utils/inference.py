import torch
import numpy as np
from PIL import Image
import supervision as sv
from typing import Tuple, List, Dict

from .florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from .sam import load_sam_image_model, run_sam_inference
from diffusers import AutoPipelineForInpainting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

# Reemplazar la definición del pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)
pipe.enable_model_cpu_offload()
# pipe.enable_xformers_memory_efficient_attention()

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

def remove_masked_element(original_image: Image.Image, 
                        mask_image: Image.Image, 
                        prompt: str = "background",
                        negative_prompt: str = "",
                        guidance_scale: float = 7.5,
                        strength: float = 1,
                        num_inference_steps: int = 30) -> Image.Image:
    
    # Guardar las dimensiones originales
    original_size = original_image.size

    # Redimensionar las imágenes si es necesario
    width, height = original_image.size
    if width % 8 != 0 or height % 8 != 0:
        width = (width // 8) * 8
        height = (height // 8) * 8
        original_image = original_image.resize((width, height))
        mask_image = mask_image.resize((width, height))

    # Realizar el inpainting para eliminar el elemento
    generator = torch.Generator(DEVICE).manual_seed(92)
    inpainted_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator
    ).images[0]

    # Asegurarse de que todas las imágenes tengan el mismo tamaño
    inpainted_image = inpainted_image.resize(original_image.size)
    mask_image = mask_image.resize(original_image.size)

    # Combinar la imagen original con la región inpaintada
    result = Image.new("RGB", original_image.size)
    result.paste(original_image, (0, 0))
    result.paste(inpainted_image, (0, 0), mask_image)

    # Redimensionar el resultado a las dimensiones originales si es necesario
    if result.size != original_size:
        result = result.resize(original_size)

    return result


# Ejemplo de uso:
# mask_coords, masked_img, mask_img = process_image_with_prompt("ruta/a/la/imagen.jpg", "gato")
# image_without_element = remove_masked_element(Image.open("ruta/a/la/imagen.jpg"), mask_img)
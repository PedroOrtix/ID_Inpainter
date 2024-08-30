import torch
import numpy as np
from PIL import Image
import supervision as sv
from typing import Tuple, List, Dict

from .florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from .sam import load_sam_image_model, run_sam_inference
from simple_lama_inpainting import SimpleLama
from diffusers import StableDiffusionInpaintPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

# Inicializar SimpleLama (esto debería hacerse una sola vez, fuera de la función)
simple_lama = SimpleLama()

# # Definir el pipeline fuera de la función
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     torch_dtype=torch.float16
# ).to(DEVICE)

def detect_image_with_prompt(img_dict: Dict[str, Image.Image]) -> Tuple[List[Dict[str, List[float]]], Image.Image]:
    # Cargar la imagen
    image = img_dict["background"].convert("RGB")
    
    # Ejecutar la inferencia de Florence
    _, result = run_florence_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image,
        task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
        text="black signature"
    )
    
    # Convertir los resultados a detecciones de Supervision
    detections = sv.Detections.from_lmm(
        lmm=sv.LMM.FLORENCE_2,
        result=result,
        resolution_wh=image.size
    )
    
    # Preparar los resultados
    bbox_coordinates = [{"bbox": xyxy.tolist()} for xyxy in detections.xyxy]
    x1, y1, x2, y2 = bbox_coordinates[0]["bbox"]
    print("the coordinates are: x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2)
    
    return x1, y1, x2, y2

def segment_image_with_prompt(img_path: str, prompt: str) -> Tuple[List[Dict[str, np.ndarray]], Image.Image, Image.Image]:
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

# función auxiliar para preparar la máscara
def prepare_mask(mask):
    # Tomar el valor máximo a través de los canales
    mask = mask.max(dim=1, keepdim=True)[0]
    
    # Asegurar máscara binaria
    mask = (mask > 0.5).float()
    
    # Mover a CPU antes de convertir a numpy
    mask = mask.cpu()
    
    return mask

def remove_masked_lama(image_dict: Dict[str, Image.Image]) -> Image.Image:
    # Get the original image and mask from the dictionary
    original_image = image_dict["background"].convert("RGB")
    mask_image = np.array(image_dict["layers"][0])

    # lo pasamos a tensor para preprocesarlo
    mask_image = torch.tensor(mask_image).transpose(0, 2).unsqueeze(0)
    mask_image = prepare_mask(mask_image)
    # lo volvemos a pasar a numpy
    mask_image = mask_image.squeeze().transpose(1, 0).numpy()
    # Initialize SimpleLama
    simple_lama = SimpleLama()

    # Perform inpainting
    result_array = simple_lama(original_image, mask_image)
    return result_array

def remove_masked_sd(original_image: Image.Image, 
                        mask_image: Image.Image, 
                        prompt: str = "background",
                        negative_prompt: str = "",
                        guidance_scale: float = 7,
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
    inpainted_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
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
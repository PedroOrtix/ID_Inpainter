from td_inpaint import inpaint
from inpaint_functions import parse_bounds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

import ocr
from ocr import recalcular_cuadricula_rotada
# from ocr import comparar_imagenes
# from ocr import reemplazar_parte_imagen

from utils import separar_cadenas, mostrar_diccionario_ascii

def simple_inpaint(image, bounds, word, slider_step=30, slider_guidance=1.5, slider_batch=6):
    """
    Perform inpainting on the given image using the specified bounds and word.
    Args:
        image (PIL.Image): The image to inpaint.
        bounds (str): The bounds for inpainting.
        word (str): The word for inpainting.
        slider_step (int, optional): The step size for the slider. Defaults to 25.
        slider_guidance (float, optional): The guidance value for the slider. Defaults to 2.5.
        slider_batch (int, optional): The batch size for the slider. Defaults to 4.
    Returns:
        The inpainted image, coordinates
    """
    global_dict = {}
    global_dict["stack"] = parse_bounds(bounds, word)
    # print(global_dict["stack"])   
    #image = "./hat.jpg"
    prompt = ""
    keywords = ""
    positive_prompt = ""
    radio = 8
    slider_natural= False
    return inpaint(image, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict)

def process_image(palabra, 
                    replace, 
                    bounds, 
                    img_array,
                    slider_step=30,
                    slider_guidance=2,
                    slider_batch=6,
                    show_plot=False,
                    save_intermediate_images=True):
    """
    Processes an image by identifying a word, replacing it with a new word, and returning
    the modified images. Optionally, displays the modified images in a grid.

    Parameters:
    ----------
    palabra : str
        The word to be identified and replaced in the image.
    replace : str
        The word that will replace the identified word.
    bounds : list
        The bounding box coordinates of the word in the original image.
    img_array : np.ndarray
        The image array where the word is located.
    show_plot : bool, optional
        If True, displays the modified images in a grid. Default is False.
    save_intermediate_images : bool, optional
        If True, saves the intermediate images during the processing. Default is True.

    Returns:
    -------
    list of PIL.Image
        A list containing the modified images after the word replacement.
    right_bounds : list
        The bounding box coordinates of the identified word in the modified image.
    coordenadas_originales : list
        The original bounding box coordinates of the word in the image
    """
    # Step 1: Resize and crop the image based on the bounding box and the word to be replaced
    img_resized, coordenadas_originales = ocr.recortar_imagen(bounds, palabra, img_array, nueva_dimension=512)
    img_pil = Image.fromarray(img_resized).convert('RGB')
    
    # Save the cropped and resized image for reference (optional)
    if save_intermediate_images:
        img_pil.save("images/imagen_dni_recortada.jpg")

    # Inicializar el modelo OCR Paddle
    model = PaddleOCR(use_angle_cls=True, lang='es')  # 'es' para español
    # Step 2: Perform OCR on the resized and cropped image to identify word bounding boxes
    bounds_resized = model.ocr(np.array(img_pil))
    bounds_resized = ocr.convert_paddle_to_easyocr(bounds_resized)
    
    # mostramos otra vez las palabras detectadas, pero de la imagen recortada 512x512
    # reason: para que el usuario pueda ver las palabras detectadas y elegir la que desea reemplazar
    # el modelo ocr puede detectar en la nueva imagen recortada palabras que no estaban en la imagen original
    lista_elementos_detectados = []
    for bound in bounds_resized:
        lista_elementos_detectados.append(bound[1])
    
    dict_elems = separar_cadenas(lista_elementos_detectados)
    mostrar_diccionario_ascii(dict_elems)
    
    # ask the user to input the word to be replaced
    print("Por favor, introduzca la palabra a reemplazar: (por motivos de la demo)")
    palabra = input()

    # Step 3: Find the bounding box that matches the word to be replaced
    right_bounds = [bound for bound in bounds_resized if bound[1] == palabra]

    # Step 4: Recalculate the bounding box if necessary (e.g., if the replacement word is longer)
    if palabra.isalpha() and len(palabra) < len(replace.strip()):
        right_bounds[0] = list(right_bounds[0])
        right_bounds[0][0] = recalcular_cuadricula_rotada(right_bounds[0][0], palabra, replace)

    # Step 5: Perform inpainting to replace the word in the image
    modified_images, composed_prompt = simple_inpaint(img_pil,
                                                    right_bounds,
                                                    [replace],
                                                    slider_step=slider_step,
                                                    slider_guidance=slider_guidance,
                                                    slider_batch=slider_batch)

    # Step 6: Optionally display the results in a grid format (3x2)
    if show_plot:
        fig, axs = plt.subplots(slider_batch//2, 2, figsize=(10, 10))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(modified_images[i])
            ax.axis('off')
            ax.set_title(f"Resultado {i+1}")

        plt.show()

    return modified_images, right_bounds, coordenadas_originales
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
def display(image):
    # Display image using Matplotlib
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Function to fit text within a bounding box
def fit_text_in_box(draw, text, box, font_path='arial.ttf'):
    p0, p1, p2, p3 = box
    box_width = max(p1[0] - p0[0], p2[0] - p3[0])
    box_height = max(p3[1] - p0[1], p2[1] - p1[1])
    
    # Start with a large font size and decrease until the text fits within the box
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    while text_width > box_width or text_height > box_height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
    
    return font
# Function to draw bounding boxes around detected text on a given image
def draw_boxes(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    draw = ImageDraw.Draw(image)
    for i in range(len(bounds)):
        bound = bounds[i]
        box = bound[0]
        
        p0, p1, p2, p3 = box
        if fill_color:
            # Fill the box with the fill color
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(0,255,0), width=width)
        
    return image

# Function to draw bounding boxes around detected text on a given image
def draw_mask(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    draw = ImageDraw.Draw(image)
    for i in range(len(bounds)):
        bound = bounds[i]
        box = bound[0]
        text = replace[i] if replace else bound[1]
        
        p0, p1, p2, p3 = box
        if fill_color:
            # Fill the box with the fill color
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(193,193,193), width=width)
        
        font = fit_text_in_box(draw, text, box)
        text_width, text_height = draw.textsize(text, font=font)
        
        text_x = p0[0] + (p1[0] - p0[0] - text_width) / 2
        text_y = p0[1] + (p3[1] - p0[1] - text_height) / 2
        draw.text((text_x, text_y), text, fill=(0,0,0), font=font)
    
    return image



# # Function to perform OCR and draw bounding boxes on the image and a blank canvas
# def inference(image_path, lang=['en']):
#     reader = easyocr.Reader(lang)
#     bounds = reader.readtext(image_path)
#     print(bounds)
    
#     im_with_boxes = Image.open(image_path)
#     draw_boxes(im_with_boxes, bounds, (0, 255, 0), 5)
    
#     # Create a blank canvas with the same size as the original image
#     mask = Image.new('RGB', im_with_boxes.size, (255, 255, 255))
#     draw_mask(mask, bounds, (0, 0, 0), 5, fill_color=(193,193,193))
    
#     return im_with_boxes, mask

# def inference_piped(image, lang=['en'], replace=None):
    
#     image_arr = np.array(image)
#     reader = easyocr.Reader(lang)
#     bounds = reader.readtext(image_arr)
#     print(bounds)
#     im_with_boxes = image.copy()
#     draw_boxes(im_with_boxes, bounds, (0, 255, 0), 5)
    
#     # Create a blank canvas with the same size as the original image
#     mask = Image.new('RGB', image.size, (255, 255, 255))
#     draw_mask(mask, bounds, (0, 0, 0), 5, fill_color=(193,193,193), replace=replace)
#     return im_with_boxes, mask

def stitch(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    stitched = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        stitched.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return stitched


def recortar_imagen(lista_palabras, palabra, img_array, ancho=512, alto=512):
    '''
    Recorta una imagen alrededor de una palabra específica y devuelve las coordenadas relativas.

    Args:
        lista_palabras (list): Lista de palabras detectadas en la imagen. (easyocr output)
        palabra (str): Palabra a recortar.
        img_array (np.array): Array de la imagen.
        ancho (int): Ancho de la imagen recortada.
        alto (int): Alto de la imagen recortada.

    Returns:
        imagen_recortada (np.array): La imagen recortada.
        (x_min, y_min, x_max, y_max) (tuple): Coordenadas relativas a la imagen original.
    '''
    
    altura_img, anchura_img, _ = img_array.shape
    mitad_ancho = ancho // 2
    mitad_alto = alto // 2
    
    for detection in lista_palabras:
        if detection[1] == palabra:
            x1, y1 = detection[0][0]
            x2, y2 = detection[0][2]
            x3, y3 = detection[0][1]
            x4, y4 = detection[0][3]
            x_min = min(x1, x2, x3, x4)
            x_max = max(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            y_max = max(y1, y2, y3, y4)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # Calcular límites
            x_min = int(x_center - mitad_ancho)
            x_max = int(x_center + mitad_ancho)
            y_min = int(y_center - mitad_alto)
            y_max = int(y_center + mitad_alto)
            
            # Asegurar que los límites estén dentro de la imagen
            if x_min < 0:
                x_min = 0
                x_max = min(ancho, anchura_img)
            if x_max > anchura_img:
                x_max = anchura_img
                x_min = max(0, anchura_img - ancho)
            if y_min < 0:
                y_min = 0
                y_max = min(alto, altura_img)
            if y_max > altura_img:
                y_max = altura_img
                y_min = max(0, altura_img - alto)
            
            # Recortar la imagen
            imagen_recortada = img_array[y_min:y_max, x_min:x_max]
            
            # Devolver la imagen recortada y las coordenadas relativas
            return imagen_recortada, (x_min, y_min, x_max, y_max)

    # Si la palabra no se encuentra, retornar None
    return None, None

def calcular_temperatura(imagen):
    # Convertir la imagen a formato LAB
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    # Obtener el promedio de los valores del canal B
    promedio_temperatura = np.mean(imagen_lab[:,:,2]) - np.mean(imagen_lab[:,:,0])
    return promedio_temperatura

def ajustar_temperatura(imagen, ajuste):
    # Convertir la imagen a formato LAB
    imagen_lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    
    # Ajustar el canal B para cambiar la temperatura
    imagen_lab[:,:,2] = cv2.add(imagen_lab[:,:,2], int(ajuste))
    
    # Convertir la imagen de nuevo a formato BGR
    imagen_ajustada = cv2.cvtColor(imagen_lab, cv2.COLOR_LAB2BGR)
    
    # Asegurarse de que los valores estén en el rango de 0 a 255
    imagen_ajustada = np.clip(imagen_ajustada, 0, 255).astype(np.uint8)
    
    return imagen_ajustada
        
def comparar_imagenes(image_path1, image_path2, coordinates, save_path = False, adjust_temp = False):
    """
    Recorta y ajusta la temperatura de dos imágenes basándose en las mismas coordenadas 
    y muestra las imágenes recortadas una al lado de la otra para comparación.

    Args:
    - image_path1 (str): Ruta de la primera imagen.
    - image_path2 (str): Ruta de la segunda imagen.
    - coordinates (list): Lista de 8 valores que representan las coordenadas de la región de interés.
                          [x1, y1, x2, y2, x3, y3, x4, y4]
    - save_path (str, optional): Ruta para guardar las imágenes recortadas. Si no se proporciona, no se guarda nada.
    - adjust_temp (bool, optional): Si es True, ajusta la temperatura de la segunda imagen para igualarla a la primera.
    
    Returns:
    - comparison_image (PIL.Image): Imagen combinada mostrando ambas recortes lado a lado.
    """
    # Cargar imágenes usando OpenCV
    imagen_original = cv2.imread(image_path1)
    imagen_modificada = cv2.imread(image_path2)

    # Extraer coordenadas
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

    # Calcular el rectángulo de recorte
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)

    # Recortar ambas imágenes usando el bounding box
    imagen_original_cropped = imagen_original[top:bottom, left:right]
    imagen_modificada_cropped = imagen_modificada[top:bottom, left:right]

    if adjust_temp:
        # Calcular las temperaturas
        temp_original = calcular_temperatura(imagen_original_cropped)
        temp_modificada = calcular_temperatura(imagen_modificada_cropped)

        # Calcular la diferencia de temperatura
        diferencia_temperatura = temp_original - temp_modificada

        # Ajustar la temperatura de la imagen modificada para igualarla a la original
        imagen_ajustada = ajustar_temperatura(imagen_modificada_cropped, diferencia_temperatura)

        # Convertir imágenes ajustadas a formato PIL para la combinación final
        imagen_original_pil = Image.fromarray(cv2.cvtColor(imagen_original_cropped, cv2.COLOR_BGR2RGB))
        imagen_ajustada_pil = Image.fromarray(cv2.cvtColor(imagen_ajustada, cv2.COLOR_BGR2RGB))
    else:
        # Convertir imágenes recortadas a formato PIL para la combinación final
        imagen_original_pil = Image.fromarray(cv2.cvtColor(imagen_original_cropped, cv2.COLOR_BGR2RGB))
        imagen_ajustada_pil = Image.fromarray(cv2.cvtColor(imagen_modificada_cropped, cv2.COLOR_BGR2RGB))

    # Crear una nueva imagen que combina ambas imágenes recortadas lado a lado
    width, height = imagen_original_pil.size
    comparison_image = Image.new('RGB', (2 * width, height))
    comparison_image.paste(imagen_original_pil, (0, 0))
    comparison_image.paste(imagen_ajustada_pil, (width, 0))

    if save_path:
        imagen_original_pil.save(f"{save_path}/{image_path1.split('/')[-1].split('.')[0]}_cropped.jpg")
        imagen_ajustada_pil.save(f"{save_path}/{image_path2.split('/')[-1].split('.')[0]}_cropped.jpg")
        comparison_image.save(f"{save_path}/comparison.jpg")

    # Devolver la imagen comparativa
    return comparison_image

def reemplazar_parte_imagen(original_image_pil, modified_image_pil, coordinates, adjust_temp=False):
    """
    Reemplaza una parte de la imagen original con una parte de la imagen modificada usando las mismas coordenadas.

    Args:
    - original_image_pil (PIL.Image): Imagen original como objeto PIL.
    - modified_image_pil (PIL.Image): Imagen modificada como objeto PIL.
    - coordinates (list): Lista de 8 valores que representan las coordenadas de la región de interés.
                          [x1, y1, x2, y2, x3, y3, x4, y4]
    - adjust_temp (bool, optional): Si es True, ajusta la temperatura de la parte modificada para igualarla a la original.

    Returns:
    - combined_image (PIL.Image): Imagen resultante con la parte modificada reemplazada en la imagen original.
    """
    # Convertir las imágenes PIL a arrays de numpy para usarlas con OpenCV
    original_image_cv2 = cv2.cvtColor(np.array(original_image_pil), cv2.COLOR_RGB2BGR)
    modified_image_cv2 = cv2.cvtColor(np.array(modified_image_pil), cv2.COLOR_RGB2BGR)

    # Extraer las coordenadas
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

    # Calcular el rectángulo de recorte
    left = min(x1, x2, x3, x4)
    top = min(y1, y2, y3, y4)
    right = max(x1, x2, x3, x4)
    bottom = max(y1, y2, y3, y4)

    # Recortar la parte relevante de ambas imágenes
    original_crop = original_image_cv2[top:bottom, left:right]
    modified_crop = modified_image_cv2[top:bottom, left:right]

    if adjust_temp:
        # Calcular las temperaturas
        temp_original = calcular_temperatura(original_crop)
        temp_modificada = calcular_temperatura(modified_crop)

        # Calcular la diferencia de temperatura
        diferencia_temperatura = temp_original - temp_modificada

        # Ajustar la temperatura de la imagen modificada
        modified_crop_adjusted = ajustar_temperatura(modified_crop, diferencia_temperatura)
    else:
        modified_crop_adjusted = modified_crop

    # Reemplazar la parte de la imagen original con la parte ajustada/modificada
    original_image_cv2[top:bottom, left:right] = modified_crop_adjusted

    # Convertir la imagen de nuevo a formato RGB para PIL
    combined_image = Image.fromarray(cv2.cvtColor(original_image_cv2, cv2.COLOR_BGR2RGB))

    return combined_image

def convert_paddle_to_easyocr(paddle_result):
    """
    Convierte el resultado de PaddleOCR al formato de EasyOCR.

    Args:
        paddle_result (list): La salida de PaddleOCR.

    Returns:
        list: Una lista de tuplas en el formato de EasyOCR.
    """
    # Inicializamos la lista para almacenar los resultados convertidos
    easyocr_result = []

    # Iteramos sobre los bloques de texto en el resultado de PaddleOCR
    for block in paddle_result[0]:
        bbox = block[0]  # Coordenadas de la caja delimitadora
        text = block[1][0]  # El texto reconocido
        confidence = block[1][1]  # El puntaje de confianza

        # Añadimos el resultado en formato EasyOCR a la lista
        easyocr_result.append((bbox, text, confidence))

    return easyocr_result

def recalcular_cuadricula_rotada(cuadricula, palabra_original, palabra_objetivo):
    """
    Recalcula la cuadrícula para acomodar una palabra más grande, aumentando solo el ancho y manteniendo la orientación y la altura.
    
    Parameters:
    cuadricula (list): Lista de coordenadas en formato [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] que representan un rectángulo rotado.
    palabra_original (str): Palabra original en el documento.
    palabra_objetivo (str): Palabra objetivo (nueva palabra).
    
    Returns:
    list: Nueva lista de coordenadas ajustadas para acomodar la palabra más larga.
    """
    
    # Medir la longitud de la palabra original y de la más larga de las palabras objetivo
    longitud_original = len(palabra_original)
    longitud_maxima_objetivo = len(palabra_objetivo) 
    
    # Calcular la proporción de expansión necesaria para el ancho
    expansion_proporcion = longitud_maxima_objetivo / longitud_original
    
    # Extraer las coordenadas originales como puntos (x, y)
    p1 = np.array(cuadricula[0])
    p2 = np.array(cuadricula[1])
    p3 = np.array(cuadricula[2])
    p4 = np.array(cuadricula[3])
    
    # Calcular los vectores de los lados superior e inferior
    vector_lado_superior = p2 - p1
    vector_lado_inferior = p3 - p4
    
    # Calcular la nueva longitud de los lados, expandiendo en la dirección del vector
    nuevo_vector_lado_superior = vector_lado_superior * expansion_proporcion
    nuevo_vector_lado_inferior = vector_lado_inferior * expansion_proporcion
    
    # Recalcular las nuevas coordenadas
    p2_nuevo = p1 + nuevo_vector_lado_superior
    p3_nuevo = p4 + nuevo_vector_lado_inferior
    
    # Construir la nueva cuadrícula con las coordenadas ajustadas
    nueva_cuadricula = [p1.tolist(), p2_nuevo.tolist(), p3_nuevo.tolist(), p4.tolist()]
    
    return nueva_cuadricula

def juntar_imagenes_vertical(imagen_upper, imagen_lower):
    '''
    Combina dos imágenes verticalmente (una encima de otra).

    Args:
        imagen_upper (PIL.Image): Imagen que se colocará en la parte superior.
        imagen_lower (PIL.Image): Imagen que se colocará en la parte inferior.

    Returns:
        imagen_combined (PIL.Image): Imagen resultante de combinar ambas imágenes verticalmente.
    '''
    
    # Obtener dimensiones de las imágenes
    ancho_upper, alto_upper = imagen_upper.size
    ancho_lower, alto_lower = imagen_lower.size
    
    # Definir el ancho y el alto de la imagen combinada
    ancho_combined = max(ancho_upper, ancho_lower)
    alto_combined = alto_upper + alto_lower
    
    # Crear una nueva imagen con el tamaño combinado
    imagen_combined = Image.new('RGB', (ancho_combined, alto_combined))
    
    # Pegar la imagen superior
    imagen_combined.paste(imagen_upper, (0, 0))
    
    # Pegar la imagen inferior debajo de la superior
    imagen_combined.paste(imagen_lower, (0, alto_upper))
    
    return imagen_combined

def rellenar_imagen_uniformemente(imagen_pil, dimensiones_objetivo, color_relleno=(255, 255, 255)):
    """
    Rellena una imagen para alcanzar una dimensión específica de manera uniforme por los bordes.

    Args:
    - imagen_pil (PIL.Image): Imagen de entrada como objeto PIL.
    - dimensiones_objetivo (tuple): Dimensiones deseadas en formato (ancho, alto).
    - color_relleno (tuple): Color de relleno en formato RGB. Por defecto es blanco (255, 255, 255).

    Returns:
    - imagen_rellena (PIL.Image): Imagen rellenada con las dimensiones especificadas.
    """
    
    # Obtener dimensiones de la imagen original
    ancho_original, alto_original = imagen_pil.size
    ancho_objetivo, alto_objetivo = dimensiones_objetivo
    
    # Calcular los márgenes que se necesitan para centrar la imagen
    margen_izquierdo = (ancho_objetivo - ancho_original) // 2
    margen_superior = (alto_objetivo - alto_original) // 2
    margen_derecho = ancho_objetivo - ancho_original - margen_izquierdo
    margen_inferior = alto_objetivo - alto_original - margen_superior
    
    # Añadir márgenes para rellenar la imagen
    imagen_rellena = ImageOps.expand(imagen_pil, border=(margen_izquierdo, margen_superior, margen_derecho, margen_inferior), fill=color_relleno)
    
    return imagen_rellena

def recortar_imagen_uniformemente(imagen_pil, color_relleno=(255, 255, 255)):
    """
    Recorta los bordes blancos de una imagen, eliminando el relleno de color blanco (u otro color especificado).

    Args:
    - imagen_pil (PIL.Image): Imagen de entrada como objeto PIL.
    - color_relleno (tuple): Color de relleno en formato RGB que se desea recortar. Por defecto es blanco (255, 255, 255).

    Returns:
    - imagen_recortada (PIL.Image): Imagen sin los bordes blancos.
    - box (tuple): Coordenadas del recorte en formato (left, upper, right, lower).
    """
    
    # Convertir la imagen a un array de numpy
    imagen_np = np.array(imagen_pil)
    
    # Crear una máscara donde los píxeles que no son del color de relleno se marquen como True
    mask = np.any(imagen_np != color_relleno, axis=-1)
    
    # Encontrar los límites de los píxeles no blancos
    if mask.any():
        y_min, y_max = np.where(mask)[0][[0, -1]]
        x_min, x_max = np.where(mask)[1][[0, -1]]
        
        # Recortar la imagen usando estos límites
        imagen_recortada = imagen_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
        
        # Retornar la imagen recortada y las coordenadas del recorte
        return imagen_recortada, (x_min, y_min, x_max + 1, y_max + 1)
    else:
        # Si toda la imagen es del color de relleno, retornar la imagen original
        return imagen_pil, (0, 0, imagen_pil.width, imagen_pil.height)
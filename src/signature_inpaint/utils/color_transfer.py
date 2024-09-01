# color_transfer.py
import cv2
import numpy as np

def transfer_color(original_lab, template_lab, mask_original_np, mask_template_np):
    """
    Transfiere el color de la firma original al template en el espacio de color LAB.

    Args:
        original_lab (numpy.ndarray): Imagen original en espacio LAB.
        template_lab (numpy.ndarray): Imagen template en espacio LAB.
        mask_original_np (numpy.ndarray): Máscara de la firma original.
        mask_template_np (numpy.ndarray): Máscara del template.

    Returns:
        numpy.ndarray: Template con el color transferido en espacio RGB.
    """
    for i in range(3):
        mean_i = np.mean(original_lab[:,:,i][mask_original_np > 0])
        std_i = np.std(original_lab[:,:,i][mask_original_np > 0])
        template_lab[:,:,i] = np.where(
            mask_template_np > 0,
            (template_lab[:,:,i] - np.mean(template_lab[:,:,i][mask_template_np > 0])) / np.std(template_lab[:,:,i][mask_template_np > 0]) * std_i + mean_i,
            template_lab[:,:,i]
        )
    
    colored_template = cv2.cvtColor(template_lab, cv2.COLOR_LAB2RGB)
    return colored_template
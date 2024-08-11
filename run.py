from td_inpaint import inpaint
from inpaint_functions import parse_bounds

def simple_inpaint(image, bounds, word, slider_step=25, slider_guidance=2.5, slider_batch=4):
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
    slider_step = 25
    slider_guidance= 2.5
    slider_batch= 4
    slider_natural= False
    return inpaint(image, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_natural, global_dict)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    combined_images = []  # List to store filenames of images with masks overlaid
    mask_images = []      # List to store filenames of separate mask images

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # ---- Original Image with Mask Overlaid ----
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)  # Draw the mask with borders
        """
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        """
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')

        # Save the figure as a JPG file
        combined_filename = f"combined_image_{i+1}.jpg"
        plt.savefig(combined_filename, format='jpg', bbox_inches='tight')
        combined_images.append(combined_filename)

        plt.close()  # Close the figure to free up memory

        # ---- Separate Mask Image (White Mask on Black Background) ----
        # Create a black image
        mask_image = np.zeros_like(image, dtype=np.uint8)
        
        # The mask is a binary array where the masked area is 1, else 0.
        # Convert the mask to a white color in the mask_image
        mask_layer = (mask > 0).astype(np.uint8) * 255
        for c in range(3):  # Assuming RGB, repeat mask for all channels
            mask_image[:, :, c] = mask_layer

        # Save the mask image
        mask_filename = f"mask_image_{i+1}.png"
        Image.fromarray(mask_image).save(mask_filename)
        mask_images.append(mask_filename)

        plt.close()  # Close the figure to free up memory

    return combined_images, mask_images
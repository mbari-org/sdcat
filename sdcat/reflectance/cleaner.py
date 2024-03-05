import time

import cv2
import numpy as np

from sdcat.logger import info
from sdcat.reflectance import specularity as spc


def specularity_removal(image: np.ndarray) -> np.ndarray:
    """
    Remove specular highlights from an image. This is done by inpainting the specular highlights.
    Caution: This method is slow for large images. Be sure to resize the image before using this method.
    :param image: color image to remove specular highlights from in RGB format
    :return: cleaned image
    """
    start_time = time.time()
    # Radius for inpainting should be smaller for smaller images, but max out at 30
    radius = min(int(min(image.shape[:2]) * .01), 30)
    info(f'Specularity removal using radius {radius} for inpainting')

    img_color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    r_img = np.array(img_gray)
    rimg = spc.derive_m(img_color, r_img)
    # TODO: debug overflows by changing the dtype to uint16 and scaling the values
    s_img = spc.derive_saturation(img_color, rimg)
    spec_mask = spc.check_pixel_specularity(rimg, s_img)
    enlarged_spec = spc.enlarge_specularity(spec_mask)
    # Use inpaint methods to remove specularity
    telea = cv2.inpaint(img_color, enlarged_spec, radius, cv2.INPAINT_TELEA)

    end_time = time.time()
    info(f"Specularity removal took {end_time - start_time:.2f} seconds")
    clean_img = cv2.cvtColor(telea, cv2.COLOR_BGR2RGB)
    return clean_img

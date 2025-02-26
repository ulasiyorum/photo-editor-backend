import cv2
import numpy as np

def apply_sketch_filter(image: np.ndarray, intensity: int) -> np.ndarray:

    if not 0 <= intensity <= 100:
        raise ValueError("intensity should be between 0 and 100")

    if intensity == 0:
        return image

    intensity = intensity / 100
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sketch_lines = cv2.adaptiveThreshold(grayscale, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 9, 2)

    inverted = 255 - grayscale

    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=10, sigmaY=10)

    sketch_shading = cv2.divide(grayscale, 255 - blurred, scale=256)

    result = cv2.bitwise_and(sketch_shading, sketch_lines)
    result = result.astype('uint8')

    if intensity == 1:
        return result

    result = (image * (1 - intensity) + (result * intensity)).astype('uint8')
    return result

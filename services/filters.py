import cv2
import numpy as np

def apply_sketch_filter(image: np.ndarray, intensity: int) -> np.ndarray:

    if not 0 <= intensity <= 100:
        raise ValueError("intensity should be between 0 and 100")

    if intensity == 0:
        return image

    intensity = intensity / 100
    grayscale = np.array(np.dot(image[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8)
    grayscale = np.stack((grayscale,) * 3, axis=-1)
    inverted = 255 - grayscale
    blurred = cv2.GaussianBlur(inverted, ksize=(0, 0), sigmaX=5)
    result = grayscale * 255.0 / (255.0 - blurred)
    result[result > 255] = 255
    result[grayscale == 255] = 255

    result = result.astype('uint8')

    if intensity == 1:
        return result

    result = (image * (1 - intensity) + (result * intensity)).astype('uint8')
    return result

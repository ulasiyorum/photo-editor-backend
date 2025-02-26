import cv2
import numpy as np

def apply_sketch_filter(image: np.ndarray, intensity: int) -> np.ndarray:


    paper_texture = cv2.imread('services/texture/paper_texture.jpg', cv2.IMREAD_GRAYSCALE)
    if not 0 <= intensity <= 100:
        raise ValueError("intensity should be between 0 and 100")

    if intensity == 0:
        return image

    intensity = intensity / 100
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sketch_lines = cv2.adaptiveThreshold(grayscale, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 9, 5)

    inverted = 255 - grayscale

    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=10, sigmaY=10)

    sketch_shading = cv2.divide(grayscale, 255 - blurred, scale=256)

    result = cv2.bitwise_and(sketch_shading, sketch_lines)

    paper_texture = cv2.resize(paper_texture, (result.shape[1], result.shape[0]))

    result = cv2.multiply(result, paper_texture, scale=1 / 255.0)

    result = result.astype('uint8')

    if intensity == 1:
        return result

    result = (image * (1 - intensity) + (result * intensity)).astype('uint8')
    return result

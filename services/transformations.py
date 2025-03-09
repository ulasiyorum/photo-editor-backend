import cv2
import numpy as np


def rotate(image: np.ndarray, angle: int) -> np.ndarray:

    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE)
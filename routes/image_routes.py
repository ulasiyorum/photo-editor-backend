from fastapi import APIRouter, UploadFile, File, Response
from services.filters import apply_sketch_filter
from PIL import Image
import cv2
import io
import numpy as np

router = APIRouter(prefix="/image", tags=["Image Processing"])

@router.post("/sketch")
async def sketch_image(file: UploadFile = File(...), intensity: int = 50):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_format = image.format
    image = np.array(image)

    sketch = apply_sketch_filter(image, intensity)
    correct_color = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
    _, encoded_img = cv2.imencode('.' + image_format, correct_color)
    return Response(content=encoded_img.tobytes(), media_type=file.content_type)

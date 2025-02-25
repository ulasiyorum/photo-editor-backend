from fastapi import APIRouter, UploadFile, File
from services.filters import apply_sketch_filter
from PIL import Image
import io
import numpy as np

router = APIRouter(prefix="/image", tags=["Image Processing"])

@router.post("/sketch")
async def sketch_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    # Çizim efektini uygula
    sketch = apply_sketch_filter(image)

    # Görseli PNG formatında geri döndür
    _, encoded_img = cv2.imencode('.png', sketch)
    return {"filename": file.filename, "image": encoded_img.tobytes()}

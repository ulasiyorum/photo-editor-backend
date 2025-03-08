from fastapi import FastAPI
from routes.image_routes import router as image_router
app = FastAPI(title="Photo Editor API", version="1.0")

app.include_router(image_router)
@app.get("/")
def read_root ():
    return {"message": "Photo Editor API is running 🚀"}

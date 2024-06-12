# app/main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from app.model import ImageChatModel

# Initialize the model
model = ImageChatModel()

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...), question: str = Form(...)):
    temp_file_path = Path("/tmp") / file.filename
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    response = model.chat(str(temp_file_path), question)
    
    # Clean up temp file
    temp_file_path.unlink()

    return JSONResponse(content={"response": response})

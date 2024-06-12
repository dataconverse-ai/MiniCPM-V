# app/main.py

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
import argparse
from app.model import ImageChatModel

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
args, _ = parser.parse_known_args()

# Initialize the model with the specified device
model = ImageChatModel(device=args.device)

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

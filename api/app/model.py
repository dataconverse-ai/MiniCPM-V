# app/model.py

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class ImageChatModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, low_cpu_mem_usage=True)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        self.model.eval()

    def chat(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': question}]
        response = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return response

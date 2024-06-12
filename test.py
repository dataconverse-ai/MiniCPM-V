# test.py  Need more than 16GB memory.
# PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, low_cpu_mem_usage=True)
model = model.to(device='mps')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open('./assets/hk_OCR.jpg').convert('RGB')
question = 'Where is this photo taken?'
msgs = [{'role': 'user', 'content': question}]

# answer, context, _ = model.chat(
#     image=image,
#     msgs=msgs,
#     context=None,
#     tokenizer=tokenizer,
#     sampling=True
# )

# print(answer)

image_path = './assets/hk_OCR.jpg'

response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "guess what I am doing?"
        }
    ],
    tokenizer=tokenizer
)

print(response)
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPTokenizer, CLIPModel
import torch

app = FastAPI()

# Load CLIP model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

class Query(BaseModel):
    text: str

@app.post("/embed")
def embed(query: Query):
    inputs = tokenizer([query.text], return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)[0]
    return {"embedding": embedding.tolist()}

#!/usr/bin/env python3
"""Quick VRAM smoke test: load gemma-4 2B in 4-bit without device_map='auto'."""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

base_model = "google/gemma-4-E2B-it"

compute_dtype = torch.float16  # GTX 1650 Ti es SM 7.5, no soporta bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

print("Cargando modelo base en 4-bit directo a cuda:0...")
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    device_map={"": 0},  # fuerza TODO al GPU 0, sin consultarle a accelerate
)

print(f"Modelo cargado en: {model.device}")
print(f"VRAM usada: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
print(f"VRAM reservada: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")

# Generar un token de prueba
prompt = "Responde en español: ¿Qué es la preeclampsia?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32)
print("Generación OK:", tokenizer.decode(out[0], skip_special_tokens=True)[:200])

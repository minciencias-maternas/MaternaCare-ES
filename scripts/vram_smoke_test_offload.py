#!/usr/bin/env python3
"""VRAM smoke test with CPU offloading for tight GPUs."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

base_model = "google/gemma-4-E2B-it"
compute_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

print("Cargando modelo con device_map='auto' (permite CPU offloading)...")
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    device_map="auto",  # deja que accelerate decida GPU/CPU
    offload_buffers=True,
    max_memory={0: "3500MiB", "cpu": "12GiB"},  # limitar GPU a 3.5GB
)

print(f"Modelo cargado. Dispositivos usados: {set(p.device for p in model.parameters())}")
print(f"VRAM usada: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

prompt = "Responde en español: ¿Qué es la preeclampsia?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32)
print("Generación OK:", tokenizer.decode(out[0], skip_special_tokens=True)[:200])

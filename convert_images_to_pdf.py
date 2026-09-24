#!/usr/bin/env python3
"""Convert PNG images in a directory to individual PDFs for LaTeX inclusion."""

import os
from pathlib import Path
from PIL import Image

ASSETS_DIR = Path("papers/assets/img")
OUTPUT_DIR = ASSETS_DIR / "pdf"

def convert_images_to_pdf():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in ASSETS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )

    if not image_paths:
        print("No se encontraron imágenes para convertir.")
        return

    print(f"Convirtiendo {len(image_paths)} imagen(es) a PDF...")
    for img_path in image_paths:
        # Preserve the base name but replace final extension with .pdf
        base = img_path.stem
        # Handle double extensions like .png.png
        if base.endswith(img_path.suffix.lower().lstrip(".")):
            base = base[: -len(img_path.suffix.lower().lstrip("."))]
        out_path = OUTPUT_DIR / f"{base}.pdf"

        try:
            img = Image.open(img_path)
            # Convert palette / RGBA images to RGB for PDF compatibility
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(out_path, "PDF", resolution=100.0)
            print(f"  {img_path.name} -> {out_path}")
        except Exception as e:
            print(f"  ERROR convirtiendo {img_path.name}: {e}")

    print(f"\nPDFs guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_images_to_pdf()

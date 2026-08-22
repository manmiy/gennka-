"""
PDF処理モジュール
"""
import io
from typing import List, Tuple
import fitz
from PIL import Image
def pdf_to_images(pdf_bytes, dpi=200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        images.append((page_num + 1, img))
    doc.close()
    return images
def image_to_bytes(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()
def load_image_file(file_bytes):
    return Image.open(io.BytesIO(file_bytes))

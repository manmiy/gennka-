"""
ファイル・画像処理モジュール（PDF / TIFF / JPEG / PNG等）
"""

import io
from typing import List, Tuple
import fitz  # PyMuPDF
from PIL import Image, ImageSequence


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[int, Image.Image]]:
    """
    PDFを各ページのPIL Imageに変換する

    Args:
        pdf_bytes: PDFバイトデータ
        dpi: 解像度

    Returns:
        (ページ番号, PIL Image) のリスト
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append((page_num + 1, img))
    doc.close()
    return images


def tiff_to_images(tiff_bytes: bytes) -> List[Tuple[int, Image.Image]]:
    """
    マルチページTIFFを含むTIFF画像を各ページのPIL Imageに変換する

    Args:
        tiff_bytes: TIFFバイトデータ

    Returns:
        (ページ番号, PIL Image) のリスト
    """
    img = Image.open(io.BytesIO(tiff_bytes))
    images = []
    page_num = 1
    for frame in ImageSequence.Iterator(img):
        images.append((page_num, frame.convert("RGB").copy()))
        page_num += 1
    return images


def load_image_file(file_bytes: bytes) -> Image.Image:
    """
    単一画像（JPG, JPEG, PNG等）をPIL Imageとして読み込みRGB変換する

    Args:
        file_bytes: 画像バイトデータ

    Returns:
        PIL Image
    """
    img = Image.open(io.BytesIO(file_bytes))
    return img.convert("RGB")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    PIL Imageをバイトデータに変換する
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()

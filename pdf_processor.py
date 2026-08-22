"""
PDF処理モジュール
PDFファイルを画像に変換し、OCR処理に渡す
"""

import io
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[int, Image.Image]]:
    """
    PDFの各ページを画像に変換する

    Args:
        pdf_bytes: PDFファイルのバイトデータ
        dpi: 出力画像の解像度（デフォルト200dpi）

    Returns:
        (ページ番号, PIL Image) のリスト
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    zoom = dpi / 72  # 72dpiがデフォルト
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        # PyMuPDFのPixmapをPIL Imageに変換
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        images.append((page_num + 1, img))

    doc.close()
    return images


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    PIL ImageをバイトデータにPNG変換する

    Args:
        image: PIL Image
        format: 画像フォーマット

    Returns:
        画像のバイトデータ
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def load_image_file(file_bytes: bytes) -> Image.Image:
    """
    画像ファイルのバイトデータからPIL Imageを作成する

    Args:
        file_bytes: 画像ファイルのバイトデータ

    Returns:
        PIL Image
    """
    return Image.open(io.BytesIO(file_bytes))

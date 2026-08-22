"""
OCRエンジンモジュール
Google Cloud Vertex AI (Gemini) を使って請求書画像からデータを抽出する
"""
import json
import re
from typing import Any, Dict, List, Optional
from google.oauth2 import service_account
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel, Part
# Geminiに送る請求書OCR用のプロンプト
INVOICE_OCR_PROMPT = """あなたは日本の建設資材の請求書を読み取る専門のOCRシステムです。
この画像は「請求内訳明細表」または「御請求書」です。
## 画像が「請求内訳明細表」（明細テーブルがある）場合:
テーブルの各明細行を読み取り、以下のJSON形式で出力してください。
「伝票合計」「件名合計」などの合計行は除外し、個別の品目行のみを抽出してください。
```json
{
  "type": "明細表",
  "supplier": "請求元会社名",
  "customer": "請求先会社名",
  "date": "請求日（令和X年X月X日 → YYYY/MM/DD形式に変換）",
  "items": [
    {
      "month": "月（数値）",
      "day": "日（数値）",
      "slip_no": "伝票No",
      "maker": "メーカ名",
      "product_code": "品番",
      "product_name": "品名",
      "quantity": "数量（数値）",
      "unit": "単位（枚、本、式、台など）",
      "unit_price": "単価（数値）",
      "amount": "金額（数値）",
      "order_no": "注文No",
      "remarks": "備考"
    }
  ]
}
```
## 画像が「御請求書」（サマリーページ）の場合:
以下のJSON形式で出力してください。
```json
{
  "type": "請求書サマリー",
  "supplier": "請求元会社名",
  "customer": "請求先会社名",

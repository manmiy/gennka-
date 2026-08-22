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
  "date": "請求日（YYYY/MM/DD形式）",
  "billing_period": "請求期間",
  "total_amount": "御請求額（数値）",
  "subtotal": "当月御買上額（数値）",
  "tax": "消費税額（数値）",
  "items": []
}
```

## 重要な注意事項:
- 数値はカンマなしの数字のみで出力してください（例: 6300、-30）
- 負の数値はマイナス記号をつけてください（例: -30）
- 空欄のセルは空文字列 "" にしてください
- 必ず有効なJSONのみを出力してください（説明文は不要）
- 「伝票合計」「件名合計」の行はitemsに含めないでください
- 同じ伝票Noの中に複数の品目がある場合、各品目を個別の行として出力してください
"""


def init_vertex_ai(
    credentials_json: dict,
    project_id: str,
    location: str = "asia-northeast1",
) -> None:
    """
    Vertex AIを初期化する

    Args:
        credentials_json: サービスアカウントJSONの辞書
        project_id: Google CloudプロジェクトID
        location: Vertex AIリージョン
    """
    credentials = service_account.Credentials.from_service_account_info(
        credentials_json,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    vertexai.init(
        project=project_id,
        location=location,
        credentials=credentials,
    )


def extract_invoice_data(image: Image.Image) -> Dict[str, Any]:
    """
    画像から請求書データを抽出する

    Args:
        image: 請求書画像（PIL Image）

    Returns:
        抽出されたデータの辞書
    """
    import io

    # PIL ImageをバイトデータとしてPartに変換
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    image_part = Part.from_data(data=image_bytes, mime_type="image/png")

    model = GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        [image_part, INVOICE_OCR_PROMPT],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 8192,
        },
    )

    # レスポンスからJSONを抽出
    response_text = response.text.strip()
    return _parse_json_response(response_text)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Geminiの応答テキストからJSONを抽出・パースする

    Args:
        text: Geminiの応答テキスト

    Returns:
        パースされたJSON辞書
    """
    # ```json ... ``` ブロックを探す
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # ブロックがない場合、テキスト全体をJSONとして試す
        json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # JSONパースに失敗した場合、エラー情報を含めて返す
        return {
            "type": "エラー",
            "error": f"JSONパースエラー: {str(e)}",
            "raw_text": text,
            "items": [],
        }


def process_multiple_images(
    images: List[Image.Image],
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    複数の画像を処理してデータを抽出する

    Args:
        images: 画像リスト
        progress_callback: 進捗コールバック関数 (current, total)

    Returns:
        各画像の抽出結果リスト
    """
    results = []
    total = len(images)

    for i, img in enumerate(images):
        if progress_callback:
            progress_callback(i, total)

        result = extract_invoice_data(img)
        results.append(result)

    if progress_callback:
        progress_callback(total, total)

    return results


def merge_items(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    複数の結果からすべての明細アイテムを統合する
    サマリーページはスキップし、明細表のitemsのみ収集する

    Args:
        results: 各画像の抽出結果リスト

    Returns:
        統合された全明細アイテムリスト
    """
    all_items = []
    for result in results:
        if result.get("type") == "明細表":
            items = result.get("items", [])
            for item in items:
                all_items.append(item)
    return all_items


def aggregate_by_product(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同じ品名のアイテムの数量と金額を集約する

    Args:
        items: 明細アイテムリスト

    Returns:
        集約された明細アイテムリスト
    """
    aggregated = {}

    for item in items:
        product_name = item.get("product_name", "")
        product_code = item.get("product_code", "")
        key = f"{product_name}_{product_code}"

        if key in aggregated:
            # 数量と金額を加算
            try:
                existing_qty = _to_number(aggregated[key].get("quantity", 0))
                new_qty = _to_number(item.get("quantity", 0))
                aggregated[key]["quantity"] = existing_qty + new_qty
            except (ValueError, TypeError):
                pass

            try:
                existing_amt = _to_number(aggregated[key].get("amount", 0))
                new_amt = _to_number(item.get("amount", 0))
                aggregated[key]["amount"] = existing_amt + new_amt
            except (ValueError, TypeError):
                pass
        else:
            # 新しいアイテムをコピーして追加
            aggregated[key] = dict(item)
            # 数値に変換
            try:
                aggregated[key]["quantity"] = _to_number(item.get("quantity", 0))
            except (ValueError, TypeError):
                pass
            try:
                aggregated[key]["amount"] = _to_number(item.get("amount", 0))
            except (ValueError, TypeError):
                pass

    return list(aggregated.values())


def _to_number(value) -> float:
    """
    値を数値に変換する

    Args:
        value: 変換する値

    Returns:
        数値
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # カンマを除去して数値変換
        cleaned = value.replace(",", "").replace("，", "").strip()
        if cleaned == "" or cleaned == "-":
            return 0.0
        return float(cleaned)
    return 0.0

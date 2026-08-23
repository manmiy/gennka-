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

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Geminiに送る請求書OCR用のプロンプト
INVOICE_OCR_PROMPT = """あなたは日本の建設資材の請求書を読み取る専門のOCRシステムです。
この画像は「請求内訳明細表」または「御請求書」です。

## 画像が「請求内訳明細表」（明細テーブルがある）場合:

テーブルの各行（個別の品目明細行、および「件名合計」行）を上から出現順に読み取り、以下のJSON形式で出力してください。

【重要な規則】
1. 個別の品目行は `row_type: "item"` として抽出してください。
2. 「伝票合計」行や「得意先合計」行は除外してください。
3. 「件名合計」の行は必ず `row_type: "project_total"` として抽出してください。
   「件名合計」行には備考欄や合計の横に現場名・工事名（例: 「白石送電倉庫新築工事」「仙南個展分」「石井様邸離れ新築工事」など）が記載されています。これを `project_name` に正確に抽出してください。
4. 同じ伝票Noの中に複数の品目がある場合、各品目を個別の行として出力してください。

```json
{
  "type": "明細表",
  "supplier": "請求元会社名",
  "customer": "請求先会社名",
  "date": "請求日（令和X年X月X日 → YYYY/MM/DD形式に変換）",
  "rows": [
    {
      "row_type": "item",
      "month": "月（数値文字列）",
      "day": "日（数値文字列）",
      "slip_no": "伝票No",
      "maker": "メーカ名",
      "product_code": "品番",
      "product_name": "品名",
      "quantity": "数量（数値）",
      "unit": "単位（枚、本、式、台、袋、箱など）",
      "unit_price": "単価（数値）",
      "amount": "金額（数値）",
      "order_no": "注文No",
      "remarks": "備考"
    },
    {
      "row_type": "project_total",
      "amount": "件名合計金額（数値）",
      "order_no": "注文No（例: 250901）",
      "project_name": "現場名・工事名（例: 白石送電倉庫新築工事）",
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
  "rows": []
}
```

## 重要な注意事項:
- 数値はカンマなしの数字のみで出力してください（例: 6300、-30）
- 負の数値はマイナス記号をつけてください（例: -30）
- 空欄のセルは空文字列 "" にしてください
- 必ず有効なJSONのみを出力してください（説明文や余計なテキストは含めないでください）
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


def extract_invoice_data(
    image: Image.Image, model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """
    画像から請求書データを抽出する

    Args:
        image: 請求書画像（PIL Image）
        model_name: 使用するGeminiモデル名

    Returns:
        抽出されたデータの辞書
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    image_part = Part.from_data(data=image_bytes, mime_type="image/png")

    model = GenerativeModel(model_name)

    response = model.generate_content(
        [image_part, INVOICE_OCR_PROMPT],
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 8192,
        },
    )

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
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = text

    try:
        data = json.loads(json_str)
        # rows がなくて items がある場合の互換性対応
        if "rows" not in data and "items" in data:
            data["rows"] = data["items"]
        return data
    except json.JSONDecodeError as e:
        return {
            "type": "エラー",
            "error": f"JSONパースエラー: {str(e)}",
            "raw_text": text,
            "rows": [],
            "items": [],
        }


def process_multiple_images(
    images: List[Image.Image],
    model_name: str = DEFAULT_MODEL_NAME,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    複数の画像を処理してデータを抽出する

    Args:
        images: 画像リスト
        model_name: 使用するGeminiモデル名
        progress_callback: 進捗コールバック関数 (current, total)

    Returns:
        各画像の抽出結果リスト
    """
    results = []
    total = len(images)

    for i, img in enumerate(images):
        if progress_callback:
            progress_callback(i, total)

        result = extract_invoice_data(img, model_name=model_name)
        results.append(result)

    if progress_callback:
        progress_callback(total, total)

    return results


def merge_items(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    複数の結果からすべての明細アイテムを統合し、
    各「件名合計」行の現場名（工事名）を、それより上に並んでいる明細
    （前回の件名合計の次の行から、今回の件名合計の行まで）に適用する。

    Args:
        results: 各画像の抽出結果リスト（ページ順）

    Returns:
        現場名（order_location）が適用された明細アイテムリスト
    """
    all_rows = []

    for result in results:
        if result.get("type") == "明細表":
            rows = result.get("rows", [])
            if not rows and "items" in result:
                rows = result.get("items", [])
            for r in rows:
                all_rows.append(r)

    final_items = []
    current_block = []

    for r in all_rows:
        row_type = r.get("row_type", "item")

        # 明細アイテム行の場合
        if row_type == "item":
            current_block.append(r)

        # 件名合計行の場合: 直前のブロックの全明細にこの現場名を設定
        elif row_type in ("project_total", "subject_total"):
            project_name = (
                r.get("project_name", "")
                or r.get("order_location", "")
                or r.get("remarks", "")
            )
            order_no = r.get("order_no", "")

            for item in current_block:
                item["order_location"] = project_name
                # 注文Noが未設定の場合は件名合計の注文Noをセット
                if not item.get("order_no") and order_no:
                    item["order_no"] = order_no
                final_items.append(item)

            current_block = []

    # 末尾に残ったブロック（もし最後の件名合計がなかった場合）
    for item in current_block:
        if "order_location" not in item:
            item["order_location"] = ""
        final_items.append(item)

    return final_items


def aggregate_by_product(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同じ発注場所（現場名）・同じ品名のアイテムの数量と金額を集約する

    Args:
        items: 明細アイテムリスト

    Returns:
        集約された明細アイテムリスト
    """
    aggregated = {}

    for item in items:
        product_name = item.get("product_name", "")
        product_code = item.get("product_code", "")
        order_location = item.get("order_location", "")
        key = f"{order_location}_{product_name}_{product_code}"

        if key in aggregated:
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
            aggregated[key] = dict(item)
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
        cleaned = value.replace(",", "").replace("，", "").strip()
        if cleaned == "" or cleaned == "-":
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

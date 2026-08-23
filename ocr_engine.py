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

# Geminiに送る汎用請求書・納品書・見積書OCRプロンプト
INVOICE_OCR_PROMPT = """あなたは日本の建設業・工務店向けの資材・建材・金物・設備・照明・インテリア等の各種請求書、納品書、見積書を正確に読み取る専門のAI OCRシステムです。

この画像から品名・仕様・数量・単価・金額・現場名などの明細データを抽出し、以下の統一JSON形式で出力してください。

--------------------------------------------------
【最重要：明細抽出のルール】
- 画像内に1行でも品名や商品（資材・金物・工事等）が書かれている場合は、たとえ用紙タイトルが「請求書」「御請求書」「納品書」「御見積書」であっても、**必ず `type: "明細表"` とし、すべての明細行を `rows` に抽出してください**。
- `type: "請求書サマリー"` にするのは、「前月請求額」「当月買上額」「消費税額」などの合計金額表しかなく、**具体的な商品の明細行が1行も存在しない表紙ページのみ**です。

--------------------------------------------------
【各フォーマットにおける抽出ガイドライン】

1. 三浦金物商会などの手書き伝票フォーマット（重要）:
   - 用紙上部に「請 求 書」と合計金額があっても、下部に手書きの品名が書かれています。**必ず `type: "明細表"` として各手書き明細行を抽出してください**。
   - 左下や枠外・備考にある「現場名」（例: 「石井様邸分」「春日様」「白石送電様分」「白石送電柱事務所」など）を抽出し、各明細の `order_location` に設定してください。
   - 手書きの「品名」「仕様」「数量」「単位」「単価」「金額」を正確に読み取ってください。
     - 「〃」や「”」（同上記号）は上の行の品名・仕様を補完してください。
     - 斜線で消されている空行は無視してください。

2. 金長商事などの請求書フォーマット:
   - 「商品名・入金区分」列において、「普通預金本社」などの入金・相殺行は除外してください（商品の売上明細行のみ抽出）。
   - 現場名（order_location）は、「ご直送先名」列（例: 白石送電事務所新築工事、石井様邸新築工事）から取得してください。
   - 「ご直送先名」が自社宛（(有)佐幸建築店など）の場合は、伝票内の「備考」欄（例: 「白石送電事務所様」「石井様邸離れ新築工事」「高橋様邸」）に記載された現場名を採用してください。

3. 日本ライティングなどの請求書フォーマット:
   - 表の右端にある「件名」列（例: 白石送電事務所）を `order_location` に設定してください。
   - 品番・品名、数量、単価、金額を抽出してください。

4. ルームワンなどのインテリア・カーテン見積書フォーマット:
   - 上部に「現場名: ○○ 様邸」がある場合は、その現場名をすべての明細行の `order_location` に設定してください。
   - 販売価格（税別）の「単価」「金額」を抽出してください。「取付施工費」「諸経費」等の行も抽出してください。
   - 部屋名や窓サイズ・仕様（例: 「寝室(畳敷き) 1F 腰窓 1400×1200」など）を `spec` に設定してください。

5. 岡田電気産業などの請求内訳明細表フォーマット:
   - 個別の品目行は `row_type: "item"` として抽出。
   - 「件名合計」行は `row_type: "project_total"` として抽出し、`project_name` に現場名（例: 白石送電倉庫新築工事、石井様邸離れ新築工事）を設定。
   - 「伝票合計」「得意先合計」行は除外。

--------------------------------------------------
【出力JSONスキーマ】
```json
{
  "type": "明細表",
  "supplier": "請求元・発行元会社名（例: 株式会社三浦金物商会、株式会社金長商事、株式会社日本ライティング、岡田電気産業株式会社など）",
  "customer": "請求先・宛先会社名（例: 有限会社佐幸建築店）",
  "date": "請求日または伝票日（YYYY/MM/DD形式）",
  "order_location": "全体共通の現場名・工事名（もしあれば）",
  "rows": [
    {
      "row_type": "item",
      "month": "月（数値文字列）",
      "day": "日（数値文字列）",
      "slip_no": "伝票No",
      "maker": "メーカ名",
      "product_code": "品番・コード",
      "product_name": "品名・商品名",
      "spec": "仕様・サイズ・規格",
      "quantity": "数量（数値）",
      "unit": "単位（枚、本、式、台、袋、箱、巻、缶、ケ、ケース、P、窓、個、束など）",
      "unit_price": "単価（数値）",
      "amount": "金額（数値）",
      "order_no": "注文No",
      "remarks": "備考",
      "order_location": "現場名・工事名・直送先名（例: 石井様邸分、春日様など）"
    },
    {
      "row_type": "project_total",
      "amount": "件名合計金額（数値）",
      "order_no": "注文No",
      "project_name": "現場名・工事名",
      "remarks": "備考"
    }
  ]
}
```

## 商品明細が一切ない「表紙サマリーのみのページ」の場合のみ:
```json
{
  "type": "請求書サマリー",
  "supplier": "請求元会社名",
  "customer": "請求先会社名",
  "date": "請求日（YYYY/MM/DD形式）",
  "total_amount": "御請求額（数値）",
  "rows": []
}
```

## 重要な注意事項:
- 明細が1行でもあれば必ず `type: "明細表"` とし、`rows` に明細行を入れてください。
- 数値はカンマなしの数字のみで出力してください（例: 6300、-30）
- 負の数値はマイナス記号をつけてください（例: -30）
- 空欄のセルは空文字列 "" にしてください
- 必ず有効なJSONのみを出力してください
"""


def init_vertex_ai(
    credentials_json: dict,
    project_id: str,
    location: str = "global",
) -> None:
    """
    Vertex AIを初期化する

    Args:
        credentials_json: サービスアカウントJSONの辞書
        project_id: Google CloudプロジェクトID
        location: Vertex AIリージョン（デフォルト: global）
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
    複数の結果からすべての明細アイテムを統合する。
    - 活字明細表の場合: 「件名合計」行の現場名をそれより前の明細行に割り当てる
    - 単票形式・手書き伝票の場合: 各行またはページ共通の現場名を維持する

    Args:
        results: 各画像の抽出結果リスト（ページ順）

    Returns:
        現場名（order_location）が正しく適用された明細アイテムリスト
    """
    all_rows = []

    for result in results:
        page_location = result.get("order_location", "")
        page_date = result.get("date", "")
        rows = result.get("rows", [])
        if not rows and "items" in result:
            rows = result.get("items", [])

        for r in rows:
            if page_location and not r.get("order_location"):
                r["order_location"] = page_location
            if page_date and not r.get("date"):
                r["date"] = page_date
            all_rows.append(r)

    final_items = []
    current_block = []

    for r in all_rows:
        row_type = r.get("row_type", "item")

        if row_type == "item":
            current_block.append(r)

        elif row_type in ("project_total", "subject_total"):
            project_name = (
                r.get("project_name", "")
                or r.get("order_location", "")
                or r.get("remarks", "")
            )
            order_no = r.get("order_no", "")

            for item in current_block:
                item["order_location"] = project_name
                if not item.get("order_no") and order_no:
                    item["order_no"] = order_no
                final_items.append(item)

            current_block = []

    # 「件名合計」行が無かったブロック（手書き伝票などの単票形式）
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
        spec = item.get("spec", "")
        order_location = item.get("order_location", "")
        key = f"{order_location}_{product_name}_{spec}_{product_code}"

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

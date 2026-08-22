"""
Excel出力モジュール
原価管理表形式のExcelファイルを生成する
"""

import io
from typing import Any, Dict, List, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


COLUMNS = [
    {"key": "row_no", "header": "行", "width": 5},
    {"key": "category", "header": "分類", "width": 12},
    {"key": "product_code", "header": "コード", "width": 14},
    {"key": "product_name", "header": "名称", "width": 30},
    {"key": "spec", "header": "仕様", "width": 20},
    {"key": "unit", "header": "単位", "width": 6},
    {"key": "quantity", "header": "数量", "width": 8},
    {"key": "unit_price", "header": "単価", "width": 12},
    {"key": "amount", "header": "金額", "width": 14},
    {"key": "slip_no", "header": "伝票No", "width": 10},
    {"key": "order_no", "header": "注文No", "width": 12},
    {"key": "remarks", "header": "備考", "width": 25},
]

CATEGORY_MAP = {
    "吉野石膏": "ボード類",
    "DAIKEN": "ボード類",
    "大建": "ボード類",
    "TOTO": "設備",
    "LIXIL": "設備",
    "リクシル": "設備",
    "INAX": "設備",
    "JSP": "断熱材",
    "旭ファイバーグラ": "断熱材",
    "旭ファイバー": "断熱材",
    "樋口仕入先": "ケイカル板",
    "城東テクノ": "設備",
    "ニチハ": "外壁材",
    "TOTO機": "設備",
}


def classify_item(maker: str, product_name: str) -> str:
    for key, category in CATEGORY_MAP.items():
        if key in maker:
            return category

    keywords = {
        "ベベルボード": "ボード類",
        "ダイライト": "ボード類",
        "石膏ボード": "ボード類",
        "プラスターボード": "ボード類",
        "ミラフォーム": "断熱材",
        "グラスウール": "断熱材",
        "アクリアウール": "断熱材",
        "断熱": "断熱材",
        "ケイカル板": "ケイカル板",
        "合板": "合板類",
        "コンパネ": "合板類",
        "ベニヤ": "合板類",
        "サッシ": "設備",
        "窓枠": "設備",
        "サーモス": "設備",
        "値引き": "値引き",
        "NEBIKI": "値引き",
        "送料": "送料",
    }

    for keyword, category in keywords.items():
        if keyword in product_name:
            return category

    return ""


def items_to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        maker = str(item.get("maker", ""))
        product_name = str(item.get("product_name", ""))

        row = {
            "行": i,
            "分類": classify_item(maker, product_name),
            "コード": str(item.get("product_code", "")),
            "名称": product_name,
            "仕様": _build_spec(item),
            "単位": str(item.get("unit", "")),
            "数量": _safe_number(item.get("quantity", "")),
            "単価": _safe_number(item.get("unit_price", "")),
            "金額": _safe_number(item.get("amount", "")),
            "伝票No": str(item.get("slip_no", "")),
            "注文No": str(item.get("order_no", "")),
            "備考": str(item.get("remarks", "")),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["行", "分類", "コード", "名称", "仕様", "単位", "数量", "単価", "金額", "伝票No", "注文No", "備考"]
        )

    return pd.DataFrame(rows)


def _build_spec(item: Dict[str, Any]) -> str:
    product_name = str(item.get("product_name", ""))
    product_code = str(item.get("product_code", ""))
    return product_code


def _safe_number(value) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("，", "").strip()
        if cleaned == "" or cleaned == "-":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def create_excel(
    df: pd.DataFrame,
    title: str = "原価管理表",
    supplier: str = "",
    date_str: str = "",
) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "原価管理表"

    header_font = Font(name="Yu Gothic", size=10, bold=True)
    header_fill = PatternFill(start_color="B8CCE4", end_color="B8CCE4", fill_type="solid")
    data_font = Font(name="Yu Gothic", size=10)
    number_font = Font(name="Yu Gothic", size=10)
    title_font = Font(name="Yu Gothic", size=14, bold=True)
    subtitle_font = Font(name="Yu Gothic", size=10)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    ws.merge_cells("A1:L1")
    ws["A1"] = title
    ws["A1"].font = title_font
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    if supplier or date_str:
        ws.merge_cells("A2:F2")
        ws["A2"] = f"仕入先: {supplier}" if supplier else ""
        ws["A2"].font = subtitle_font

        ws.merge_cells("G2:L2")
        ws["G2"] = f"請求日: {date_str}" if date_str else ""
        ws["G2"].font = subtitle_font
        ws["G2"].alignment = Alignment(horizontal="right")
        ws.row_dimensions[2].height = 20

    header_row = 3
    headers = ["行", "分類", "コード", "名称", "仕様", "単位", "数量", "単価", "金額", "伝票No", "注文No", "備考"]
    widths = [5, 12, 14, 30, 20, 6, 8, 12, 14, 10, 12, 25]

    for col_idx, (header, width) in enumerate(zip(headers, widths), start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.row_dimensions[header_row].height = 25

    data_start_row = header_row + 1
    number_columns = {7, 8, 9}

    for row_idx, (_, row_data) in enumerate(df.iterrows(), start=data_start_row):
        for col_idx, col_name in enumerate(headers, start=1):
            value = row_data.get(col_name, "")

            if pd.isna(value):
                value = ""

            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font = data_font
            cell.border = thin_border

            if col_idx in number_columns:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.font = number_font
                if isinstance(value, (int, float)) and value != "":
                    cell.number_format = "#,##0"
            elif col_idx == 1:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

        ws.row_dimensions[row_idx].height = 20

    total_row = data_start_row + len(df)
    ws.merge_cells(f"A{total_row}:F{total_row}")
    total_label_cell = ws.cell(row=total_row, column=1, value="合計")
    total_label_cell.font = Font(name="Yu Gothic", size=10, bold=True)
    total_label_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    total_label_cell.border = thin_border
    total_label_cell.alignment = Alignment(horizontal="center", vertical="center")

    for col_idx in range(2, 7):
        cell = ws.cell(row=total_row, column=col_idx)
        cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        cell.border = thin_border

    qty_cell = ws.cell(row=total_row, column=7)
    qty_cell.font = Font(name="Yu Gothic", size=10, bold=True)
    qty_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    qty_cell.border = thin_border
    qty_cell.alignment = Alignment(horizontal="right", vertical="center")
    if len(df) > 0:
        qty_cell.value = f"=SUM(G{data_start_row}:G{total_row - 1})"
        qty_cell.number_format = "#,##0"

    price_cell = ws.cell(row=total_row, column=8)
    price_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    price_cell.border = thin_border

    amt_cell = ws.cell(row=total_row, column=9)
    amt_cell.font = Font(name="Yu Gothic", size=10, bold=True)
    amt_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    amt_cell.border = thin_border
    amt_cell.alignment = Alignment(horizontal="right", vertical="center")
    if len(df) > 0:
        amt_cell.value = f"=SUM(I{data_start_row}:I{total_row - 1})"
        amt_cell.number_format = "#,##0"

    for col_idx in range(10, 13):
        cell = ws.cell(row=total_row, column=col_idx)
        cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        cell.border = thin_border

    ws.row_dimensions[total_row].height = 25

    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1

    ws.freeze_panes = f"A{data_start_row}"

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

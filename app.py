"""
請求書OCR → 原価管理表 Streamlitアプリ

請求書（PDF/画像）をアップロードし、Vertex AI Geminiで読み取り、
原価管理表形式のExcelファイルに変換するアプリケーション
"""

import json
import io
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from pdf_processor import pdf_to_images, load_image_file
from ocr_engine import (
    init_vertex_ai,
    extract_invoice_data,
    process_multiple_images,
    merge_items,
    aggregate_by_product,
    DEFAULT_MODEL_NAME,
    AVAILABLE_MODELS,
)
from excel_exporter import items_to_dataframe, create_excel


# ページ設定
st.set_page_config(
    page_title="請求書OCR → 原価管理表",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# カスタムCSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E5090;
        border-bottom: 2px solid #2E5090;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_password() -> bool:
    """パスワード認証の画面および判定を行う"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # ログイン画面の表示
    st.markdown('<div class="main-title">📄 請求書OCR ツール</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔒 アクセス認証")
        password_input = st.text_input("パスワードを入力してください", type="password")

        if st.button("ログイン", use_container_width=True, type="primary"):
            if password_input == "sto0123":
                st.session_state.authenticated = True
                st.success("認証に成功しました")
                st.rerun()
            else:
                st.error("パスワードが正しくありません")

    return False


def init_session_state():
    """セッション状態を初期化する"""
    if "ocr_results" not in st.session_state:
        st.session_state.ocr_results = None
    if "all_items" not in st.session_state:
        st.session_state.all_items = None
    if "edited_df" not in st.session_state:
        st.session_state.edited_df = None
    if "vertex_initialized" not in st.session_state:
        st.session_state.vertex_initialized = False
    if "supplier_name" not in st.session_state:
        st.session_state.supplier_name = ""
    if "invoice_date" not in st.session_state:
        st.session_state.invoice_date = ""


def sidebar_settings():
    """サイドバーの設定UIを表示する"""
    st.sidebar.markdown("## ⚙️ 設定")

    # ログアウトボタン
    if st.sidebar.button("🚪 ログアウト"):
        st.session_state.authenticated = False
        st.rerun()

    st.sidebar.markdown("---")

    # Google Cloud認証
    st.sidebar.markdown("### 🔑 Google Cloud 認証")

    location = st.sidebar.selectbox(
        "リージョン",
        options=[
            "global",
            "us-central1",
            "asia-northeast1",
            "europe-west1",
            "asia-southeast1",
        ],
        index=0,
        help=(
            "Vertex AIのリージョンを選択してください。"
            "Gemini 3.x系モデル（gemini-3.5-flash-lite等）は global リージョンでのみ利用可能です。"
        ),
    )

    # 1. Streamlit Secretsからの自動認証試行
    if "gcp_service_account" in st.secrets and not st.session_state.vertex_initialized:
        try:
            credentials_json = dict(st.secrets["gcp_service_account"])
            project_id = credentials_json.get("project_id", "")
            if project_id:
                init_vertex_ai(credentials_json, project_id, location)
                st.session_state.vertex_initialized = True
                st.sidebar.success("⚡ Secretsから自動認証完了")
        except Exception as e:
            st.sidebar.warning(f"Secrets自動認証失敗: {str(e)}")

    # 2. 手動認証用フォーム（Secretsが無い場合や再認証用）
    with st.sidebar.expander("手動でJSONキーを設定する場合", expanded=not st.session_state.vertex_initialized):
        credentials_file = st.file_uploader(
            "サービスアカウント JSON ファイル",
            type=["json"],
            help="Google CloudのサービスアカウントJSONキーファイルをアップロードしてください",
        )

        project_id_input = st.text_input(
            "プロジェクトID",
            value="",
            help="Google CloudのプロジェクトIDを入力してください",
        )

        if st.button("🔐 手動認証する", use_container_width=True):
            if credentials_file is None:
                st.sidebar.error("JSONファイルをアップロードしてください")
            elif not project_id_input:
                st.sidebar.error("プロジェクトIDを入力してください")
            else:
                try:
                    credentials_json = json.load(credentials_file)
                    init_vertex_ai(credentials_json, project_id_input, location)
                    st.session_state.vertex_initialized = True
                    st.sidebar.success("✅ 認証成功！")
                except Exception as e:
                    st.sidebar.error(f"認証エラー: {str(e)}")
                    st.session_state.vertex_initialized = False

    # 認証状態表示
    if st.session_state.vertex_initialized:
        st.sidebar.markdown("🟢 **GCP認証済み**")
    else:
        st.sidebar.markdown("🔴 **未認証**")

    st.sidebar.markdown("---")

    # モデル設定
    st.sidebar.markdown("### 🤖 OCRモデル")
    model_name = st.sidebar.selectbox(
        "使用するGeminiモデル",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL_NAME)
        if DEFAULT_MODEL_NAME in AVAILABLE_MODELS
        else 0,
        help=(
            "Vertex AIのモデルは定期的に廃止されます。"
            "404エラーが出た場合はここで別のモデルに切り替えてください。"
        ),
    )

    st.sidebar.markdown("---")

    # 出力設定
    st.sidebar.markdown("### 📊 出力設定")
    aggregate = st.sidebar.checkbox(
        "同じ品名の数量・金額を集約する",
        value=False,
        help="チェックすると同じ品名の行を1行にまとめ、数量と金額を合計します",
    )

    return aggregate, model_name


def main():
    """メインアプリケーション"""
    init_session_state()

    # パスワードチェック（未認証の場合は処理停止）
    if not check_password():
        return

    # タイトル
    st.markdown('<div class="main-title">📄 請求書OCR → 原価管理表 変換ツール</div>', unsafe_allow_html=True)

    # サイドバー設定
    aggregate, model_name = sidebar_settings()

    # STEP 1: ファイルアップロード
    st.markdown('<div class="step-header">📁 STEP 1: 請求書ファイルをアップロード</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "請求書ファイルを選択してください（複数可）",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="PDF、PNG、JPG形式に対応しています。複数ファイルを同時にアップロードできます。",
    )

    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} 件のファイルがアップロードされました")

        with st.expander("📷 アップロードファイル プレビュー", expanded=False):
            for uploaded_file in uploaded_files:
                st.markdown(f"**{uploaded_file.name}**")
                if uploaded_file.type == "application/pdf":
                    st.markdown("📄 PDFファイル（OCR実行時にページ画像に変換されます）")
                else:
                    img = Image.open(uploaded_file)
                    st.image(img, width=400)
                    uploaded_file.seek(0)
                st.markdown("---")

    # STEP 2: OCR実行
    st.markdown('<div class="step-header">🤖 STEP 2: AI-OCR 実行</div>', unsafe_allow_html=True)

    if not st.session_state.vertex_initialized:
        st.warning("⚠️ サイドバーでGoogle Cloud認証を行ってください")

    col1, col2 = st.columns([1, 3])
    with col1:
        ocr_button = st.button(
            "🔍 OCR実行",
            disabled=not uploaded_files or not st.session_state.vertex_initialized,
            use_container_width=True,
            type="primary",
        )

    if ocr_button and uploaded_files:
        all_images = []

        with st.spinner("📄 ファイルを処理中..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()

                if uploaded_file.type == "application/pdf":
                    page_images = pdf_to_images(file_bytes)
                    for page_num, img in page_images:
                        all_images.append((f"{uploaded_file.name} - P{page_num}", img))
                else:
                    img = load_image_file(file_bytes)
                    all_images.append((uploaded_file.name, img))

        if not all_images:
            st.error("処理対象の画像がありません")
        else:
            st.info(f"🔍 {len(all_images)} ページをOCR処理中... （モデル: {model_name}）")
            progress_bar = st.progress(0, text="OCR処理中...")
            results = []

            for i, (name, img) in enumerate(all_images):
                progress_bar.progress(
                    (i) / len(all_images),
                    text=f"OCR処理中... ({i + 1}/{len(all_images)}) {name}",
                )
                try:
                    result = extract_invoice_data(img, model_name=model_name)
                    result["source_file"] = name
                    results.append(result)

                    if result.get("type") == "請求書サマリー":
                        st.session_state.supplier_name = result.get("supplier", "")
                        st.session_state.invoice_date = result.get("date", "")

                except Exception as e:
                    st.error(f"❌ {name} の処理でエラー: {str(e)}")
                    results.append({
                        "type": "エラー",
                        "error": str(e),
                        "source_file": name,
                        "items": [],
                    })

            progress_bar.progress(1.0, text="✅ OCR完了！")

            st.session_state.ocr_results = results
            all_items = merge_items(results)
            st.session_state.all_items = all_items

            if all_items:
                df = items_to_dataframe(all_items)
                st.session_state.edited_df = df
            else:
                st.session_state.edited_df = None

            st.success(f"✅ OCR完了！ {len(all_items)} 件の明細が検出されました")

    # STEP 3: データ表示・編集
    if st.session_state.ocr_results is not None:
        st.markdown('<div class="step-header">📊 STEP 3: データ確認・編集</div>', unsafe_allow_html=True)

        results = st.session_state.ocr_results
        with st.expander("🔎 OCR結果の詳細（JSON）", expanded=False):
            for result in results:
                source = result.get("source_file", "不明")
                result_type = result.get("type", "不明")
                st.markdown(f"**{source}** — タイプ: {result_type}")

                if result.get("type") == "エラー":
                    st.error(result.get("error", ""))
                    if "raw_text" in result:
                        st.code(result["raw_text"], language="text")
                else:
                    st.json(result)
                st.markdown("---")

        if aggregate and st.session_state.all_items:
            aggregated = aggregate_by_product(st.session_state.all_items)
            df = items_to_dataframe(aggregated)
            st.info(f"📌 品名集約: {len(st.session_state.all_items)} 行 → {len(aggregated)} 行")
        elif st.session_state.edited_df is not None:
            df = st.session_state.edited_df
        else:
            df = None

        if df is not None and len(df) > 0:
            st.markdown("**データを直接編集できます（ダブルクリックでセルを編集）:**")

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "行": st.column_config.NumberColumn("行", width="small"),
                    "分類": st.column_config.TextColumn("分類", width="medium"),
                    "コード": st.column_config.TextColumn("コード", width="medium"),
                    "名称": st.column_config.TextColumn("名称", width="large"),
                    "仕様": st.column_config.TextColumn("仕様", width="medium"),
                    "単位": st.column_config.TextColumn("単位", width="small"),
                    "数量": st.column_config.NumberColumn("数量", format="%d", width="small"),
                    "単価": st.column_config.NumberColumn("単価", format="%,.0f", width="medium"),
                    "金額": st.column_config.NumberColumn("金額", format="%,.0f", width="medium"),
                    "伝票No": st.column_config.TextColumn("伝票No", width="medium"),
                    "注文No": st.column_config.TextColumn("注文No", width="medium"),
                    "備考": st.column_config.TextColumn("備考", width="large"),
                },
                key="data_editor",
            )

            total_qty = edited_df["数量"].sum() if "数量" in edited_df.columns else 0

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("明細行数", f"{len(edited_df)} 行")
            with col_b:
                st.metric("数量合計", f"{total_qty:,.0f}")

            # STEP 4: Excelダウンロード
            st.markdown('<div class="step-header">📥 STEP 4: Excelダウンロード</div>', unsafe_allow_html=True)

            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                today = datetime.now().strftime("%Y%m%d")
                default_filename = f"原価管理表_{today}.xlsx"
                filename = st.text_input("ファイル名", value=default_filename)

            try:
                excel_bytes = create_excel(
                    edited_df,
                    title="原価管理表",
                    supplier=st.session_state.supplier_name,
                    date_str=st.session_state.invoice_date,
                )

                st.download_button(
                    label="📥 Excelファイルをダウンロード",
                    data=excel_bytes,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=False,
                )
            except Exception as e:
                st.error(f"Excel生成エラー: {str(e)}")

        else:
            st.warning("明細データが見つかりませんでした。御請求書（サマリーページ）のみの場合、明細表のページもアップロードしてください。")


if __name__ == "__main__":
    main()

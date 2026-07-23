from __future__ import annotations

import hashlib
import json
import uuid

import altair as alt
import streamlit as st

from browser_storage import browser_storage, hydrate_llm_config

def _relevant_business_rules(question: str, business_context: str) -> str:
    keyword_groups = {
        "退货": ("退货", "运费"),
        "发货": ("发货", "快递", "到货"),
        "支付": ("支付", "货到付款", "微信", "支付宝", "银行卡", "花呗", "信用卡"),
        "发票": ("发票", "电子发票", "纸质发票"),
        "会员": ("会员", "银卡", "金卡"),
        "优惠": ("优惠", "优惠券", "满 200", "满 500"),
        "客服": ("客服", "在线", "电话", "邮件"),
        "保修": ("保修",),
        "注销": ("注销",),
    }
    matched_terms = next((terms for group, terms in keyword_groups.items() if group in question or any(term in question for term in terms)), ())
    lines = [line.strip() for line in business_context.splitlines() if line.strip() and not line.lstrip().startswith(("#", ">"))]
    matched_lines = [line for line in lines if any(term in line for term in matched_terms)] if matched_terms else []
    return "\n".join(matched_lines) or "当前业务规则摘要未提供该问题的明确标准答案。"


from kb_quality import (
    AnalysisResult,
    LLMConfig,
    LLM_DEFAULT_BASE_URL,
    LLM_DEFAULT_MODEL,
    analyze_knowledge_base,
    fetch_available_models,
    issues_as_csv,
    aggregate_issue_rows,
    coverage_gap_markdown,
    load_articles,
    report_as_html,
    report_as_markdown,
    SEVERITY_ORDER,
    sort_issues,
)


st.set_page_config(page_title="知识库质量体检", page_icon="🩺", layout="wide")
LLM_STORAGE_KEY = "kb_quality_llm_config_v1"
storage_session_id = st.session_state.setdefault("_llm_storage_session_id", uuid.uuid4().hex)
stored_llm_config = browser_storage("load", LLM_STORAGE_KEY, {"session_id": storage_session_id})
hydrate_llm_config(stored_llm_config, st.session_state)

st.markdown(
    """
    <style>
    .block-container { max-width: 1280px; padding-top: 2rem; color: var(--text-color); }
    .hero { background: linear-gradient(135deg, #172554, #2563eb); color: #ffffff; padding: 28px 32px; border-radius: 20px; margin-bottom: 22px; }
    .hero h1 { margin: 0 0 8px; font-size: 2.2rem; color: #ffffff; }
    .hero p { margin: 0; color: #dbeafe; }
    div[data-testid="stMetric"] {
        height: 112px;
        min-height: 112px;
        box-sizing: border-box;
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        padding: 14px;
        border-radius: 14px;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--text-color) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        background: transparent;
    }
    section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
        font-size: 0;
    }
    section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child::after {
        content: "拖拽文件到这里";
        font-size: 14px;
        color: var(--text-color);
    }
    section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2) {
        font-size: 0;
    }
    section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2)::after {
        content: "单个文件最大 200 MB";
        font-size: 12px;
        color: var(--secondary-text-color);
    }
    section[data-testid="stFileUploaderDropzone"] button {
        color: transparent !important;
        font-size: 0 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button::after {
        content: "选择文件";
        color: var(--text-color);
        font-size: 14px;
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child,
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2),
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child,
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div > span:nth-child(2) {
        font-size: 0 !important;
        color: transparent !important;
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] > div > div > span:nth-of-type(2)::after {
        content: "单个文件最大 200 MB · 支持 Markdown、TXT";
        font-size: 12px;
        color: var(--secondary-text-color);
    }
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] > div > div > span:nth-of-type(2)::after {
        content: "单个文件最大 200 MB · 支持 JSON、CSV、TSV、Excel";
        font-size: 12px;
        color: var(--secondary-text-color);
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2)::after {
        content: "拖拽文件到这里\\A单个文件最大 200 MB · 支持 Markdown、TXT" !important;
        white-space: pre-line !important;
        line-height: 1.45 !important;
        font-size: 12px !important;
        color: var(--secondary-text-color) !important;
    }
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > div:nth-child(2)::after {
        content: "拖拽文件到这里\\A单个文件最大 200 MB · 支持 JSON、CSV、TSV、Excel" !important;
        white-space: pre-line !important;
        line-height: 1.45 !important;
        font-size: 12px !important;
        color: var(--secondary-text-color) !important;
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] > span,
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] > span {
        position: relative;
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] > span::after,
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] > span::after {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: none;
        content: "选择文件";
        color: var(--text-color, #ffffff);
        font-size: 14px;
    }
    .st-key-context_file section[data-testid="stFileUploaderDropzone"] > span > button,
    .st-key-articles_file section[data-testid="stFileUploaderDropzone"] > span > button {
        color: transparent !important;
        font-size: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🩺 知识库质量体检</h1><p>上传当前业务规则与 FAQ，自动识别过时、冲突、重复和缺失内容，并生成治理建议。</p></div>',
    unsafe_allow_html=True,
)

def _load_articles_for_ui(value: str) -> list[dict[str, object]]:
    if not value.strip():
        raise ValueError("请先提供知识库 FAQ：请上传文件或粘贴 JSON 数组。")
    try:
        return load_articles(value)
    except json.JSONDecodeError as exc:
        raise ValueError("FAQ 内容不是有效 JSON，请粘贴以 [ 开头的 FAQ 数组。") from exc
    except ValueError as exc:
        raise ValueError(f"FAQ 内容无法识别：{exc}") from exc


def _preview_text(value: str, limit: int = 90) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else f"{text[:limit - 1]}…"

llm_config: LLMConfig | None = None
with st.sidebar:
    st.header("检测设置")
    mode = st.selectbox(
        "检测模式",
        ["mock", "llm"],
        format_func=lambda value: "Mock（本地演示）" if value == "mock" else "LLM（真实 API）",
    )
    if mode == "mock":
        st.caption("Mock 模式不调用外部 API，适合离线演示与截图。")
    else:
        st.markdown("**LLM API 配置**")
        remember_llm_config = st.checkbox("记住 LLM 配置（保存在本机浏览器）", key="remember_llm_config")
        llm_api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="请输入当前服务的 API Key",
            key="llm_api_key",
        )
        llm_base_url = st.text_input(
            "API 地址",
            value="",
            placeholder=LLM_DEFAULT_BASE_URL,
            key="llm_base_url",
        )
        llm_config = LLMConfig(api_key=llm_api_key, base_url=llm_base_url)
        llm_models: list[str] = []
        if llm_api_key.strip() and llm_base_url.strip():
            credentials_fingerprint = hashlib.sha256(
                f"{llm_base_url.strip()}\0{llm_api_key.strip()}".encode("utf-8")
            ).hexdigest()
            if st.session_state.get("llm_models_fingerprint") != credentials_fingerprint:
                st.session_state["llm_models_fingerprint"] = credentials_fingerprint
                st.session_state["llm_models"] = []
                st.session_state.pop("llm_models_error", None)
                try:
                    with st.spinner("正在获取模型列表…"):
                        st.session_state["llm_models"] = fetch_available_models(llm_config)
                except ValueError as exc:
                    st.session_state["llm_models_error"] = str(exc)
            llm_models = st.session_state.get("llm_models", [])
            if st.button("重新获取模型列表", key="refresh_llm_models", width="stretch"):
                st.session_state["llm_models_fingerprint"] = None
                st.rerun()
            if llm_models:
                if st.session_state.get("llm_model") not in llm_models:
                    st.session_state.pop("llm_model", None)
                default_model_index = llm_models.index(LLM_DEFAULT_MODEL) if LLM_DEFAULT_MODEL in llm_models else 0
                llm_model = st.selectbox(
                    "模型名称",
                    llm_models,
                    index=default_model_index,
                    key="llm_model",
                )
            elif st.session_state.get("llm_models_error"):
                st.warning(f"模型列表获取失败：{st.session_state['llm_models_error']}")
                llm_model = LLM_DEFAULT_MODEL
            else:
                st.info("正在准备模型列表，请稍候。")
                llm_model = LLM_DEFAULT_MODEL
        else:
            st.info("填写 API Key 和 API 地址后，将自动获取模型列表。")
            llm_model = LLM_DEFAULT_MODEL
        llm_config = LLMConfig(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
        if remember_llm_config and llm_api_key.strip() and llm_base_url.strip() and llm_model.strip():
            config_fingerprint = hashlib.sha256(
                f"{llm_api_key.strip()}\0{llm_base_url.strip()}\0{llm_model.strip()}".encode("utf-8")
            ).hexdigest()
            if st.session_state.get("_saved_llm_config_fingerprint") != config_fingerprint:
                browser_storage(
                    "save",
                    LLM_STORAGE_KEY,
                    {"api_key": llm_api_key, "base_url": llm_base_url, "model": llm_model},
                )
                st.session_state["_saved_llm_config_fingerprint"] = config_fingerprint
        elif not remember_llm_config and st.session_state.get("_saved_llm_config_fingerprint"):
            browser_storage("clear", LLM_STORAGE_KEY)
            st.session_state.pop("_saved_llm_config_fingerprint", None)
        st.caption("配置仅用于当前浏览器会话；勾选记住后会保存在本机浏览器。")
        if remember_llm_config:
            st.caption("已开启本地记住：API Key 会写入此浏览器的 localStorage，仅建议在个人设备使用。")
    st.divider()
    st.markdown("**问题分类**")
    st.markdown("- 内容过时 / 业务规则冲突\n- 条目重复或相互矛盾\n- 答案缺失\n- 答案可执行性不足")

input_col, preview_col = st.columns([1.1, 0.9])
with input_col:
    st.subheader("1. 提供当前业务规则")
    with st.container(key="context_upload"):
        context_file = st.file_uploader("上传 Markdown / TXT", type=["md", "txt"], key="context_file")
    context_text = st.text_area("业务规则摘要", value="", placeholder="可直接粘贴规则摘要，例如：普通商品 7 天无理由退货；质量问题 30 天内可退换。", label_visibility="collapsed", height=220, key="context_text")
    if context_file is not None:
        context_text = context_file.getvalue().decode("utf-8-sig")

    st.subheader("2. 提供知识库 FAQ")
    with st.container(key="articles_upload"):
        articles_file = st.file_uploader("上传 JSON / CSV / TSV / Excel", type=["json", "csv", "tsv", "xlsx", "xls"], key="articles_file")
    articles_text = st.text_area("FAQ 内容", value="", placeholder="可直接粘贴 JSON 数组，例如：[{\"id\": \"KB001\", \"question\": \"如何申请退货？\", \"answer\": \"请在订单详情页申请。\"}]", label_visibility="collapsed", height=250, key="articles_text")
    if articles_file is not None:
        raw_articles = articles_file.getvalue()
        try:
            loaded_articles = load_articles(raw_articles, articles_file.name)
            articles_text = json.dumps(loaded_articles, ensure_ascii=False, indent=2)
            st.success(f"已读取 {len(loaded_articles)} 条 FAQ")
        except ValueError as exc:
            st.error(f"FAQ 文件读取失败：{exc}")

with preview_col:
    st.subheader("输入预览")
    try:
        preview_articles = _load_articles_for_ui(articles_text)
        st.info(f"当前待扫描：{len(preview_articles)} 条 FAQ")
        st.dataframe(
            [
                {
                    "ID": item["id"],
                    "分类": item["category"],
                    "问题": item["question"],
                    "答案状态": "已填写" if item["answer"] else "空答案",
                }
                for item in preview_articles
            ],
            width="stretch",
            hide_index=True,
            height=530,
        )
    except ValueError as exc:
        if articles_text.strip():
            st.warning(f"FAQ 预览不可用：{exc}")
        else:
            st.info("请上传 FAQ 文件或粘贴 JSON 数组后预览。")

run_clicked = st.button("开始质量体检", type="primary", width="stretch")
if run_clicked:
    st.session_state.pop("result", None)
    try:
        if not context_text.strip() and not articles_text.strip():
            raise ValueError("请先提供业务规则摘要和知识库 FAQ。")
        if not context_text.strip():
            raise ValueError("请先提供当前业务规则摘要。")
        if not articles_text.strip():
            raise ValueError("请先提供知识库 FAQ。")
        articles = _load_articles_for_ui(articles_text)
        with st.spinner("正在扫描 FAQ、比对业务规则并生成治理建议…"):
            st.session_state["result"] = analyze_knowledge_base(articles, context_text, mode=mode, llm_config=llm_config)
        st.success("质量体检完成")
    except ValueError as exc:
        st.error(f"无法开始检测：{exc}")

result: AnalysisResult | None = st.session_state.get("result")
if result is not None:
    st.divider()
    st.header("3. 治理报告")
    if result.warnings:
        st.warning("；".join(result.warnings))
    if result.invalid_llm_results:
        with st.expander(f"查看被忽略的 LLM 原始结果（{len(result.invalid_llm_results)} 条）"):
            st.caption("以下 JSON 保留模型原始返回的问题对象及其被忽略原因，便于排查模型输出格式。")
            st.json(result.invalid_llm_results, expanded=False)
    metric_cols = st.columns(4)
    metric_cols[0].metric("总条目", len(result.articles))
    problem_ratio = len(result.problem_article_ids) / len(result.articles)
    metric_cols[1].metric(
        f"问题条目（占全部 FAQ {problem_ratio:.0%}）",
        len(result.problem_article_ids),
    )
    metric_cols[2].metric("高风险问题", result.high_priority_count)
    metric_cols[3].metric("覆盖缺口", len(result.coverage_gaps))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("问题类型分布")
        counts = {key: value for key, value in result.issue_counts().items() if value}
        chart_counts = {
            "规则冲突 / 过时": counts.get("内容过时 / 业务规则冲突", 0),
            "重复 / 相互矛盾": counts.get("条目重复或相互矛盾", 0),
            "答案缺失": counts.get("答案缺失", 0),
            "可执行性不足": counts.get("答案可执行性不足", 0),
        }
        chart_counts = {key: value for key, value in chart_counts.items() if value}
        chart_data = [{"问题类型": key, "问题数量": value} for key, value in chart_counts.items()]
        chart = (
            alt.Chart(alt.Data(values=chart_data))
            .mark_bar(color="#4f7cff")
            .encode(
                y=alt.Y("问题类型:N", sort="-x", title=None),
                x=alt.X("问题数量:Q", title="问题条目数"),
                tooltip=[
                    alt.Tooltip("问题类型:N", title="问题类型"),
                    alt.Tooltip("问题数量:Q", title="问题数量"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("优先处理建议")
        st.markdown("🔴 **P0｜立即修复**：先处理高风险业务规则冲突、错误承诺和空答案。")
        st.markdown("🟠 **P1｜尽快处理**：合并相互矛盾的重复条目，统一会员、优惠券、物流等动态信息。")
        st.markdown("🟡 **P2｜排期治理**：补齐覆盖缺口，建立业务规则变更后的复检机制。")

    if result.coverage_gaps:
        st.subheader("建议新增主题")
        st.caption("以下主题当前 FAQ 未充分覆盖，建议业务确认后新增或补充条目。")
        for gap in result.coverage_gaps:
            st.markdown(coverage_gap_markdown(gap))

    st.subheader("问题条目列表")
    if result.issues:
        rows = aggregate_issue_rows(result.issues, result.articles)
        st.dataframe(
            [
                {
                    "ID": row["article_id"],
                    "客户原始问题": row["question"],
                    "不符合业务规则的回答": row["answer"],
                    "问题类型": row["issue_label"],
                    "严重度": row["severity"],
                }
                for row in rows
            ],
            width="stretch",
            hide_index=True,
            height=420,
        )
        st.caption("主表仅保留定位信息；不符合业务规则的回答、业务规则标准答案和问题明细统一放在下方详情中。")
        st.subheader("问题条目详情")
        issues_by_article: dict[str, list] = {}
        for issue in result.issues:
            issues_by_article.setdefault(issue.article_id, []).append(issue)
        detail_options = [f"{row['article_id']}｜{row['question']}（{row['issue_count']} 个问题）" for row in rows]
        selected_detail = st.selectbox("选择问题条目", detail_options, key="selected_issue_detail")
        selected_row = rows[detail_options.index(selected_detail)]
        selected_article_id = selected_row["article_id"]
        article_issues = sort_issues(issues_by_article[selected_article_id])
        with st.container(border=True):
            st.markdown(f"**客户原始问题**：{selected_row['question']}")
            st.markdown(f"**不符合业务规则的回答**：{selected_row['answer'] or '（空）'}")
            st.markdown("**业务规则标准答案**：")
            st.markdown(_relevant_business_rules(selected_row["question"], context_text))
            st.markdown("**问题明细**")
            detail_rows = [
                {
                    "问题类型": issue.issue_label,
                    "严重度": issue.severity,
                    "具体问题": issue.summary,
                    "检测依据": issue.evidence,
                    "治理建议": issue.recommendation,
                    "关联条目": "、".join(issue.related_ids),
                }
                for issue in article_issues
            ]
            st.dataframe(
                detail_rows,
                width="stretch",
                hide_index=True,
                height=min(360, 92 + len(detail_rows) * 86),
            )
    else:
        st.success("未发现问题条目")

    st.subheader("下载报告")
    download_cols = st.columns(3)
    download_cols[0].download_button("下载 Markdown", report_as_markdown(result), file_name="kb_quality_report.md", mime="text/markdown", width="stretch")
    download_cols[1].download_button("下载 CSV", issues_as_csv(result), file_name="kb_quality_issues.csv", mime="text/csv", width="stretch")
    download_cols[2].download_button("下载 HTML", report_as_html(result), file_name="kb_quality_report.html", mime="text/html", width="stretch")

    with st.expander("检测方式与 AI 使用情况"):
        st.markdown(f"- 当前模式：**{result.mode}**")
        st.markdown("- 确定性检查负责业务规则、空答案、重复和基础可执行性判断；LLM 适合补充复杂语义复核。")
        st.markdown("- 所有治理建议都需要业务方确认后再回写生产知识库。")





























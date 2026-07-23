"""Reusable knowledge-base quality analysis engine."""

from __future__ import annotations

import ast
import csv
import html
import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ISSUE_TYPES = {
    "OUTDATED_CONTENT": "内容过时 / 业务规则冲突",
    "INTERNAL_CONFLICT": "条目重复或相互矛盾",
    "MISSING_ANSWER": "答案缺失",
    "LOW_ACTIONABILITY": "答案可执行性不足",
}
ISSUE_TYPE_ALIASES = {
    "factual_inaccuracy": "OUTDATED_CONTENT",
    "incorrect_info": "OUTDATED_CONTENT",
    "incorrect_information": "OUTDATED_CONTENT",
    "information_error": "OUTDATED_CONTENT",
    "info_error": "OUTDATED_CONTENT",
    "inaccurate": "OUTDATED_CONTENT",
    "inaccurate_information": "OUTDATED_CONTENT",
    "wrong_info": "OUTDATED_CONTENT",
    "policy_error": "OUTDATED_CONTENT",
    "outdated_info": "OUTDATED_CONTENT",
    "outdated_content": "OUTDATED_CONTENT",
    "business_rule_conflict": "OUTDATED_CONTENT",
    "content_conflict": "INTERNAL_CONFLICT",
    "duplicate_content": "INTERNAL_CONFLICT",
    "contradiction": "INTERNAL_CONFLICT",
    "missing_information": "LOW_ACTIONABILITY",
    "missing_info": "LOW_ACTIONABILITY",
    "incomplete_information": "LOW_ACTIONABILITY",
    "missing_answer": "MISSING_ANSWER",
    "empty_answer": "MISSING_ANSWER",
    "answer_missing": "MISSING_ANSWER",
    "low_actionability": "LOW_ACTIONABILITY",
}
SEVERITY_ORDER = {"高": 0, "中": 1, "低": 2}


def _article_id_sort_key(article_id: str) -> tuple[str, int, int, str]:
    text = str(article_id).strip().upper()
    match = re.match(r"^(.*?)(\d+)$", text)
    if match:
        return match.group(1), 0, int(match.group(2)), text
    return text, 1, 0, text


def sort_issues(issues: Iterable["Issue"]) -> list["Issue"]:
    return sorted(
        issues,
        key=lambda issue: (
            SEVERITY_ORDER[issue.severity],
            _article_id_sort_key(issue.article_id),
            issue.issue_type,
        ),
    )



LLM_REVIEWER = Callable[[list[dict[str, Any]], str], dict[str, Any]]
LLM_DEFAULT_MODEL = "gpt-5.6"
LLM_DEFAULT_BASE_URL = "https://api.openai.com/v1"
LLM_TIMEOUT_SECONDS = 90
LLM_MODEL_DISCOVERY_TIMEOUT_SECONDS = 15

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = LLM_DEFAULT_MODEL
    base_url: str = LLM_DEFAULT_BASE_URL


def fetch_available_models(config: LLMConfig) -> list[str]:
    api_key = config.api_key.strip()
    if not api_key:
        raise ValueError("请先填写 LLM API Key，再获取模型列表。")
    base_url = config.base_url.strip().rstrip("/") or LLM_DEFAULT_BASE_URL
    request = Request(
        f"{base_url}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=LLM_MODEL_DISCOVERY_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = "无法读取服务端错误详情"
        raise ValueError(f"获取模型列表失败（HTTP {exc.code}）：{detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"获取模型列表失败：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("模型列表接口返回的不是有效 JSON。") from exc

    raw_models = response_data.get("data") if isinstance(response_data, dict) else response_data
    if not isinstance(raw_models, list):
        raise ValueError("模型列表接口返回格式无法识别。")
    models = sorted({str(item.get("id", "")).strip() for item in raw_models if isinstance(item, dict) and str(item.get("id", "")).strip()})
    if not models:
        raise ValueError("模型列表接口没有返回可用模型。")
    return models

def _llm_response_schema(valid_ids: Iterable[str] | None = None) -> dict[str, Any]:
    article_id_schema: dict[str, Any] = {"type": "string"}
    if valid_ids is not None:
        allowed_ids = list(valid_ids)
        if allowed_ids:
            article_id_schema["enum"] = allowed_ids
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "article_id": article_id_schema,
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "OUTDATED_CONTENT",
                                "INTERNAL_CONFLICT",
                                "MISSING_ANSWER",
                                "LOW_ACTIONABILITY",
                            ],
                        },
                        "severity": {"type": "string", "enum": ["高", "中", "低"]},
                        "customer_question": {"type": "string"},
                        "summary": {"type": "string"},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "related_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "article_id",
                        "issue_type",
                        "severity",
                        "summary",
                        "evidence",
                        "recommendation",
                        "related_ids",
                    ],
                },
            },
            "coverage_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["issues", "coverage_gaps"],
    }


def _llm_prompt(
    articles: list[dict[str, Any]],
    business_context: str,
    repair_instruction: str = "",
) -> str:
    faq_json = json.dumps(articles, ensure_ascii=False, indent=2)
    valid_ids = "、".join(str(article["id"]) for article in articles)
    return f"""请对下面的智能客服知识库做一次语义质量复核。

当前业务规则摘要（唯一权威依据）：
{business_context}

待复核 FAQ：
{faq_json}

可用 FAQ ID（issues.article_id 只能从这里选择）：{valid_ids}

请严格按以下检测机制处理每条 FAQ：
1. 根据 FAQ 的 question 提炼用户原始意图和问题类型；
2. 先判断 FAQ 库中是否存在与该用户意图对应的 FAQ。没有对应 FAQ 或没有可用答案时，只将主题放入 coverage_gaps，不要在 issues 中编造 article_id；
3. 如果存在对应 FAQ，使用该 FAQ 的 answer 与当前业务规则摘要进行比对：符合规则则不报告问题，不符合规则才写入 issues；
4. 只报告有明确证据的问题，不要因为规则摘要没有提及某个细节就自行判定错误。

只输出一个 JSON 对象，格式固定为：
{{"issues": [{{"article_id": "真实 FAQ ID", "customer_question": "客户原始问题（如果输入中没有独立客户问题，就填写 FAQ question）", "issue_type": "OUTDATED_CONTENT|INTERNAL_CONFLICT|MISSING_ANSWER|LOW_ACTIONABILITY", "severity": "高|中|低", "summary": "问题摘要", "evidence": "比对依据", "recommendation": "具体治理动作", "related_ids": ["真实 FAQ ID"]}}], "coverage_gaps": ["没有对应 FAQ 的主题"]}}

issues.article_id 必须从上面给出的可用 FAQ ID 中选择；issue_type 只能使用四个规定的英文枚举；不要使用 inaccurate、information_error、incorrect_info、outdated_info、empty_answer 等其他类型名称。coverage_gaps 只能放没有对应 FAQ 或没有可用答案的主题。没有问题时返回空数组。

{repair_instruction}"""


def _extract_message_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM API 返回结果缺少可解析的 message.content。") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        if text_parts:
            return "".join(text_parts)
    raise ValueError("LLM API 返回的 message.content 不是文本。")


def _is_response_format_compatibility_error(status_code: int, detail: str) -> bool:
    normalized = detail.lower()
    return status_code in {400, 422} and "response_format" in normalized and any(
        marker in normalized for marker in ("unavailable", "unsupported", "not support", "invalid")
    )


def _openai_llm_reviewer(
    articles: list[dict[str, Any]],
    business_context: str,
    config: LLMConfig,
    repair_instruction: str = "",
) -> dict[str, Any]:
    api_key = config.api_key.strip()
    if not api_key:
        raise ValueError("LLM 模式请填写 LLM API Key。")

    model = config.model.strip() or LLM_DEFAULT_MODEL
    base_url = config.base_url.strip().rstrip("/") or LLM_DEFAULT_BASE_URL
    schema_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "kb_quality_review",
            "strict": True,
            "schema": _llm_response_schema(str(article["id"]) for article in articles),
        },
    }
    json_response_format = {"type": "json_object"}

    def request_review(response_format: dict[str, Any]) -> dict[str, Any]:
        json_mode_instruction = (
            "请只输出一个 JSON 对象，不要输出 Markdown。JSON 对象必须包含 issues 数组和 coverage_gaps 数组。"
            if response_format["type"] == "json_object"
            else "请严格按照给定 JSON Schema 输出，不要输出 Markdown 或额外解释。"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"你是知识库治理专家。{json_mode_instruction}"},
                {"role": "user", "content": _llm_prompt(articles, business_context, repair_instruction)},
            ],
            "response_format": response_format,
            "max_tokens": 12000,
        }
        request = Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = "无法读取服务端错误详情"
            raise ValueError(f"LLM API 调用失败（HTTP {exc.code}）：{detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ValueError(f"LLM API 连接失败：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("LLM API 返回的不是有效 JSON。") from exc

        try:
            return json.loads(_extract_message_content(response_data))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM API 返回的模型内容不是有效 JSON。") from exc

    try:
        return request_review(schema_response_format)
    except ValueError as exc:
        detail = str(exc)
        status_match = re.search(r"HTTP (400|422)", detail)
        status_code = int(status_match.group(1)) if status_match else 0
        if not _is_response_format_compatibility_error(status_code, detail):
            raise
        return request_review(json_response_format)

@dataclass
class Issue:
    article_id: str
    issue_type: str
    severity: str
    summary: str
    evidence: str
    recommendation: str
    related_ids: list[str] = field(default_factory=list)
    customer_question: str = ""

    @property
    def issue_label(self) -> str:
        return ISSUE_TYPES.get(self.issue_type, self.issue_type)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issue_label"] = self.issue_label
        data["related_ids"] = ", ".join(self.related_ids)
        return data

@dataclass
class AnalysisResult:
    articles: list[dict[str, Any]]
    issues: list[Issue]
    coverage_gaps: list[Any]
    mode: str
    scanned_at: str
    effective_date: str | None
    business_rules_excerpt: str
    warnings: list[str] = field(default_factory=list)
    invalid_llm_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def problem_article_ids(self) -> list[str]:
        return sorted({issue.article_id for issue in self.issues}, key=_article_id_sort_key)

    @property
    def high_priority_count(self) -> int:
        return sum(issue.severity == "高" for issue in self.issues)

    def issue_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in ISSUE_TYPES.values()}
        for issue in self.issues:
            counts[issue.issue_label] = counts.get(issue.issue_label, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_at": self.scanned_at,
            "mode": self.mode,
            "effective_date": self.effective_date,
            "total_articles": len(self.articles),
            "problem_articles": len(self.problem_article_ids),
            "high_priority_issues": self.high_priority_count,
            "issue_counts": self.issue_counts(),
            "coverage_gaps": self.coverage_gaps,
            "warnings": self.warnings,
            "invalid_llm_results": self.invalid_llm_results,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_articles(raw: str | bytes, filename: str = "articles.json") -> list[dict[str, Any]]:
    """Load FAQ records from JSON, CSV/TSV, or XLSX bytes."""
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    suffix = Path(filename).suffix.lower()
    if suffix == ".json" or text.lstrip().startswith("["):
        data = json.loads(text)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        data = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    elif suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ValueError("读取 Excel 需要安装 pandas 和 openpyxl") from exc
        data = pd.read_excel(io.BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))).fillna("").to_dict("records")
    else:
        raise ValueError("暂不支持该文件格式，请使用 JSON、CSV、TSV 或 Excel")

    if not isinstance(data, list):
        raise ValueError("知识库文件必须是 FAQ 数组")
    normalized = []
    for index, article in enumerate(data, start=1):
        if not isinstance(article, dict):
            raise ValueError(f"第 {index} 条 FAQ 不是对象")
        item = {str(key): value for key, value in article.items()}
        item["id"] = str(item.get("id") or f"KB{index:03d}")
        item["question"] = str(item.get("question") or "").strip()
        if not item["question"]:
            raise ValueError(f"第 {index} 条 FAQ 缺少 question")
        item["answer"] = str(item.get("answer") or "").strip()
        item["category"] = str(item.get("category") or "未分类").strip()
        normalized.append(item)
    if not normalized:
        raise ValueError("知识库文件没有可分析的 FAQ")
    return normalized


def _effective_date(context: str) -> date | None:
    match = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", context)
    if not match:
        return None
    return date(int(match.group(1)), int(match.group(2)), 1)


def _first_int(patterns: Iterable[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def parse_rules(context: str) -> dict[str, Any]:
    """Extract business facts used by deterministic checks."""
    rules: dict[str, Any] = {"effective_date": _effective_date(context)}
    rules["no_reason_days"] = _first_int([r"普通商品[^\n]*?(\d+)\s*天", r"(\d+)\s*天无理由"], context)
    rules["quality_days"] = _first_int([r"质量问题[^\n]*?(\d+)\s*天"], context)
    rules["dispatch_hours"] = _first_int([r"发货时间[^\n]*?(\d+)\s*小时"], context)
    rules["non_quality_shipping"] = "买家" if "非质量问题退货运费：买家" in context else None
    rules["quality_shipping"] = "商家" if "质量问题退货运费：商家" in context else None
    rules["supported_payments"] = re.findall(r"支持：([^\n]+)", context)
    rules["unsupported_payments"] = re.findall(r"不支持：([^\n]+)", context)
    rules["invoice_electronic"] = "电子发票" in context
    rules["invoice_paper"] = "纸质发票" in context
    rules["silver_threshold"] = _first_int([r"银卡会员[^\n]*?消费满\s*(\d+)"], context)
    rules["gold_threshold"] = _first_int([r"金卡会员[^\n]*?消费满\s*(\d+)"], context)
    rules["silver_discount"] = _first_int([r"银卡会员[^\n]*?享\s*(\d+)\s*折"], context)
    rules["gold_discount"] = _first_int([r"金卡会员[^\n]*?享\s*(\d+)\s*折"], context)
    rules["coupon_text"] = next((line.strip("- ") for line in context.splitlines() if "当前活动" in line), "")
    rules["coupon_stackable"] = "优惠券不可叠加" in context
    rules["online_service_hours"] = "9:00-22:00" if "在线客服：9:00-22:00" in context else None
    return rules


def _issue(issues: list[Issue], article: dict[str, Any], issue_type: str, severity: str, summary: str, evidence: str, recommendation: str, related_ids: list[str] | None = None) -> None:
    issues.append(Issue(article["id"], issue_type, severity, summary, evidence, recommendation, related_ids or []))


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _normalized_question(text: str) -> str:
    return re.sub(r"[\s，。？！：；、'‘’“”（）()\-]", "", text).lower()


def _detect_rule_conflicts(article: dict[str, Any], context: str, rules: dict[str, Any], issues: list[Issue]) -> None:
    question = article["question"]
    answer = article["answer"]
    if not answer:
        return

    if "退货" in question or "退款" in question:
        no_reason_days = rules.get("no_reason_days")
        if "无理由" in answer and no_reason_days and re.search(r"(\d+)\s*天\s*无理由", answer):
            stated = int(re.search(r"(\d+)\s*天\s*无理由", answer).group(1))
            if stated != no_reason_days:
                _issue(issues, article, "OUTDATED_CONTENT", "高", "无理由退货期限与当前业务规则不一致", f"业务规则为 {no_reason_days} 天，答案写成 {stated} 天。", "按业务规则改为正确期限，并补充适用条件。")
        if ("所有退货" in answer or "全部退货" in answer) and "商家" in answer and rules.get("non_quality_shipping") == "买家":
            _issue(issues, article, "OUTDATED_CONTENT", "高", "退货运费承担方表述过宽", "当前规则区分质量问题与非质量问题，答案却称所有退货均由商家承担。", "拆分两种场景：非质量问题买家承担，质量问题商家承担。")

    if "发货" in question and rules.get("dispatch_hours"):
        match = re.search(r"(\d+)\s*小时", answer)
        if match and int(match.group(1)) != rules["dispatch_hours"]:
            _issue(issues, article, "OUTDATED_CONTENT", "高", "承诺发货时效与当前规则不一致", f"业务规则要求 {rules['dispatch_hours']} 小时内，答案写成 {match.group(1)} 小时。", "改为 24 小时内，并保留预售商品例外。")

    if "快递" in question and "系统自动分配" in context and "指定" not in question:
        if "顺丰" in answer or "中通" in answer:
            _issue(issues, article, "OUTDATED_CONTENT", "中", "固定快递公司表述与自动分配规则不一致", "当前规则为中通、韵达、圆通且由系统自动分配，答案出现固定承运商。", "改为说明系统自动从合作快递中分配，不承诺固定公司。")

    if "货到付款" in question and "货到付款" in answer and "不支持" in context:
        _issue(issues, article, "OUTDATED_CONTENT", "高", "支付方式与当前业务规则相反", "当前规则明确不支持货到付款，答案却表示支持。", "删除支持货到付款的表述，并说明可用的在线支付方式。")

    if "发票" in question and "纸质发票" in answer and not rules.get("invoice_paper"):
        _issue(issues, article, "OUTDATED_CONTENT", "高", "发票类型与当前业务规则不一致", "当前规则只支持电子发票，答案承诺纸质发票。", "删除纸质发票承诺，改为引导用户在订单详情页申请电子发票。")

    if "会员" in question and ("银卡" in answer or "金卡" in answer):
        for label, threshold, discount in [("银卡", rules.get("silver_threshold"), rules.get("silver_discount")), ("金卡", rules.get("gold_threshold"), rules.get("gold_discount"))]:
            if label not in answer:
                continue
            segment = answer[answer.find(label): answer.find(label) + 80]
            threshold_match = re.search(r"消费满\s*(\d+)", segment)
            discount_match = re.search(r"(\d+)\s*折", segment)
            if threshold and threshold_match and int(threshold_match.group(1)) != threshold:
                _issue(issues, article, "OUTDATED_CONTENT", "高", f"{label}会员门槛与当前规则不一致", f"当前规则门槛为 {threshold} 元，答案写成 {threshold_match.group(1)} 元。", "按当前会员规则更新门槛，并核对折扣。")
            if discount and discount_match and int(discount_match.group(1)) != discount:
                _issue(issues, article, "OUTDATED_CONTENT", "高", f"{label}会员折扣与当前规则不一致", f"当前规则为 {discount} 折，答案写成 {discount_match.group(1)} 折。", "按当前会员规则更新折扣。")

    if "优惠券" in question:
        if rules.get("coupon_stackable") and _has_any(answer, ["叠加", "可以的"]):
            _issue(issues, article, "OUTDATED_CONTENT", "高", "优惠券叠加规则与当前业务规则相反", "当前规则规定优惠券不可叠加，答案却允许每单叠加多张。", "改为说明优惠券不可叠加，并解释结算时只能选择一张。")
        if "满300减50" in answer or "满600减120" in answer:
            _issue(issues, article, "OUTDATED_CONTENT", "中", "优惠券面额与当前活动不一致", f"当前活动为“{rules.get('coupon_text') or '满200减20、满500减60'}”，答案使用了旧活动。", "删除固定旧面额，改为引用优惠券页面实时可用活动。")

    if "在线客服" in question and rules.get("online_service_hours") and "7x24" in answer:
        _issue(issues, article, "OUTDATED_CONTENT", "高", "在线客服时间与当前规则不一致", f"当前规则为 {rules['online_service_hours']}，答案承诺 7x24 小时。", "改为在线客服 9:00-22:00，并补充非服务时段的留言渠道。")


def _detect_duplicates(articles: list[dict[str, Any]], issues: list[Issue]) -> None:
    for index, current in enumerate(articles):
        current_q = _normalized_question(current["question"])
        if not current_q:
            continue
        for previous in articles[:index]:
            previous_q = _normalized_question(previous["question"])
            ratio = SequenceMatcher(None, current_q, previous_q).ratio()
            if ratio < 0.82:
                continue
            if current["answer"] == previous["answer"]:
                summary = "问题与已有条目重复"
                evidence = f"与 {previous['id']} 的问题相似度为 {ratio:.0%}，答案也相同。"
                recommendation = f"保留信息更完整的一条，删除或重定向 {current['id']}。"
            else:
                summary = "相似问题存在相互矛盾的答案"
                evidence = f"与 {previous['id']} 的问题相似度为 {ratio:.0%}，但两条答案给出的规则不同。"
                recommendation = f"合并为一个权威条目，按业务规则统一答案；建议保留 {previous['id']} 作为主条目。"
            _issue(issues, current, "INTERNAL_CONFLICT", "高", summary, evidence, recommendation, [previous["id"]])
            break


def _detect_actionability(article: dict[str, Any], issues: list[Issue]) -> None:
    answer = article["answer"]
    question = article["question"]
    if not answer:
        _issue(issues, article, "MISSING_ANSWER", "高", "答案为空，无法为用户提供帮助", "该条目的 answer 字段为空。", "补充经过业务确认的完整答案；若业务暂不支持，明确写出“不支持”及替代路径。")
        return
    if answer in {"暂无", "不清楚", "请联系客服"}:
        _issue(issues, article, "LOW_ACTIONABILITY", "中", "答案过短或缺少处理路径", "答案没有给出足够的操作步骤、条件或替代渠道。", "补充入口、前置条件、处理时效和异常场景，避免只让用户联系客服。")
    if "怎么" in question and not _has_any(answer, ["点击", "进入", "选择", "联系", "输入", "申请", "操作", "找到", "填写", "注册", "查看"]):
        _issue(issues, article, "LOW_ACTIONABILITY", "中", "操作类问题缺少明确步骤", "问题询问操作方法，但答案没有动作动词或页面路径。", "补充可复现的操作路径，并说明失败时的人工渠道。")


def _coverage_gaps(articles: list[dict[str, Any]]) -> list[str]:
    gaps = []
    question_text = " ".join(item["question"] for item in articles)
    if "保修" in question_text and any("保修" in item["question"] and not item["answer"] for item in articles):
        gaps.append("商品保修范围、期限与申请流程")
    if "注销" in question_text and any("注销" in item["question"] and not item["answer"] for item in articles):
        gaps.append("账号注销条件、入口与数据处理说明")
    if not any("投诉" in item["question"] or "申诉" in item["question"] for item in articles):
        gaps.append("投诉/申诉渠道与升级处理时效")
    if not any("隐私" in item["question"] or "个人信息" in item["question"] for item in articles):
        gaps.append("隐私、个人信息使用与删除请求")
    return gaps


def normalize_coverage_gap(raw_gap: Any) -> dict[str, str] | str:
    if isinstance(raw_gap, dict):
        parsed = raw_gap
    elif isinstance(raw_gap, str):
        text = raw_gap.strip()
        if not text:
            return ""
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            return text
    else:
        return str(raw_gap).strip()

    topic = str(parsed.get("topic") or parsed.get("主题") or "未命名覆盖缺口").strip()
    reason = str(parsed.get("reason") or parsed.get("规则依据") or parsed.get("原因") or "").strip()
    suggestion = str(parsed.get("suggestion") or parsed.get("治理建议") or parsed.get("建议") or "").strip()
    return {"topic": topic, "reason": reason, "suggestion": suggestion}


def coverage_gap_markdown(raw_gap: Any) -> str:
    gap = normalize_coverage_gap(raw_gap)
    if isinstance(gap, dict):
        lines = [f"- **主题**：{gap['topic']}"]
        if gap["reason"]:
            lines.append(f"  - **规则依据**：{gap['reason']}")
        if gap["suggestion"]:
            lines.append(f"  - **新增建议**：{gap['suggestion']}")
        return "\n".join(lines)
    return f"- {gap}"


def coverage_gap_html(raw_gap: Any) -> str:
    gap = normalize_coverage_gap(raw_gap)
    if isinstance(gap, dict):
        details = [f"<strong>{html.escape(gap['topic'])}</strong>"]
        if gap["reason"]:
            details.append(f"<br><span class='muted'>规则依据：{html.escape(gap['reason'])}</span>")
        if gap["suggestion"]:
            details.append(f"<br><span class='muted'>新增建议：{html.escape(gap['suggestion'])}</span>")
        return "".join(details)
    return html.escape(gap)


def _coverage_gap_has_answered_faq(gap: Any, articles: list[dict[str, Any]]) -> bool:
    normalized_gap = normalize_coverage_gap(gap)
    topic = normalized_gap.get("topic", "") if isinstance(normalized_gap, dict) else str(normalized_gap)
    topic_runs = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", topic)
    stop_terms = {"说明", "流程", "条件", "入口", "处理", "时效", "相关", "问题", "信息", "使用"}
    terms: set[str] = set()
    for run in topic_runs:
        if len(run) <= 4 and run not in stop_terms:
            terms.add(run)
        for size in (2, 3, 4):
            terms.update(run[index:index + size] for index in range(len(run) - size + 1))
    terms -= stop_terms
    if not terms:
        return False
    for article in articles:
        answer = str(article.get("answer", "")).strip()
        if answer and any(term in f"{article.get('question', '')}{answer}" for term in terms):
            return True
    return False


def _normalize_llm_issue(raw_issue: dict[str, Any], valid_ids: set[str]) -> Issue:
    if not isinstance(raw_issue, dict):
        raise ValueError("LLM 返回的问题项不是对象。")
    article_id = str(
        raw_issue.get("article_id")
        or raw_issue.get("faq_id")
        or raw_issue.get("id")
        or ""
    ).strip()
    if article_id not in valid_ids:
        raise ValueError(f"LLM 返回了不存在的 FAQ ID：{article_id or '空 ID'}")
    issue_type = _normalize_issue_type(raw_issue.get("issue_type") or raw_issue.get("type", ""))
    if issue_type not in ISSUE_TYPES:
        raise ValueError(f"LLM 返回了不支持的问题类型：{issue_type or '空类型'}")
    severity = str(raw_issue.get("severity", "")).strip()
    if severity not in SEVERITY_ORDER:
        raise ValueError(f"LLM 返回了不支持的严重度：{severity or '空严重度'}")
    text_fields = {}
    field_sources = {
        "summary": ("summary", "description", "problem"),
        "evidence": ("evidence", "basis", "reason", "description", "summary"),
        "recommendation": ("recommendation", "suggestion", "action"),
    }
    for field_name, source_names in field_sources.items():
        value = next((str(raw_issue.get(name)).strip() for name in source_names if raw_issue.get(name)), "")
        if not value:
            raise ValueError(f"LLM 返回的问题项缺少 {field_name}。")
        text_fields[field_name] = value
    related_ids = raw_issue.get("related_ids", raw_issue.get("related_faq_ids", []))
    if not isinstance(related_ids, list):
        raise ValueError("LLM 返回的 related_ids 不是数组。")
    normalized_related_ids = [str(item).strip() for item in related_ids if str(item).strip() in valid_ids and str(item).strip() != article_id]
    customer_question = next(
        (
            str(raw_issue.get(name)).strip()
            for name in ("customer_question", "original_question", "user_question", "customer_query")
            if raw_issue.get(name)
        ),
        "",
    )
    return Issue(
        article_id,
        issue_type,
        severity,
        text_fields["summary"],
        text_fields["evidence"],
        text_fields["recommendation"],
        normalized_related_ids,
        customer_question,
    )


def _normalize_issue_type(value: Any) -> str:
    issue_type = str(value).strip()
    label_to_type = {label: issue_type for issue_type, label in ISSUE_TYPES.items()}
    if issue_type in label_to_type:
        return label_to_type[issue_type]
    normalized = re.sub(r"[\s-]+", "_", issue_type.lower())
    return ISSUE_TYPE_ALIASES.get(normalized, issue_type)


def _has_invalid_llm_issue(review: Any, valid_ids: set[str]) -> bool:
    if not isinstance(review, dict) or not isinstance(review.get("issues"), list):
        return False
    for raw_issue in review["issues"]:
        try:
            _normalize_llm_issue(raw_issue, valid_ids)
        except ValueError:
            return True
    return False


def _merge_llm_review(
    issues: list[Issue],
    coverage_gaps: list[Any],
    review: dict[str, Any],
    articles: list[dict[str, Any]],
    warnings: list[str],
    invalid_llm_results: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(review, dict):
        raise ValueError("LLM 返回结果必须是 JSON 对象。")
    raw_issues = review.get("issues", [])
    if not isinstance(raw_issues, list):
        raise ValueError("LLM 返回的 issues 不是数组。")
    valid_ids = {str(article["id"]) for article in articles}
    for index, raw_issue in enumerate(raw_issues, start=1):
        try:
            issues.append(_normalize_llm_issue(raw_issue, valid_ids))
        except ValueError as exc:
            invalid_llm_results.append({"index": index, "reason": str(exc), "raw": raw_issue})
    if invalid_llm_results:
        error_counts = Counter(item["reason"] for item in invalid_llm_results)
        reasons = "；".join(
            f"{message}（{count} 条）" for message, count in error_counts.items()
        )
        warnings.append(
            f"LLM 返回了 {len(invalid_llm_results)} 条无法关联到具体 FAQ 的结果，已忽略；"
            f"原因：{reasons}。当前报告仅保留可关联结果。"
        )
    raw_gaps = review.get("coverage_gaps", [])
    if not isinstance(raw_gaps, list):
        raise ValueError("LLM 返回的 coverage_gaps 不是数组。")
    merged_gaps = list(coverage_gaps)
    for gap in raw_gaps:
        normalized_gap = normalize_coverage_gap(gap)
        if not normalized_gap:
            continue
        if isinstance(normalized_gap, dict) and _coverage_gap_has_answered_faq(normalized_gap, articles):
            continue
        if normalized_gap not in merged_gaps:
            merged_gaps.append(normalized_gap)
    return merged_gaps


def analyze_knowledge_base(
    articles: list[dict[str, Any]],
    business_context: str,
    mode: str = "mock",
    llm_config: LLMConfig | None = None,
    llm_reviewer: LLM_REVIEWER | None = None,
) -> AnalysisResult:
    if mode not in {"mock", "llm"}:
        raise ValueError("检测模式必须是 mock 或 llm。")
    rules = parse_rules(business_context)
    issues: list[Issue] = []
    for article in articles:
        _detect_actionability(article, issues)
        _detect_rule_conflicts(article, business_context, rules, issues)
    _detect_duplicates(articles, issues)
    coverage_gaps = _coverage_gaps(articles)
    warnings: list[str] = []
    invalid_llm_results: list[dict[str, Any]] = []
    if mode == "llm":
        if llm_reviewer is not None:
            review = llm_reviewer(articles, business_context)
        else:
            if llm_config is None:
                raise ValueError("LLM 模式请在页面填写 LLM API Key。")
            review = _openai_llm_reviewer(articles, business_context, llm_config)
            valid_ids = {str(article["id"]) for article in articles}
            if _has_invalid_llm_issue(review, valid_ids):
                try:
                    review = _openai_llm_reviewer(
                        articles,
                        business_context,
                        llm_config,
                        repair_instruction=(
                            "上一轮输出包含无法关联到 FAQ 的问题项。请重新生成完整结果："
                            "issues 只能填写上面列出的真实 FAQ ID；无法关联具体 FAQ 的主题只能放入 coverage_gaps，"
                            "不得在 issues 中使用空字符串或虚构 ID。"
                        ),
                    )
                except ValueError as exc:
                    warnings.append(f"LLM 结果自动纠正失败：{exc}")
        coverage_gaps = _merge_llm_review(issues, coverage_gaps, review, articles, warnings, invalid_llm_results)

    deduped: list[Issue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue.article_id, issue.issue_type, issue.summary)
        if key not in seen:
            deduped.append(issue)
            seen.add(key)
    deduped = sort_issues(deduped)
    mode_note = "LLM 模式：已执行确定性规则检查，并调用真实 API 完成语义复核。" if mode == "llm" else "Mock 模式：使用本地可复现规则模拟 AI 辅助判断，未调用外部 API。"
    return AnalysisResult(
        articles=articles,
        issues=deduped,
        coverage_gaps=coverage_gaps,
        mode=f"{mode}｜{mode_note}",
        scanned_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        effective_date=rules["effective_date"].isoformat() if rules["effective_date"] else None,
        business_rules_excerpt=business_context[:500],
        warnings=warnings,
        invalid_llm_results=invalid_llm_results,
    )


def _issue_question_values(issue: Issue, article_by_id: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    article = article_by_id.get(issue.article_id, {})
    standard_question = str(article.get("question", ""))
    customer_question = str(issue.customer_question or article.get("customer_question") or standard_question)
    answer = str(article.get("answer", ""))
    return customer_question, standard_question, answer

def issues_as_csv(result: AnalysisResult) -> str:
    output = io.StringIO()
    fields = ["article_id", "customer_question", "standard_question", "answer", "issue_label", "severity", "summary", "evidence", "recommendation", "related_ids"]
    article_by_id = {str(article["id"]): article for article in result.articles}
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    rows = []
    for issue in result.issues:
        customer_question, standard_question, answer = _issue_question_values(issue, article_by_id)
        rows.append({
            **{field: issue.to_dict().get(field, "") for field in fields},
            "customer_question": customer_question,
            "standard_question": standard_question,
            "answer": answer,
        })
    writer.writerows(rows)
    return output.getvalue()

def aggregate_issue_rows(issues: list[Issue], articles: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Collapse multiple findings for one FAQ into one report row."""
    article_by_id = {str(article["id"]): article for article in (articles or [])}
    grouped: dict[str, dict[str, Any]] = {}
    for issue in issues:
        row = grouped.setdefault(issue.article_id, {
            "article_id": issue.article_id,
            "question": issue.customer_question or str(article_by_id.get(issue.article_id, {}).get("customer_question", "")) or str(article_by_id.get(issue.article_id, {}).get("question", "")),
            "standard_question": str(article_by_id.get(issue.article_id, {}).get("question", "")),
            "answer": str(article_by_id.get(issue.article_id, {}).get("answer", "")),
            "issue_labels": [],
            "severity": "低",
            "issue_count": 0,
            "summaries": [],
            "recommendations": [],
            "related_ids": [],
            "_has_explicit_customer_question": bool(issue.customer_question or article_by_id.get(issue.article_id, {}).get("customer_question")),
        })
        if issue.customer_question and not row["_has_explicit_customer_question"]:
            row["question"] = issue.customer_question
            row["_has_explicit_customer_question"] = True
        row["issue_count"] += 1
        if issue.issue_label not in row["issue_labels"]:
            row["issue_labels"].append(issue.issue_label)
        if SEVERITY_ORDER[issue.severity] < SEVERITY_ORDER[row["severity"]]:
            row["severity"] = issue.severity
        if issue.summary not in row["summaries"]:
            row["summaries"].append(issue.summary)
        if issue.recommendation not in row["recommendations"]:
            row["recommendations"].append(issue.recommendation)
        for related_id in issue.related_ids:
            if related_id not in row["related_ids"]:
                row["related_ids"].append(related_id)
    rows = [
        {
            "article_id": row["article_id"],
            "question": row["question"],
            "standard_question": row["standard_question"],
            "answer": row["answer"],
            "issue_label": "、".join(row["issue_labels"]),
            "severity": row["severity"],
            "issue_count": row["issue_count"],
            "summary": "；".join(row["summaries"]),
            "recommendation": "；".join(row["recommendations"]),
            "related_ids": "、".join(row["related_ids"]),
        }
        for row in grouped.values()
    ]
    return sorted(
        rows,
        key=lambda row: (SEVERITY_ORDER[row["severity"]], _article_id_sort_key(row["article_id"])),
    )

def report_as_markdown(result: AnalysisResult) -> str:
    counts = result.issue_counts()
    lines = [
        "# 知识库质量治理报告", "", f"> 扫描时间：{result.scanned_at}  |  模式：{result.mode}", "",
        "## 摘要", f"- 总条目：{len(result.articles)}", f"- 问题条目：{len(result.problem_article_ids)}", f"- 高优先级问题：{result.high_priority_count}", "",
        "## 问题分布",
    ]
    lines.extend(f"- {label}：{count}" for label, count in counts.items() if count)
    if result.warnings:
        lines.extend(["", "## 检测提示"])
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.invalid_llm_results:
        lines.extend([
            "",
            "## 被忽略的 LLM 原始结果",
            "```json",
            json.dumps(result.invalid_llm_results, ensure_ascii=False, indent=2, default=str),
            "```",
        ])
    lines.extend(["", "## 优先处理建议", "- P0：先修复高风险的业务规则冲突和答案缺失。", "- P1：合并相互矛盾的重复条目，并统一优惠、会员、物流等动态规则。", "- P2：补齐覆盖缺口，建立每月复检和规则变更触发机制。", "", "## 问题条目"])
    article_by_id = {str(article["id"]): article for article in result.articles}
    for issue in result.issues:
        question = str(article_by_id.get(issue.article_id, {}).get("question", ""))
        answer = str(article_by_id.get(issue.article_id, {}).get("answer", ""))
        lines.extend([f"### {issue.article_id}｜{issue.issue_label}｜{issue.severity}", f"- 客户原始问题：{question}", f"- FAQ库标准回复：{answer or '（空）'}", f"- 问题：{issue.summary}", f"- 依据：{issue.evidence}", f"- 建议：{issue.recommendation}"])
    lines.extend(["", "## 覆盖缺口"])
    lines.extend(coverage_gap_markdown(gap) for gap in result.coverage_gaps)
    return "\n".join(lines) + "\n"


def report_as_html(result: AnalysisResult) -> str:
    counts = result.issue_counts()
    cards = [("总条目", len(result.articles), "blue"), ("问题条目", len(result.problem_article_ids), "red"), ("高风险问题", result.high_priority_count, "orange"), ("覆盖缺口", len(result.coverage_gaps), "purple")]
    card_html = "".join(f'<div class="metric {color}"><span>{html.escape(label)}</span><strong>{value}</strong></div>' for label, value, color in cards)
    article_by_id = {str(article["id"]): article for article in result.articles}
    rows = "".join(f"<tr><td>{html.escape(issue.article_id)}</td><td>{html.escape(str(article_by_id.get(issue.article_id, {}).get('question', '')))}</td><td>{html.escape(str(article_by_id.get(issue.article_id, {}).get('answer', '')))}</td><td>{html.escape(issue.issue_label)}</td><td><span class='severity {issue.severity}'>{issue.severity}</span></td><td>{html.escape(issue.summary)}</td><td>{html.escape(issue.recommendation)}</td></tr>" for issue in result.issues)
    count_rows = "".join(f"<li><span>{html.escape(label)}</span><b>{count}</b></li>" for label, count in counts.items() if count)
    gap_rows = "".join(f"<li>{coverage_gap_html(gap)}</li>" for gap in result.coverage_gaps)
    invalid_panel = ""
    if result.invalid_llm_results:
        invalid_json = html.escape(json.dumps(result.invalid_llm_results, ensure_ascii=False, indent=2, default=str))
        invalid_panel = f"<section class='panel'><h2>被忽略的 LLM 原始结果</h2><pre class='json'>{invalid_json}</pre></section>"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>知识库质量治理报告</title>
<style>body{{font-family:Inter,'Microsoft YaHei',sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:36px}}.shell{{max-width:1180px;margin:auto}}h1{{margin:0 0 8px;font-size:32px}}.muted{{color:#6b7280}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}}.metric{{background:#fff;border-radius:16px;padding:18px 20px;border-left:5px solid #4f7cff;box-shadow:0 4px 16px #1720330d}}.metric span{{display:block;color:#6b7280;font-size:14px}}.metric strong{{font-size:30px;display:block;margin-top:8px}}.red{{border-color:#ef476f}}.orange{{border-color:#f59e0b}}.purple{{border-color:#9b5de5}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.panel{{background:#fff;border-radius:16px;padding:22px;box-shadow:0 4px 16px #1720330d;margin-bottom:18px}}.counts{{list-style:none;padding:0;margin:0}}.counts li{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #eef0f5}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:13px 10px;border-bottom:1px solid #eef0f5;text-align:left;vertical-align:top}}th{{color:#6b7280;font-weight:600}}.severity{{font-weight:700}}.高{{color:#dc2626}}.中{{color:#d97706}}.低{{color:#2563eb}}pre.json{{background:#111827;color:#e5e7eb;padding:16px;border-radius:10px;overflow:auto;font-size:13px}}li{{margin:9px 0}}@media(max-width:800px){{.metrics,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:560px){{.metrics,.grid{{grid-template-columns:1fr}}body{{padding:18px}}}}</style></head>
<body><main class="shell"><h1>知识库质量治理报告</h1><div class="muted">扫描时间：{html.escape(result.scanned_at)} · 检测模式：{html.escape(result.mode)}</div><section class="metrics">{card_html}</section><section class="grid"><div class="panel"><h2>问题类型分布</h2><ul class="counts">{count_rows}</ul></div><div class="panel"><h2>覆盖缺口</h2><ul>{gap_rows or '<li>未发现规则内的明显覆盖缺口</li>'}</ul></div></section>{invalid_panel}<section class="panel"><h2>问题条目列表</h2><table><thead><tr><th>ID</th><th>客户原始问题</th><th>FAQ库标准回复</th><th>问题类型</th><th>等级</th><th>具体问题</th><th>治理建议</th></tr></thead><tbody>{rows or '<tr><td colspan="7">未发现问题</td></tr>'}</tbody></table></section></main></body></html>'''


def default_inputs(base_dir: str | Path = ".") -> tuple[str, str]:
    base = Path(base_dir)
    context_path = base / "task6_business_context.md"
    articles_path = base / "task6_kb_articles.json"
    context = context_path.read_text(encoding="utf-8") if context_path.exists() else ""
    articles = articles_path.read_text(encoding="utf-8") if articles_path.exists() else "[]"
    return context, articles

















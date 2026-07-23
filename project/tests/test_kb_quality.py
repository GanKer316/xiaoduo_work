import io
import json

import pytest

from kb_quality import load_articles


def test_loading_faq_json_normalizes_records():
    articles = load_articles(json.dumps([
        {"id": "KB100", "question": "如何退款？", "answer": "请在订单页申请。", "category": "售后"},
    ]))

    assert articles == [{
        "id": "KB100",
        "question": "如何退款？",
        "answer": "请在订单页申请。",
        "category": "售后",
    }]


def test_loading_faq_rejects_missing_question():
    with pytest.raises(ValueError, match="question"):
        load_articles(json.dumps([
            {"id": "KB100", "answer": "请联系客服。", "category": "售后"},
        ]))
import json
from pathlib import Path

import kb_quality
from kb_quality import (
    Issue,
    LLMConfig,
    aggregate_issue_rows,
    analyze_knowledge_base,
    fetch_available_models,
    normalize_coverage_gap,
    report_as_html,
    report_as_markdown,
    _llm_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def test_quality_scan_explains_payment_rule_conflict():
    articles = [{
        "id": "KB-PAYMENT",
        "question": "可以货到付款吗？",
        "answer": "支持货到付款。",
        "category": "支付",
    }]
    context = "## 支付\n- 不支持：货到付款"

    result = analyze_knowledge_base(articles, context)

    issue = next(issue for issue in result.issues if issue.article_id == "KB-PAYMENT")
    assert issue.issue_type == "OUTDATED_CONTENT"
    assert issue.severity == "高"
    assert "不支持货到付款" in issue.evidence
    assert "删除" in issue.recommendation


def test_quality_scan_links_conflicting_duplicate_questions():
    articles = [
        {"id": "KB-A", "question": "退货政策是什么？", "answer": "支持 7 天无理由退货。", "category": "退货"},
        {"id": "KB-B", "question": "退货政策是什么？", "answer": "支持 30 天无理由退货。", "category": "退货"},
    ]
    context = "## 退货政策\n- 普通商品：7 天无理由退货"

    result = analyze_knowledge_base(articles, context)

    issue = next(issue for issue in result.issues if issue.article_id == "KB-B" and issue.issue_type == "INTERNAL_CONFLICT")
    assert issue.related_ids == ["KB-A"]
    assert "相互矛盾" in issue.summary
    assert "合并" in issue.recommendation


def test_quality_scan_reports_missing_answer_and_coverage_gap():
    articles = [{
        "id": "KB-WARRANTY",
        "question": "商品有保修吗？",
        "answer": "",
        "category": "售后",
    }]

    result = analyze_knowledge_base(articles, "## 售后\n- 请联系客服处理售后问题")

    assert len(result.problem_article_ids) == 1
    assert any(issue.issue_type == "MISSING_ANSWER" for issue in result.issues)
    assert "商品保修范围、期限与申请流程" in result.coverage_gaps


def test_sample_knowledge_base_has_stable_governance_summary():
    articles = json.loads((ROOT / "task6_kb_articles.json").read_text(encoding="utf-8"))
    context = (ROOT / "task6_business_context.md").read_text(encoding="utf-8")

    result = analyze_knowledge_base(articles, context)

    assert len(result.articles) == 40
    assert len(result.problem_article_ids) == 10
    assert result.high_priority_count == 13
    assert result.issue_counts()["内容过时 / 业务规则冲突"] == 12
    assert result.issue_counts()["条目重复或相互矛盾"] == 1
    assert result.issue_counts()["答案缺失"] == 2
    assert len(result.coverage_gaps) == 4


def test_issue_rows_merge_multiple_findings_for_one_article():
    articles = json.loads((ROOT / "task6_kb_articles.json").read_text(encoding="utf-8"))
    context = (ROOT / "task6_business_context.md").read_text(encoding="utf-8")

    result = analyze_knowledge_base(articles, context)
    rows = aggregate_issue_rows(result.issues)

    assert len(rows) == len(result.problem_article_ids)
    kb012 = next(row for row in rows if row["article_id"] == "KB012")
    kb039 = next(row for row in rows if row["article_id"] == "KB039")
    assert kb012["issue_count"] == 4
    assert kb039["issue_count"] == 3


def test_coverage_gap_python_dict_string_is_normalized_for_ui():
    gap = normalize_coverage_gap(
        "{'topic': '邮件客服', 'reason': '规则要求 24 小时内回复，但 FAQ 未说明', 'suggestion': '新增邮件客服使用方式和回复时效 FAQ'}"
    )

    assert gap == {
        "topic": "邮件客服",
        "reason": "规则要求 24 小时内回复，但 FAQ 未说明",
        "suggestion": "新增邮件客服使用方式和回复时效 FAQ",
    }


def test_aggregate_issue_rows_sorts_by_severity_then_natural_article_id():
    issues = [
        Issue("KB010", "MISSING_ANSWER", "中", "中风险", "依据", "建议", []),
        Issue("KB002", "MISSING_ANSWER", "高", "高风险", "依据", "建议", []),
        Issue("KB001", "MISSING_ANSWER", "高", "高风险", "依据", "建议", []),
    ]

    rows = aggregate_issue_rows(issues)

    assert [row["article_id"] for row in rows] == ["KB001", "KB002", "KB010"]


def test_llm_invalid_issue_warnings_are_aggregated():
    def reviewer(articles, business_context):
        return {
            "issues": [
                {"article_id": "", "issue_type": "LOW_ACTIONABILITY", "severity": "中", "summary": "x", "evidence": "x", "recommendation": "x", "related_ids": []},
                {"article_id": "", "issue_type": "LOW_ACTIONABILITY", "severity": "中", "summary": "y", "evidence": "y", "recommendation": "y", "related_ids": []},
            ],
            "coverage_gaps": [],
        }

    result = analyze_knowledge_base(
        [{"id": "KB-INVALID", "question": "如何退款？", "answer": "请在订单页申请。", "category": "售后"}],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert len(result.warnings) == 1
    assert "2 条" in result.warnings[0]
    assert "空 ID" in result.warnings[0]
    assert "第 1 条" not in result.warnings[0]


def test_llm_mode_calls_reviewer_and_merges_semantic_findings():
    calls = []

    def reviewer(articles, business_context):
        calls.append((articles, business_context))
        return {
            "issues": [{
                "article_id": "KB-LLM",
                "issue_type": "LOW_ACTIONABILITY",
                "severity": "中",
                "summary": "答案没有说明异常处理路径",
                "evidence": "用户遇到支付失败时没有可执行的下一步",
                "recommendation": "补充重试、换支付方式和人工渠道。",
                "related_ids": [],
            }],
            "coverage_gaps": ["支付失败后的异常处理说明"],
        }

    result = analyze_knowledge_base(
        [{
            "id": "KB-LLM",
            "question": "支付失败怎么办？",
            "answer": "请重新支付。",
            "category": "支付",
        }],
        "## 支付\n- 支持：微信支付、支付宝",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert len(calls) == 1
    assert calls[0][0][0]["id"] == "KB-LLM"
    assert "## 支付" in calls[0][1]
    assert any(issue.summary == "答案没有说明异常处理路径" for issue in result.issues)
    assert "支付失败后的异常处理说明" in result.coverage_gaps
    assert result.mode.startswith("llm｜")


def test_llm_mode_accepts_common_provider_issue_type_aliases():
    def reviewer(articles, business_context):
        return {
            "issues": [
                {"article_id": "KB-ALIASES", "issue_type": "factual_inaccuracy", "severity": "高", "summary": "事实不准确", "evidence": "规则与答案不一致", "recommendation": "按规则更新", "related_ids": []},
                {"article_id": "KB-ALIASES", "issue_type": "content_conflict", "severity": "中", "summary": "内容冲突", "evidence": "同类内容矛盾", "recommendation": "合并条目", "related_ids": []},
                {"article_id": "KB-ALIASES", "issue_type": "missing_information", "severity": "中", "summary": "缺少信息", "evidence": "未说明处理步骤", "recommendation": "补充步骤", "related_ids": []},
            ],
            "coverage_gaps": [],
        }

    result = analyze_knowledge_base(
        [{"id": "KB-ALIASES", "question": "如何退款？", "answer": "请在订单页申请。", "category": "售后"}],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert {issue.issue_type for issue in result.issues} == {
        "OUTDATED_CONTENT",
        "INTERNAL_CONFLICT",
        "LOW_ACTIONABILITY",
    }
    assert result.warnings == []


def test_llm_mode_recovers_provider_description_shape_instead_of_ignoring_it():
    def reviewer(articles, business_context):
        return {
            "issues": [
                {"article_id": "KB-PROVIDER", "issue_type": "incorrect_info", "severity": "高", "description": "答案与当前退货运费规则冲突。", "recommendation": "按规则修改答案。"},
                {"article_id": "KB-PROVIDER", "issue_type": "outdated_info", "severity": "中", "description": "答案中的发货时效已过时。", "recommendation": "更新为当前发货时效。"},
                {"article_id": "KB-PROVIDER", "issue_type": "empty_answer", "severity": "高", "description": "答案为空。", "recommendation": "补充 FAQ 答案。"},
            ],
            "coverage_gaps": [],
        }

    result = analyze_knowledge_base(
        [{"id": "KB-PROVIDER", "question": "如何退款？", "answer": "请在订单页申请。", "category": "售后"}],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert len(result.issues) == 3
    assert {issue.issue_type for issue in result.issues} == {
        "OUTDATED_CONTENT",
        "MISSING_ANSWER",
    }
    assert any(issue.summary == "答案与当前退货运费规则冲突。" for issue in result.issues)
    assert result.warnings == []


def test_llm_mode_normalizes_information_error_for_existing_faq():
    def reviewer(articles, business_context):
        return {
            "issues": [{
                "article_id": "KB002",
                "issue_type": "information_error",
                "severity": "高",
                "description": "答案声称所有退货运费由商家承担，但规则明确非质量问题退货运费由买家承担。",
                "recommendation": "拆分非质量问题和质量问题两种退货运费场景。",
            }],
            "coverage_gaps": [],
        }

    result = analyze_knowledge_base(
        [{"id": "KB002", "question": "退货运费谁承担？", "answer": "所有退货运费由商家承担。", "category": "售后"}],
        "## 售后\n- 非质量问题退货运费由买家承担\n- 质量问题退货运费由商家承担",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert len(result.issues) == 1
    assert result.issues[0].issue_type == "OUTDATED_CONTENT"
    assert result.issues[0].article_id == "KB002"
    assert result.warnings == []


def test_llm_coverage_gap_is_removed_when_answered_faq_already_exists():
    def reviewer(articles, business_context):
        return {
            "issues": [],
            "coverage_gaps": [
                {"topic": "会员权益", "reason": "已有 FAQ 已覆盖会员权益", "suggestion": "无需新增"},
                {"topic": "邮件客服", "reason": "没有 FAQ 说明邮件客服时效", "suggestion": "新增邮件客服 FAQ"},
            ],
        }

    result = analyze_knowledge_base(
        [{"id": "KB-MEMBER", "question": "会员有哪些权益？", "answer": "会员可享受积分和会员折扣。", "category": "会员"}],
        "## 会员\n- 会员可享受积分和会员折扣",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert not any(isinstance(gap, dict) and gap["topic"] == "会员权益" for gap in result.coverage_gaps)
    assert any(isinstance(gap, dict) and gap["topic"] == "邮件客服" for gap in result.coverage_gaps)


def test_issue_rows_include_original_customer_question():
    issue = Issue("KB002", "OUTDATED_CONTENT", "高", "规则冲突", "依据", "建议", [])

    rows = aggregate_issue_rows([issue], [{"id": "KB002", "question": "退货运费谁承担？", "answer": "所有退货运费由商家承担。"}])

    assert rows[0]["question"] == "退货运费谁承担？"
    assert rows[0]["standard_question"] == "退货运费谁承担？"
    assert rows[0]["answer"] == "所有退货运费由商家承担。"


def test_issue_rows_keep_llm_customer_question_separate_from_standard_question():
    issue = Issue(
        "KB002",
        "OUTDATED_CONTENT",
        "高",
        "规则冲突",
        "依据",
        "建议",
        [],
        "退货运费谁出？",
    )

    rows = aggregate_issue_rows(
        [issue],
        [{"id": "KB002", "question": "退货运费承担规则", "answer": "按质量问题区分承担方。"}],
    )

    assert rows[0]["question"] == "退货运费谁出？"
    assert rows[0]["standard_question"] == "退货运费承担规则"

def test_llm_prompt_requires_fixed_standardized_detection_protocol():
    prompt = _llm_prompt(
        [{"id": "KB001", "question": "如何退款？", "answer": "请在订单页申请。"}],
        "## 售后\n- 普通商品 7 天无理由退货",
    )

    assert "根据 FAQ 的 question 提炼用户原始意图" in prompt
    assert "使用该 FAQ 的 answer 与当前业务规则摘要进行比对" in prompt
    assert "不要使用 inaccurate、information_error" in prompt
    assert '"issues":' in prompt and '"coverage_gaps":' in prompt


def test_llm_mode_requires_browser_config_without_injected_reviewer():

    with pytest.raises(ValueError, match="页面填写 LLM API Key"):
        analyze_knowledge_base(
            [{
                "id": "KB-LLM",
                "question": "如何退款？",
                "answer": "请在订单页申请。",
                "category": "售后",
            }],
            "## 售后\n- 普通商品 7 天无理由退货",
            mode="llm",
        )

def test_llm_mode_accepts_browser_config():
    captured = []

    def reviewer(articles, business_context):
        captured.append((articles, business_context))
        return {"issues": [], "coverage_gaps": []}

    result = analyze_knowledge_base(
        [{
            "id": "KB-BROWSER",
            "question": "如何修改地址？",
            "answer": "请进入订单详情页修改。",
            "category": "订单",
        }],
        "## 订单\n- 支持订单详情页修改地址",
        mode="llm",
        llm_config=LLMConfig(api_key="browser-secret", model="test-model", base_url="https://example.test/v1"),
        llm_reviewer=reviewer,
    )

    assert len(captured) == 1
    assert result.mode.startswith("llm｜")

def test_fetch_available_models_uses_browser_credentials(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b'{"data": [{"id": "model-z"}, {"id": "model-a"}, {"id": "model-z"}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(kb_quality, "urlopen", fake_urlopen)

    models = fetch_available_models(LLMConfig(
        api_key="browser-secret",
        base_url="https://example.test/v1",
    ))

    assert models == ["model-a", "model-z"]
    assert captured == {
        "url": "https://example.test/v1/models",
        "method": "GET",
        "auth": "Bearer browser-secret",
        "timeout": 15,
    }

def test_llm_mode_falls_back_when_provider_rejects_json_schema(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "{\\"issues\\": [], \\"coverage_gaps\\": []}"}}]}'

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        if len(requests) == 1:
            raise kb_quality.HTTPError(
                request.full_url,
                400,
                "unsupported response format",
                {},
                io.BytesIO(b'{"error":{"message":"This response_format type is unavailable now"}}'),
            )
        return FakeResponse()

    monkeypatch.setattr(kb_quality, "urlopen", fake_urlopen)

    result = analyze_knowledge_base(
        [{
            "id": "KB-FALLBACK",
            "question": "如何退款？",
            "answer": "请在订单页申请。",
            "category": "售后",
        }],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_config=LLMConfig(api_key="browser-secret", base_url="https://example.test/v1"),
    )

    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"]["type"] == "json_object"
    assert result.mode.startswith("llm｜")


def test_llm_mode_retries_when_provider_returns_unscoped_issue(monkeypatch):
    responses = [
        '{"choices":[{"message":{"content":"{\\"issues\\":[{\\"article_id\\":\\"\\",\\"issue_type\\":\\"LOW_ACTIONABILITY\\",\\"severity\\":\\"中\\",\\"summary\\":\\"缺少异常处理\\",\\"evidence\\":\\"未说明失败后的路径\\",\\"recommendation\\":\\"补充处理步骤\\",\\"related_ids\\":[]}],\\"coverage_gaps\\":[]}"}}]}'.encode("utf-8"),
        '{"choices":[{"message":{"content":"{\\"issues\\":[{\\"article_id\\":\\"KB-RETRY\\",\\"issue_type\\":\\"LOW_ACTIONABILITY\\",\\"severity\\":\\"中\\",\\"summary\\":\\"缺少异常处理\\",\\"evidence\\":\\"未说明失败后的路径\\",\\"recommendation\\":\\"补充处理步骤\\",\\"related_ids\\":[]}],\\"coverage_gaps\\":[]}"}}]}'.encode("utf-8"),
    ]
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return responses.pop(0)

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(kb_quality, "urlopen", fake_urlopen)

    result = analyze_knowledge_base(
        [{"id": "KB-RETRY", "question": "如何退款？", "answer": "请在订单页申请。", "category": "售后"}],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_config=LLMConfig(api_key="browser-secret", model="test-model", base_url="https://example.test/v1"),
    )

    assert len(requests) == 2
    assert requests[0]["response_format"]["json_schema"]["schema"]["properties"]["issues"]["items"]["properties"]["article_id"]["enum"] == ["KB-RETRY"]
    assert [issue.article_id for issue in result.issues] == ["KB-RETRY"]
    assert result.warnings == []

def test_llm_mode_ignores_issue_without_article_id_and_reports_warning():
    def reviewer(articles, business_context):
        return {
            "issues": [{
                "article_id": "",
                "issue_type": "LOW_ACTIONABILITY",
                "severity": "中",
                "summary": "模型返回了不完整问题",
                "evidence": "缺少关联 FAQ",
                "recommendation": "忽略该条结果",
                "related_ids": [],
            }],
            "coverage_gaps": [],
        }

    result = analyze_knowledge_base(
        [{
            "id": "KB-INVALID",
            "question": "如何退款？",
            "answer": "请在订单页申请。",
            "category": "售后",
        }],
        "## 售后\n- 普通商品 7 天无理由退货",
        mode="llm",
        llm_reviewer=reviewer,
    )

    assert result.issues == []
    assert any("空 ID" in warning for warning in result.warnings)
    assert result.invalid_llm_results == [{
        "index": 1,
        "reason": "LLM 返回了不存在的 FAQ ID：空 ID",
        "raw": {
            "article_id": "",
            "issue_type": "LOW_ACTIONABILITY",
            "severity": "中",
            "summary": "模型返回了不完整问题",
            "evidence": "缺少关联 FAQ",
            "recommendation": "忽略该条结果",
            "related_ids": [],
        },
    }]
    assert '"raw"' in report_as_markdown(result)
    assert "被忽略的 LLM 原始结果" in report_as_html(result)

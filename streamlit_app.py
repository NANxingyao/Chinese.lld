import re
import json
import requests
import traceback
from typing import Dict, Any, Tuple, List
import streamlit as st
import plotly.graph_objects as go

# 假设MODEL_CONFIGS和MODEL_OPTIONS已定义（保持原有配置）
MODEL_CONFIGS = {
    # 这里保持原有模型配置结构
    "OpenAI": {"base_url": "https://api.openai.com", "endpoint": "/v1/chat/completions", 
               "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
               "payload": lambda model, messages, max_tokens, temperature: {
                   "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature
               }},
    # 其他模型配置...
}

MODEL_OPTIONS = {
    # 这里保持原有模型选项
    "OpenAI (GPT-3.5)": {"provider": "OpenAI", "model": "gpt-3.5-turbo", "api_key_env": "OPENAI_API_KEY"},
    # 其他模型选项...
}

# 词类规则集（保持不变）
RULE_SETS = {
    # 2.4 语气词
    "语气词": [
        {"name": "MOD1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD2_只能附着在其他成分之后（四种用法之一得60）", "desc": "只能附着在其他成分之后（句末/话题性成分后/并列项后/假设分句后）", "match_score": 60, "mismatch_score": -60},
        {"name": "MOD3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 其他词类规则...
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 工具函数（保持不变）
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        return ""
    try:
        choices = resp_json.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            msg = first.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            for k in ("content", "text", "message"):
                if k in first and isinstance(first[k], str):
                    return first[k]
    except:
        pass
    return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[dict, str]:
    if not text:
        return None, ""
    s = text.strip()
    try:
        return json.loads(s), s
    except:
        m = re.search(r"(\{[\s\S]*\})", s)
        if not m:
            return None, s
        cand = m.group(1)
        c = cand.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
        c = re.sub(r"'(\s*[^']+?\s*)'\s*:", r'"\1":', c)
        c = re.sub(r":\s*'([^']*?)'", r': "\1"', c)
        c = re.sub(r",\s*([}\]])", r"\1", c)
        c = re.sub(r"\bTrue\b", "true", c)
        c = re.sub(r"\bFalse\b", "false", c)
        c = re.sub(r"\bNone\b", "null", c)
        try:
            return json.loads(c), c
        except:
            return None, s

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str):
        return None
    kk = re.sub(r'\s+', '', k).upper()
    for r in pos_rules:
        if r["name"].upper() == kk or re.sub(r'\s+', '', r["name"]).upper() == kk:
            return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match = rule["match_score"]
    mismatch = rule["mismatch_score"]
    if isinstance(raw_val, (int, float)):
        cand = [match, mismatch]
        return min(cand, key=lambda x: abs(x - float(raw_val)))
    if isinstance(raw_val, bool):
        return match if raw_val else mismatch
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"):
            return match
        if s in ("no", "n", "false", "否", "×", "不符合"):
            return mismatch
    return mismatch

# ===============================
# 安全的 LLM 调用函数
# ===============================
def call_llm_api(messages: list, provider: str, model: str, api_key: str,
                 max_tokens: int = 1024, temperature: float = 0.0, timeout: int = 30) -> Tuple[bool, dict, str]:
    """
    调用指定 LLM API 获取响应。
    返回: (成功标志, 响应 dict, 错误信息)
    """
    if not api_key:
        return False, {"error": "API Key 为空"}, "API Key 未提供"

    if provider not in MODEL_CONFIGS:
        return False, {"error": f"未知提供商 {provider}"}, f"未知提供商 {provider}"

    cfg = MODEL_CONFIGS[provider]
    url = cfg["base_url"].rstrip("/") + cfg.get("endpoint", "/chat/completions")
    headers = cfg["headers"](api_key)
    payload = cfg["payload"](model, messages, max_tokens=max_tokens, temperature=temperature)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if r.status_code != 200:
            return False, {"error": f"HTTP错误 {r.status_code}", "content": r.text}, f"HTTP错误 {r.status_code}: {r.text[:200]}"
            
        r.raise_for_status()
        resp_json = r.json()
        return True, resp_json, ""
    except Exception as e:
        error_msg = str(e)
        return False, {"error": error_msg}, error_msg

# 新增：测试模型连接函数
def test_model_connection(provider: str, model: str, api_key: str) -> Tuple[bool, str]:
    """测试模型连接是否成功"""
    if not api_key:
        return False, "API Key 未提供"
    
    # 使用简单消息测试连接
    test_messages = [
        {"role": "system", "content": "请返回'连接测试成功'"},
        {"role": "user", "content": "测试连接"}
    ]
    
    ok, _, err_msg = call_llm_api(
        messages=test_messages,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=10,
        temperature=0.0
    )
    
    return ok, err_msg if not ok else "连接成功"

# ===============================
# 安全的词类判定函数（保持不变）
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str]:
    if not word:
        return {}, "", "未知"

    rules_summary_lines = []
    for pos, rules in RULE_SETS.items():
        rules_summary_lines.append(f"{pos}:")
        for r in rules:
            rules_summary_lines.append(f"  - {r['name']}: {r['desc']} (match={r['match_score']}, mismatch={r['mismatch_score']})")
    rules_text = "\n".join(rules_summary_lines)

    system_msg = (
        "你是语言学研究专家，拥有中外语言学界的所有知识。在输入一个中文词语后，请检索全网的相关知识，严格按照定义的规则，请判断最可能的词类并返回 JSON："
        '{"predicted_pos":"<词类名>", "scores": {"<词类名>": {"<规则名>": <值>, ...}, ...}, "explanation":"说明"}。'
    )
    user_prompt = f"词语：『{word}』\n请基于下列规则判定并评分：\n\n{rules_text}\n\n仅返回严格 JSON。"

    ok, resp_json, err_msg = call_llm_api(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_prompt}],
        provider=provider,
        model=model,
        api_key=api_key
    )

    if not ok or not resp_json:
        return {}, f"调用失败或返回异常: {err_msg}", "未知"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)
    if not parsed_json:
        return {}, raw_text, "未知"

    scores_out = {}
    predicted_pos = parsed_json.get("predicted_pos", "未知")
    raw_scores = parsed_json.get("scores", {})

    for pos, rules in RULE_SETS.items():
        scores_out[pos] = {r["name"]: 0 for r in rules}
        raw_for_pos = raw_scores.get(pos, {})
        if isinstance(raw_for_pos, dict):
            for k, v in raw_for_pos.items():
                nk = normalize_key(k, rules)
                if nk:
                    rule_def = next(r for r in rules if r["name"] == nk)
                    scores_out[pos][nk] = map_to_allowed_score(rule_def, v)

    return scores_out, raw_text, predicted_pos

# ===============================
# 雷达图（保持不变）
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm:
        st.warning("无法绘制雷达图：没有有效数据。")
        return
    categories = list(scores_norm.keys())
    if not categories:
        st.warning("无法绘制雷达图：没有有效词类。")
        return
    values = [float(scores_norm[c]) for c in categories]
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title=dict(text=title, x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Streamlit UI
# ===============================

# ======== 侧边栏部分 ========
st.sidebar.markdown("## 模型设置")
model_choice = st.sidebar.selectbox("选择模型", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_choice]

st.sidebar.markdown(f"**当前模型：** {model_choice}")
st.sidebar.markdown(f"**模型名称：** `{selected_model['model']}`")

# 输入API Key（密码框形式，不显示明文）
api_key_input = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder=f"请输入{model_choice}的API Key",
    help=f"需要{selected_model['api_key_env']}环境变量对应的密钥"
)

# 测试连接按钮
if st.sidebar.button("测试模型连接"):
    if not api_key_input:
        st.sidebar.error("请先输入API Key")
    else:
        with st.sidebar.spinner("测试连接中..."):
            ok, msg = test_model_connection(
                selected_model["provider"],
                selected_model["model"],
                api_key_input
            )
            if ok:
                st.sidebar.success(f"✅ {msg}")
            else:
                st.sidebar.error(f"❌ 连接失败：{msg}")

# ======== 主体部分 ========
st.markdown("<h1 style='text-align: center;'>📊汉语词类隶属度检测判类</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>输入单个词 → 模型自动判类并返回各词类规则得分与隶属度（标准化 0~1）</p>", unsafe_allow_html=True)
st.write("")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    word_input = st.text_input("", placeholder="在此输入要分析的词（例如：很 / 跑 / 美丽）")
    confirm = st.button("确认")

if confirm:
    word = (word_input or "").strip()
    if not word:
        st.warning("请输入一个词语后确认。")
    else:
        if not api_key_input:
            st.error("请先在侧边栏输入API Key")
            scores_all, raw_out, predicted_pos = {}, "", "无"
        else:
            with st.spinner("模型打分判类中……"):
                try:
                    scores_all, raw_out, predicted_pos = ask_model_for_pos_and_scores(
                        word, selected_model["provider"], selected_model["model"], api_key_input
                    )
                except Exception as e:
                    st.error(f"模型调用出错：{e}")
                    traceback.print_exc()
                    scores_all, raw_out, predicted_pos = {}, str(e), "错误"

        if scores_all:
            st.subheader(f"词类预测结果：{predicted_pos}")
            st.json(scores_all)
            st.text_area("原始输出", raw_out, height=200)
        else:
            st.info("未获得有效评分结果。请检查 API Key 或网络连接。")
            st.text_area("错误信息", raw_out, height=200)
        
        # 计算每个词类总分与归一化隶属度
        pos_totals = {}
        pos_normed = {}
        for pos, score_map in scores_all.items():
            total = sum(score_map.values())
            pos_totals[pos] = total
            max_possible = MAX_SCORES.get(pos, sum(abs(x) for x in score_map.values()) or 1)
            norm = round(max(0, total) / max_possible, 3) if max_possible != 0 else 0.0
            pos_normed[pos] = norm

        # 输出顶部摘要
        st.markdown("---")
        st.subheader("判定摘要")
        st.markdown(f"- **输入词**： `{word}`")
        st.markdown(f"- **模型预测词类**： **{predicted_pos}**")

        # 排名与表格
        ranked = []
        if pos_normed:
            ranked = sorted(pos_normed.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader("隶属度排行（前10）")
        if ranked:
            for i, (p, s) in enumerate(ranked[:10]):
                st.write(f"{i+1}. **{p}** — 隶属度：{s}")

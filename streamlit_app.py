        {"name": "PART4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "PART5_不能做修饰性成分", "desc": "不能做状语、补语和定语等修饰性成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.4 语气词
    "语气词": [
        {"name": "MOD1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD2_只能附着在其他成分之后（四种用法之一得60）", "desc": "只能附着在其他成分之后（句末/话题性成分后/并列项后/假设分句后）", "match_score": 60, "mismatch_score": -60},
        {"name": "MOD3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.5 感叹词
    "感叹词": [
        {"name": "INT1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 30, "mismatch_score": -30},
        {"name": "INT2_可以独立成句（前后长停顿）", "desc": "可以独立成句（前后都可有较长停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "INT3_不能跟其他句法成分组合构句法结构", "desc": "不能与其他句法成分组合构成主谓/述补/并列等结构", "match_score": 50, "mismatch_score": -50},
    ],
    # 2.6 拟声词
    "拟声词": [
        {"name": "ON1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "ON2_可以独立成句", "desc": "可以独立成句", "match_score": 20, "mismatch_score": -20},
        {"name": "ON3_可直接或带'的'作定语", "desc": "可以直接或带'的'后作定语修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON4_可直接或带'地'作状语", "desc": "可以直接或后带'地'作状语修饰动词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON5_不能充当主/宾/谓/补等", "desc": "不能充当主语、宾语、谓语和补语等句法成分", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.1 体代词（代词与数量词部分示例）
    "体代词": [
        {"name": "PR1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "PR2_可做定语或跟'的'构'的'字结构", "desc": "可以做定语或跟助词'的'构成'的'字结构", "match_score": 10, "mismatch_score": -10},
        {"name": "PR3_不能受数量/形容词/'的'修饰", "desc": "不能受数量词、形容词和'的'字结构的修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "PR4_不能受'不/很'等副词修饰", "desc": "不能受'不'和'很'等副词修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PR5_不能作谓语核心", "desc": "不能作谓语和谓语核心（不能带宾语/时体助词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PR6_不能做补语或状语", "desc": "不能做补语，也不能作状语", "match_score": 10, "mismatch_score": -10},
        {"name": "PR7_不能后附单音方位词构处所", "desc": "不能后附单音方位词构处所", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.2 谓代词（示例）
    "谓代词": [
        {"name": "WP1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "WP2_可作状语直接修饰动/形", "desc": "可以作状语直接修饰动词或形容词", "match_score": 20, "mismatch_score": -20},
        {"name": "WP3_不能受'很'等程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP4_可受'不/也'等副词修饰", "desc": "可以受'不'或'也'等副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP5_可做谓语或谓词核心", "desc": "可以做谓语或谓词核心", "match_score": 10, "mismatch_score": -10},
        {"name": "WP6_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 10, "mismatch_score": -10},
    ],
    # 3.3 代词（通用代词条目示例）
    "代词": [
        {"name": "DPR1_可作典型主宾语", "desc": "可做典型主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR2_不能受数量/形容/的修饰", "desc": "不能受数量词、形容词和'的'字结构修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR3_不能受程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR4_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 20, "mismatch_score": 0},
        {"name": "DPR5_可受'不/也'等副词修饰（针对谓代）或不能后附方位（针对体代）", "desc": "混合规则，按具体代词类型判定", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.4 系数词、位数词、合成数词等：
    "系数词": [
        {"name": "NUM_CO1_黏着词不能单独回答", "desc": "系数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO2_可在量词前构数量词组", "desc": "可以用在量词前，一起构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO3_可构系谓构造", "desc": "可以用在位数词/构成序数组合等", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO4_可构序数组合（第...）", "desc": "可以用在'第'的后面构成序数组合", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO5_不能直接修饰名词（除非省略'第'）", "desc": "不能直接修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
    ],
    "位数词": [
        {"name": "NUM_POS1_黏着词不能单独回答", "desc": "位数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS2_不能单独用在量词前", "desc": "不能单独用在量词前", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_POS3_可在系数词后构成系位构造", "desc": "可以用在系数词后构成系位构造", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS4_不能用于前缀'第'后面构序数组合", "desc": "不能用于前缀'第'后面构序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS5_不能作定语直接修饰名词", "desc": "不能作定语直接修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_POS6_可用在'来/把'之前构数次组合", "desc": "可以用在助词'来'或'把'之前构成数次组合", "match_score": 10, "mismatch_score": 0},
    ],
    "合成数词": [
        {"name": "NUM_COM1_可以单独回答问题（部分自由）", "desc": "合成数词可以用来单独回答问题", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_COM2_可与量词构数量词组", "desc": "可以用在量词前构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM3_可在'第'后构序数组合", "desc": "可以用在'第'后造成序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM4_不能直接作定语修饰名词（除非省第）", "desc": "不能直接作定语修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_COM5_可出现在'来/多/余'之前等特殊分布", "desc": "可以出现在特定助词之前（见原文条目）", "match_score": 30, "mismatch_score": 0},
    ],
    # 其他规则占位（便于以后补全）
    # "未列出词类": [ ... ],
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# 模型配置 - 假设API在后台部署，无需前端输入API Key
MODEL_OPTIONS = {
    "OpenAI": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "DeepSeek": {"provider": "deepseek", "model": "deepseek-chat"},
    "Moonshot (Kimi)": {"provider": "moonshot", "model": "moonshot-v1-8k"},
    "豆包": {"provider": "doubao", "model": "doubao-pro"},
    "通义千问 (Qwen)": {"provider": "qwen", "model": "qwen-plus"},
}

# 模型详细配置 - 适配后台部署的API
MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda api_key: {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        "payload": lambda model, messages, max_tokens, temperature: {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda api_key: {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        "payload": lambda model, messages, max_tokens, temperature: {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda api_key: {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        "payload": lambda model, messages, max_tokens, temperature: {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    },
    "doubao": {
        "base_url": "https://api.doubao.com/chat",
        "endpoint": "/completions",
        "headers": lambda api_key: {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        "payload": lambda model, messages, max_tokens, temperature: {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    },
    "qwen": {
        "base_url": "https://api.qwen.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda api_key: {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        "payload": lambda model, messages, max_tokens, temperature: {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    }
}

# ===============================
# 工具函数
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
def call_llm_api(messages: list, provider: str, model: str,
                 max_tokens: int = 1024, temperature: float = 0.0, timeout: int = 30) -> Tuple[bool, dict, str]:
    """
    调用指定 LLM API 获取响应（假设API在后台部署，无需前端API Key）
    返回: (成功标志, 响应 dict, 错误信息)
    """
    if provider not in MODEL_CONFIGS:
        return False, {"error": f"未知提供商 {provider}"}, f"未知提供商 {provider}"

    cfg = MODEL_CONFIGS[provider]
    url = cfg["base_url"].rstrip("/") + cfg.get("endpoint", "/chat/completions")
    # 从环境变量获取API Key（后台部署）
    api_key = os.getenv(f"{provider.upper()}_API_KEY", "")
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
        st.error(f"API调用错误: {error_msg}")
        return False, {"error": error_msg}, error_msg

# ===============================
# 安全的词类判定函数
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str) -> Tuple[Dict[str, Dict[str, int]], str, str]:
    """
    根据输入词调用 LLM 获取词类隶属度评分，返回:
        - scores_all: 每个词类的规则得分字典
        - raw_text: 模型原始输出
        - predicted_pos: 模型预测的最可能词类
    """
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
        model=model
    )

    if not ok or not resp_json:
        # 调用失败或返回为空
        return {}, f"调用失败或返回异常: {err_msg}", "未知"

    # 尝试解析原始文本
    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)
    if not parsed_json:
        return {}, raw_text, "未知"

    # 解析得分
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
# 雷达图
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
# Streamlit UI（简洁居中输入 + 模型选择 + 结果）
# ===============================

# ======== 模型选择部分（侧边栏） ========
# 由侧边栏选择模型
model_choice = st.sidebar.selectbox("选择模型", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_choice]

st.sidebar.markdown(f"**当前模型：** {model_choice}")
st.sidebar.markdown(f"**模型名称：** `{selected_model['model']}`")

# 获取选中模型的配置
PROVIDER = selected_model["provider"]
MODEL_NAME = selected_model["model"]

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
        with st.spinner("模型打分判类中……"):
            try:
                scores_all, raw_out, predicted_pos = ask_model_for_pos_and_scores(
                    word, PROVIDER, MODEL_NAME
                )
            except Exception as e:
                st.error(f"模型调用出错：{e}")
                import traceback
                traceback.print_exc()
                scores_out, raw_out, predicted_pos = {}, str(e), "错误"

        # 仅在 scores_all 有内容时才遍历
        if scores_all:
            st.subheader(f"词类预测结果：{predicted_pos}")
            st.json(scores_all)
            st.text_area("原始输出", raw_out, height=200)
        else:
            st.info("未获得有效评分结果。请检查网络连接。")
            st.text_area("错误信息", raw_out, height=200)
    
        # 计算每个词类总分与归一化隶属度（0~1）
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

        # 排名与表格（只显示前 10）
        ranked = []
        if pos_normed:
            ranked = sorted(pos_normed.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader("隶属度排行（前10）")
        if ranked:
            for i, (p, s) in enumerate(ranked[:10]):
                st.write(f"{i+1}. **{p}** — 隶属度：{s}")

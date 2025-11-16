#11.16测试用
import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Any

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测",  # 页面标题
    page_icon="📰",                  # 页面图标
    layout="centered",               # 布局居中
    initial_sidebar_state="expanded",  # 修改为默认展开侧边栏
    menu_items=None                  # 隐藏默认菜单
)

# 自定义CSS样式，隐藏Streamlit默认的顶部和底部元素
hide_streamlit_style = """
<style>
/* 隐藏顶部菜单栏（Share / GitHub 等） */
header {visibility: hidden;}
/* 隐藏右下角“Manage app” */
footer {visibility: hidden;}
/* 调整侧边栏宽度 */
/*
[data-testid="stSidebar"][aria-expanded="true"]{
    min-width: 300px;
    max-width: 400px;
}
*/
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 用于兼容 call_llm_api 旧函数
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_handler": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },

    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_handler": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },

    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_handler": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },

   "doubao": {
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "endpoint": "/chat/completions",
    "headers": lambda key: {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    "payload": lambda model, messages, **kw: {
        "model": model,
        "messages": messages,
        "max_tokens": kw.get("max_tokens", 1024),
        "temperature": kw.get("temperature", 0.0),
        "stream": False,
    },
    "response_handler": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
},

    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "input": {"messages": messages},
            "parameters": {
                "max_tokens": kw.get("max_tokens", 1024),
                "temperature": kw.get("temperature", 0.0),
            },
        },
        "response_handler": lambda resp: resp.get("output", {}).get("text", "")
    },
}

# ===============================
# 模型配置与 API Key（从环境变量获取）
# ===============================
MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-1f346646d29947d0a5e29dbaa37476b8"),
    },

    "OpenAI GPT-4o": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key": os.getenv("OPENAI_API_KEY", "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"),
    },

    "Moonshot（Kimi）": {
        "provider": "moonshot",
        "model": "moonshot-v1-32k",
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "api_key": os.getenv("MOONSHOT_API_KEY", "sk-l5FvRWegjM5DEk4AU71YPQ1QgvFPTHZIJOmq6qdssPY4sNtE"),
    },

    "Doubao（豆包）": {
        "provider": "doubao",
        "model": "doubao-pro-32k",
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "api_key": os.getenv("DOUBAO_API_KEY", "sk-222afa3f-5f27-403e-bf46-ced2a356ceee"),
    },

    "Qwen（通义千问）": {
        "provider": "qwen",
        "model": "qwen-max",
        "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "api_key": os.getenv("QWEN_API_KEY", "sk-b3f7a1153e6f4a44804a296038aa86c5"),
    },
}

# ===============================
# 词类规则示例（保持不变）
# ===============================
RULE_SETS = {
    # 1.1 名词
    "名词": [
        {"name": "N1_可受数量词修饰", "desc": "可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "N2_不能受副词修饰", "desc": "不能受副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "N3_可作主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": 0},
        {"name": "N4_可作中心语或作定语", "desc": "可以做中心语受其他名词修饰，或者作定语直接修饰其他名词", "match_score": 10, "mismatch_score": 0},
        {"name": "N5_可后附的字结构", "desc": "可以后附助词“的”构成“的”字结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N6_可后附方位词构处所", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N7_不能作谓语核心", "desc": "不能做谓语或谓语核心（不能带宾语，不能受状语和补语，不能后附时体助词）", "match_score": 10, "mismatch_score": -10},
        {"name": "N8_不能作补语/一般不作状语", "desc": "不能作补语，并且一般不能做状语直接修饰动词性成分", "match_score": 10, "mismatch_score": 0},
    ],
    # 省略其他词类规则，保持原样
    # 1.2 时间词
    "时间词": [
        {"name": "T1_可作介宾或“的时候/以来”前", "desc": "可以作介词'在/到/从'和动词性结构'等到'的宾语，或在'的时候/以来'前", "match_score": 20, "mismatch_score": -20},
        {"name": "T2_不能受程度副词", "desc": "不能受副词'很'/'不'修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "T3_可作不典型主语", "desc": "可以做不典型的主语（有人称之为状语，此时一般可在前面加'在'）", "match_score": 10, "mismatch_score": -10},
        {"name": "T4_可做不典型谓语", "desc": "可以做不典型的谓语（后附'了'或受时间副词修饰时，主谓之间一般不能插入'是'）", "match_score": 10, "mismatch_score": 0},
        {"name": "T5_不能带宾语和补语", "desc": "不能带宾语和补语（不能作述语）", "match_score": 10, "mismatch_score": -10},
        {"name": "T6_可作时间中心语/作定语", "desc": "一般可以做中心语受其他时间词修饰，或作定语修饰时间词", "match_score": 10, "mismatch_score": 0},
        {"name": "T7_一般不能受名词修饰", "desc": "一般不能作中心语受名词直接修饰，也不能作定语直接修饰名词", "match_score": 10, "mismatch_score": 0},
        {"name": "T8_可后附'的'作定语但通常不作主宾", "desc": "可以后附助词'的'作定语，但一般不能作主语和宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "T9_可用'什么时候'提问/可用'这个时候'指代", "desc": "可以用'什么时候'提问或'这个时候/那个时候'指代", "match_score": 10, "mismatch_score": 0},
    ],
    # 其他词类规则保持不变...
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any], provider: str) -> str:
    """根据不同提供商提取响应文本"""
    if not isinstance(resp_json, dict):
        return ""
    
    try:
        # 使用每个模型配置中定义的响应处理器
        if provider in MODEL_CONFIGS:
            return MODEL_CONFIGS[provider]["response_handler"](resp_json)
        
        # 通用提取方法
        if "choices" in resp_json:
            choices = resp_json["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if "message" in first and "content" in first["message"]:
                    return first["message"]["content"]
                if "content" in first:
                    return first["content"]
        
        # 通义千问等特殊格式
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
            
    except Exception as e:
        st.warning(f"提取响应文本时出错: {str(e)}")
    
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
        # 显示调试信息（可选）
        with st.expander("查看API请求详情", expanded=False):
            st.write(f"URL: {url}")
            st.write("Headers:", headers)
            st.write("Payload:", payload)

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        # 显示响应状态
        with st.expander("查看API响应状态", expanded=False):
            st.write(f"状态码: {r.status_code}")
            st.write("响应内容:", r.text[:1000])  # 只显示前1000字符
            
        if r.status_code != 200:
            # 增强错误信息
            error_detail = f"HTTP错误 {r.status_code}: {r.text[:500]}"
            return False, {"error": error_detail, "content": r.text}, error_detail
            
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
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str, max_tokens: int, temperature: float) -> Tuple[Dict[str, Dict[str, int]], str, str]:
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
        "你是语言学研究专家。请根据以下规则，判断输入中文词语的词类隶属度。"
        "你的任务是：1. 预测最可能的词类。2. 对每个词类，根据其下的规则进行打分（是/符合为match_score，否/不符合为mismatch_score）。"
        "请严格返回以下JSON格式，不要添加任何其他说明文字："
        '{"predicted_pos":"<词类名>", "scores": {"<词类名>": {"<规则名>": <得分>, ...}, ...}, "explanation":"简要说明"}'
    )
    user_prompt = f"词语：『{word}』\n请基于下列规则判定并评分：\n\n{rules_text}\n\n仅返回严格 JSON。"

    ok, resp_json, err_msg = call_llm_api(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_prompt}],
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature
    )

    if not ok or not resp_json:
        # 调用失败或返回为空
        return {}, f"调用失败或返回异常: {err_msg}", "未知"

    # 尝试解析原始文本，传入provider参数以便正确提取
    raw_text = extract_text_from_response(resp_json, provider)
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
    st.plotly_chart(fig)

# ===============================
# 主页面逻辑
# ===============================
def main():
    # 侧边栏模型选择
    with st.sidebar:
        st.title("模型设置")
        selected_model = st.selectbox(
            "选择模型",
            list(MODEL_OPTIONS.keys())
        )
        
        # 显示选中模型的信息并允许修改API Key
        model_info = MODEL_OPTIONS[selected_model]
        st.text(f"提供商: {model_info['provider']}")
        st.text(f"模型名称: {model_info['model']}")
        st.text(f"API地址: {model_info['api_url'][:50]}...")
        
        # 允许用户输入API Key
        api_key = st.text_input(
            "API Key",
            value=model_info["api_key"],
            type="password"
        )
        
        # 其他参数设置
        max_tokens = st.slider("最大 tokens", 512, 4096, 2048)
        temperature = st.slider("温度参数", 0.0, 1.0, 0.0, 0.1)

    # 主页面内容
    st.title("汉语词类隶属度检测")
    
    # 输入词语
    word = st.text_input("请输入要检测的汉语词语", "")
    
    if st.button("开始检测"):
        if not word:
            st.warning("请输入词语后再检测")
            return
            
        with st.spinner(f"正在使用 {selected_model} 检测词语『{word}』的词类隶属度..."):
            # 调用模型进行检测
            scores, raw_text, predicted_pos = ask_model_for_pos_and_scores(
                word=word,
                provider=model_info["provider"],
                model=model_info["model"],
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # 显示结果
            st.success(f"检测完成！最可能的词类: {predicted_pos}")
            
            # 显示原始响应
            with st.expander("查看模型原始响应", expanded=False):
                st.text(raw_text)
                
            # 计算并显示归一化分数
            if scores:
                st.subheader("词类隶属度分数（归一化）")
                scores_norm = {}
                for pos, pos_scores in scores.items():
                    total = sum(pos_scores.values())
                    max_total = MAX_SCORES.get(pos, 1)  # 避免除以零
                    if max_total == 0:
                        norm_score = 0.0
                    else:
                        # 归一化到0-1范围
                        norm_score = (total + max_total) / (2 * max_total)
                        norm_score = max(0.0, min(1.0, norm_score))  # 确保在0-1之间
                    scores_norm[pos] = norm_score
                
                # 显示分数表格
                scores_df = pd.DataFrame(list(scores_norm.items()), columns=["词类", "隶属度"])
                scores_df = scores_df.sort_values(by="隶属度", ascending=False)
                st.dataframe(scores_df.style.format({"隶属度": "{:.2%}"}))
                
                # 绘制雷达图
                st.subheader("词类隶属度雷达图")
                plot_radar_chart_streamlit(scores_norm, f"词语『{word}』的词类隶属度分布")

if __name__ == "__main__":
    main()




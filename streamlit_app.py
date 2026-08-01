import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
import time
import logging
import fcntl
from typing import Tuple, Dict, Any, List
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from pathlib import Path

# ===============================
# 基础配置与日志
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_log.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="基于大语言模型的汉语隶属度检测划类平台",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ===============================
# 终极 CSS 样式修复方案
# ===============================
custom_css = """
<style>
/* ===== 全局基础样式 ===== */
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebar"] {display: none !important;}
.stApp > div:first-child {padding-top: 1rem;}

/* ===== 主背景渐变 ===== */
.stApp {
    background: linear-gradient(135deg, #f0f4fa 0%, #e4e9f2 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 95% !important;
}

/* ===== 标题高亮卡片 ===== */
.title-header-card {
    background: linear-gradient(135deg, #0f2942 0%, #1e4d7b 40%, #2d6cb8 75%, #3d8bd6 100%);
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(15, 41, 66, 0.35);
    position: relative;
    overflow: hidden;
}
.title-header-card h1 {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.title-header-card .subtitle {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.95rem !important;
    margin-top: 0.5rem !important;
    font-style: italic;
}

/* ===== 解决问题一与三：重写所有 Container 边框封装层，实现色差和圆角底图 ===== */
/* 外层板块（如模型设置）底色设定 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
    border-radius: 16px !important;
    border: 1px solid #d1d9e6 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05) !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}
/* 内层/Tab内板块（词语输入、结果）底色区分设定 */
div[data-testid="stTabs"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
}

/* ===== 解决问题二：强制重写标签页(Tabs)为独立的圆角按钮 ===== */
div[data-testid="stTabs"] {
    margin-top: 1rem;
}
/* 目标精确锁定到 role="tab" 的按钮身上 */
div[data-testid="stTabs"] button[role="tab"] {
    background-color: #e2e8f0 !important; /* 未选中时的灰色背景 */
    border-radius: 12px !important; /* 强制圆角 */
    border: 1px solid #cbd5e1 !important;
    margin-right: 12px !important;
    padding: 10px 24px !important;
    color: #475569 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    height: auto !important;
    min-height: 46px !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
    background-color: #cbd5e1 !important;
    color: #1e3a5f !important;
}
/* 选中状态的圆角高亮按钮 */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important;
    color: #ffffff !important;
    border-color: #1e4d7b !important;
    box-shadow: 0 4px 12px rgba(30, 77, 123, 0.3) !important;
}
/* 屏蔽 Streamlit 默认的下划线动画与底部边框 */
div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
    border-bottom: none !important;
    gap: 0 !important;
    background-color: transparent !important;
}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ===== 标题统一样式 ===== */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #1e3a5f !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}
.section-title .icon-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2d6cb8 0%, #3d8bd6 100%);
    box-shadow: 0 2px 6px rgba(45, 108, 184, 0.4);
}

/* 按钮及组件美化 */
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 700 !important; color: white !important; box-shadow: 0 4px 15px rgba(30, 77, 123, 0.3) !important; width: 100%; transition: all 0.3s ease !important; }
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(30, 77, 123, 0.4) !important; }
.stTextInput > div > div > input, .stSelectbox > div > div > div { border-radius: 10px !important; border: 2px solid #e2e8f0 !important; }
.stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus { border-color: #2d6cb8 !important; box-shadow: 0 0 0 3px rgba(45, 108, 184, 0.1) !important; }
.result-success-card { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 12px; padding: 1.25rem; border-left: 5px solid #10b981; margin: 1rem 0; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1); }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# 全局常量
BASE_DIR = Path(__file__).parent
BACKUP_FILE = BASE_DIR / "batch_history_log.csv"
PROGRESS_FILE = BASE_DIR / "process_progress.json"

RULE_SETS = {
    "名词": [
        {"name": "N1_可受数量词修饰", "desc": "可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "N2_不能受副词修饰", "desc": "不能受副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "N3_可作主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": 0},
        {"name": "N4_可作中心语或作定语", "desc": "可以做中心语受其他名词修饰，或者作定语直接修饰其他名词", "match_score": 10, "mismatch_score": 0},
        {"name": "N5_可后附的字结构", "desc": "可以后附助词'的'构成'的'字结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N6_可后附方位词构处所", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N7_不能作谓语核心", "desc": "不能做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "N8_不能作补语/一般不作状语", "desc": "不能作补语，并且一般不能做状语直接修饰动词性成分", "match_score": 10, "mismatch_score": 0},
    ],
    "动词": [
        {"name": "V1_可受否定'不/没有'修饰", "desc": "可以受否定副词'不'或'没有'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "V2_可后附/插入时体助词'着/了/过'", "desc": "可以后附或中间插入时体助词'着/了/过'", "match_score": 10, "mismatch_score": 0},
        {"name": "V3_可带真宾语或通过介词引导论元", "desc": "可以带真宾语或通过介词引导论元", "match_score": 20, "mismatch_score": 0},
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "V5_可有重叠/正反重叠形式", "desc": "可以有'VV, V一V, V了V, V不V, V了没有'等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "V6_可做谓语或谓语核心", "desc": "可以做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "V7_不能作状语修饰动词性成分", "desc": "不能作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "V8_可作'怎么/怎样'提问或'这么/这样/那么'回答", "desc": "可以跟在'怎么/怎样'之后提问或跟在'这么/这样/那么'之后回答", "match_score": 10, "mismatch_score": 0},
        {"name": "V9_不能跟在'多/多么'之后提问或表示感叹", "desc": "不能跟在'多'之后对性质提问，不能跟在'多么'之后表示感叹", "match_score": 10, "mismatch_score": -10},
    ],
    "名动词": [
        {"name": "NV1_可被\"不/没有\"否定且肯定形式-1", "desc": "可以用\"不\"和\"没有\"来否定", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "可以有\"VV、V一V、V了V、V不V\"等重叠和正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语或谓语核心，又可以作主语或宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后提问", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多\"之后对性质的程度进行提问", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
    ]
}

# ===============================
# 模型配置
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": True, 
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": True,
        },
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": True,
        },
    },
}

MODEL_OPTIONS = {
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Google Gemini 1.5": {"provider": "gemini", "model": "models/gemini-1.5-pro", "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"},
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS:
    AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 核心业务逻辑函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict): return ""
    try:
        if "output" in resp_json and "text" in resp_json["output"]: return resp_json["output"]["text"]
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]: return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception: return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    if not match: return None, text
    json_text = match.group(1).strip()
    try: return json.loads(json_text), json_text
    except json.JSONDecodeError: return None, json_text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_norm = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        if re.sub(r'[\s_]+', '', r["name"]).upper() == k_norm: return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    try:
        if isinstance(raw_val, bool): return match_score if raw_val else mismatch_score
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if s in ("yes", "y", "true", "是", "√", "符合"): return match_score
            if s in ("no", "n", "false", "否", "×", "不符合"): return mismatch_score
    except Exception: pass
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    for pos, scores in scores_all.items():
        normalized = sum(scores.values()) / 100
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3):
    if not _api_key: return False, {"error": "API Key为空"}, "API Key未提供"
    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}/{cfg['endpoint'].lstrip('/')}"
    headers, payload = cfg["headers"](_api_key), cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    full_content = ""
    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
                if response.status_code != 200: break
                for line in response.iter_lines():
                    if not line: continue
                    line_text = line.decode('utf-8').strip()
                    json_str = line_text[5:].strip() if line_text.startswith("data:") else line_text
                    if json_str == "[DONE]": break
                    try:
                        chunk = json.loads(json_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            delta_text = choice.get("delta", {}).get("content", "") or choice.get("message", {}).get("content", "")
                            full_content += delta_text
                    except json.JSONDecodeError: continue
            if full_content: return True, {"choices": [{"message": {"content": full_content}}]}, ""
        except Exception as e: time.sleep(1)
    return False, {}, "请求失败"

def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str):
    if not word: return {}, "", "未知", ""
    full_rules = {pos: "\n".join([f"- {r['name']}: {r['desc']}" for r in rules]) for pos, rules in RULE_SETS.items()}
    system_msg = f"""分析词语「{word}」。词类规则如下：\n【名词】\n{full_rules["名词"]}\n【动词】\n{full_rules["动词"]}\n【名动词】\n{full_rules["名动词"]}\n
请给出JSON，包含 explanation (详细推理)、predicted_pos (名词/动词/名动词) 和 scores (每一条规则返回 boolean 的 true/false)。最后单独输出完整 JSON。"""
    user_prompt = f"分析词语「{word}」。"
    ok, resp_json, err = call_llm_api_cached(provider, model, api_key, [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}])
    if not ok: return {}, err, "未知", err
    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)
    if not parsed_json: return {}, raw_text, "未知", "解析失败"
    
    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    raw_scores = parsed_json.get("scores", {})
    for pos, rules in RULE_SETS.items():
        if pos in raw_scores:
            for k, v in raw_scores[pos].items():
                norm_k = normalize_key(k, rules)
                if norm_k: scores_out[pos][norm_k] = map_to_allowed_score(next(r for r in rules if r["name"] == norm_k), v)
        for rule in rules:
            if rule["name"] not in scores_out[pos]: scores_out[pos][rule["name"]] = 0
            
    return scores_out, raw_text, parsed_json.get("predicted_pos", "未知"), parsed_json.get("explanation", "")

def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    categories = list(scores_norm.keys()) + [list(scores_norm.keys())[0]]
    values = list(scores_norm.values()) + [list(scores_norm.values())[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself")])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-0.1, 1.0])), showlegend=False, title=dict(text=title, x=0.5))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 主页面逻辑（全盘替换为 Native Container 布局）
# ===============================
def main():
    st.markdown("""
    <div class="title-header-card">
        <h1>基于大语言模型的汉语词类隶属度检测划类平台</h1>
        <div class="subtitle">Chinese Membership Detection and Classification Platform Based on Large Language Models (LLMs)</div>
    </div>
    """, unsafe_allow_html=True)

    # 重点改进 1 & 3：使用原生的 st.container(border=True) 自动生成底图板块
    with st.container(border=True):
        st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型设置 (LLM) & 连接测试</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if not AVAILABLE_MODEL_OPTIONS:
                st.selectbox("选择大模型 (不可用)", list(MODEL_OPTIONS.keys()), disabled=True)
            else:
                selected_model_display_name = st.selectbox("选择大模型", list(AVAILABLE_MODEL_OPTIONS.keys()), key="model_select", label_visibility="collapsed")
                selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
                st.markdown(f"✅ 已配置提供商: **{selected_model_info['provider'].upper()}**")
                
        with col2:
            if st.button("🔌 测试模型连接", type="secondary", use_container_width=True):
                with st.spinner("测试中..."):
                    ok, _, err = call_llm_api_cached(selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"], [{"role": "user", "content": "pong"}], max_tokens=10)
                if ok: st.success("连接成功！")
                else: st.error("连接失败！")

    # 重点改进 2：强制 CSS 渲染原生的 Tabs 标签
    tab1, tab2 = st.tabs(["单个词语详细分析", "Excel 批量处理"])

    with tab1:
        with st.container(border=True):
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 词语输入</div>', unsafe_allow_html=True)
            word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑...", label_visibility="collapsed")
            analyze_button = st.button("开始分析", type="primary")

        if analyze_button and word:
            with st.spinner(f"正在分析「{word}」..."):
                scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(word, selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"])
            
            if scores_all:
                membership = calculate_membership(scores_all)
                st.markdown(f"""
                <div class="result-success-card">
                    词语「<strong>{word}</strong>」最可能的词类是 <span style="background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 6px;">{predicted_pos}</span>，隶属度为 <strong>{membership.get(predicted_pos, 0):.4f}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    with st.container(border=True):
                        st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度雷达图</div>', unsafe_allow_html=True)
                        plot_radar_chart_streamlit(dict(get_top_10_positions(membership)), f"「{word}」的隶属度")
                with col_res2:
                    with st.container(border=True):
                        st.markdown('<div class="section-title"><span class="icon-dot"></span> 详细得分与推理过程</div>', unsafe_allow_html=True)
                        st.write(explanation)

    with tab2:
        with st.container(border=True):
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 批量任务上传与处理</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])
            if uploaded_file:
                st.info("上传成功，准备处理中...")

if __name__ == "__main__":
    main()

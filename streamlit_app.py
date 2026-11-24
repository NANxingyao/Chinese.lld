import streamlit as st
import requests
import json
import re
import os
import hmac
import pandas as pd
import plotly.graph_objects as go
import time
import logging
import subprocess
import sys
import traceback
from typing import Tuple, Dict, Any, List
from pathlib import Path

SERVICE_MODE = "--worker" in sys.argv or "--supervisor" in sys.argv
CODE_VERSION = "2026-09-15-stop-on-model-error-v1"

# Gemini 原生 responseSchema：把输出格式锁死为固定 JSON（短规则码 + boolean）
def _bool_props(codes):
    return {c: {"type": "BOOLEAN"} for c in codes}

GEMINI_FIXED_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "explanation": {"type": "STRING"},
        "predicted_pos": {"type": "STRING", "enum": ["名词", "动词", "名动词"]},
        "is_dual_category": {"type": "BOOLEAN"},
        "scores": {
            "type": "OBJECT",
            "properties": {
                "名词": {
                    "type": "OBJECT",
                    "properties": _bool_props([f"N{i}" for i in range(1, 9)]),
                    "required": [f"N{i}" for i in range(1, 9)],
                },
                "动词": {
                    "type": "OBJECT",
                    "properties": _bool_props([f"V{i}" for i in range(1, 10)]),
                    "required": [f"V{i}" for i in range(1, 10)],
                },
                "名动词": {
                    "type": "OBJECT",
                    "properties": _bool_props([f"NV{i}" for i in range(1, 11)]),
                    "required": [f"NV{i}" for i in range(1, 11)],
                },
            },
            "required": ["名词", "动词", "名动词"],
        },
    },
    "required": ["explanation", "predicted_pos", "is_dual_category", "scores"],
}

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

if not SERVICE_MODE:
    st.set_page_config(
        page_title="汉语词类隶属度检测划类平台",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None
    )

# ===============================
# 自定义 CSS（彻底隐藏顶部 GitHub/Share 栏，采用两栏现代科研工作台布局）
# ===============================
custom_css = """
<style>
/* ===== 彻底隐藏顶部 Streamlit/GitHub 工具栏与侧边栏 ===== */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu,
footer {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    visibility: hidden !important;
}

/* ===== 全局画布 ===== */
.stApp {
    background: radial-gradient(circle at 12% 8%, #E0E7FF 0%, #F4F7FB 38%, #F8FAFC 100%);
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1240px !important;
}

/* ===== 顶部轻量化标题栏（无笨重纯色大块） ===== */
.top-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    padding: 1.15rem 1.5rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #2563EB;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
    margin-bottom: 1.25rem;
}
.top-hero-left {
    display: flex;
    align-items: center;
    gap: 0.95rem;
}
.hero-icon-box {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    flex-shrink: 0;
}
.top-hero h1 {
    font-size: 1.38rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.3 !important;
}
.top-hero .subtitle {
    font-size: 0.82rem !important;
    color: #64748B !important;
    margin-top: 0.15rem !important;
}
.hero-tags {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.hero-tag {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
}
.hero-tag.blue {background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE;}
.hero-tag.purple {background: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE;}
.hero-tag.emerald {background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;}
.hero-tag.amber {background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A;}

/* ===== 区块标题 ===== */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    margin: 0.5rem 0 0.75rem 0 !important;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #E2E8F0;
}
.section-title .icon-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2563EB;
    flex-shrink: 0;
}

/* ===== 面板与卡片容器 ===== */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 16px !important;
    border-color: #E2E8F0 !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 1.15rem 1.3rem;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.05);
}

/* ===== 按钮样式 ===== */
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.22) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #4338CA 100%) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"],
.stFormSubmitButton > button[kind="secondary"] {
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px !important;
    padding: 0.35rem 0.75rem !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: #334155 !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stFormSubmitButton > button[kind="secondary"]:hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
}

/* ===== 输入框 ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
}

/* ===== Tabs 导航栏 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.45rem;
    background: #E2E8F0;
    padding: 0.32rem;
    border-radius: 12px;
    margin-bottom: 1.1rem;
    width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    height: 2.25rem;
    border-radius: 9px !important;
    padding: 0 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #475569 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1D4ED8 !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
}

/* ===== 初始引导：三类规则体系卡片 ===== */
.guide-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
}
.guide-card.noun {border-top: 4px solid #10B981;}
.guide-card.verb {border-top: 4px solid #2563EB;}
.guide-card.nv {border-top: 4px solid #7C3AED;}
.guide-card h4 {
    margin: 0 0 0.4rem 0 !important;
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}
.guide-card p {
    margin: 0 !important;
    font-size: 0.82rem !important;
    color: #475569 !important;
    line-height: 1.55 !important;
}

/* ===== 结果展示卡片 ===== */
.result-success-card {
    background: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
    border-radius: 14px;
    padding: 1.05rem 1.3rem;
    border: 1px solid #A7F3D0;
    border-left: 5px solid #10B981;
    margin: 0.9rem 0 1rem 0;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}
.kpi-card.noun {border-top: 3px solid #10B981;}
.kpi-card.verb {border-top: 3px solid #2563EB;}
.kpi-card.nv {border-top: 3px solid #7C3AED;}
.kpi-label {font-size: 0.8rem; color: #64748B; font-weight: 600;}
.kpi-value {font-size: 1.38rem; font-weight: 700; color: #0F172A; margin-top: 0.15rem;}

.info-highlight {
    background: #EFF6FF;
    border-radius: 10px;
    padding: 0.8rem 1.05rem;
    border-left: 4px solid #2563EB;
    margin: 0.6rem 0;
}

/* ===== 排名卡片 ===== */
.rank-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
}
.rank-card.top-1 {
    background: linear-gradient(90deg, #FFFBEB 0%, #FFFFFF 100%);
    border-color: #FDE68A;
}
.rank-card .rank-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #E2E8F0;
    color: #475569;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.82rem;
}
.rank-card.top-1 .rank-num {background: #F59E0B; color: #FFFFFF;}
.rank-card.top-2 .rank-num {background: #64748B; color: #FFFFFF;}
.rank-card.top-3 .rank-num {background: #D97706; color: #FFFFFF;}

/* ===== 状态徽章与运行提示 ===== */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
}
.status-badge.success {background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0;}
.status-badge.warn {background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A;}

.running-status {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0 1rem 0;
    border-radius: 12px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    font-weight: 600;
}
.running-spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    min-width: 16px;
    border: 2.5px solid rgba(37, 99, 235, 0.22);
    border-top-color: #2563EB;
    border-radius: 50%;
    animation: batch-spin 0.75s linear infinite;
    flex: 0 0 auto;
}
@keyframes batch-spin {to {transform: rotate(360deg);}}
.batch-detail {margin-top: 0.25rem; font-size: 0.85rem; font-weight: 500; color: #475569;}
</style>
"""
if not SERVICE_MODE:
    st.markdown(custom_css, unsafe_allow_html=True)

# ===============================
# 可选访问密码门禁（配置 APP_PASSWORD 后自动启用）
# ===============================
def _get_app_password() -> str:
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"]).strip()
    except Exception:
        pass
    return os.getenv("APP_PASSWORD", "").strip()

def check_optional_auth():
    expected = _get_app_password()
    if not expected or st.session_state.get("authenticated"):
        return

    st.markdown("""
    <div class="top-hero">
        <div class="top-hero-left">
            <div class="hero-icon-box">汉</div>
            <div>
                <h1>汉语词类隶属度检测划类平台</h1>
                <div class="subtitle">Chinese Membership Detection &amp; Classification · 访问验证</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.form("auth_form"):
        pwd = st.text_input("请输入访问密码", type="password", placeholder="输入管理员配置的访问口令")
        submitted = st.form_submit_button("验证并进入平台", type="primary", use_container_width=True)
        if submitted:
            if hmac.compare_digest(pwd, expected):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("访问密码错误，请重试。")
    st.stop()

# ===============================
# 全局常量与动态文件路径生成
# ===============================
BASE_DIR = Path(__file__).parent

def get_project_files(project_code: str, model_key: str = "") -> Tuple[Path, Path]:
    safe_code = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', project_code) or "default_task"
    safe_model = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', model_key or "default_model") or "default_model"
    namespace = f"{safe_code}__{safe_model}"
    return BASE_DIR / f"batch_history_{namespace}.csv", BASE_DIR / f"process_progress_{namespace}.json"

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

MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek", "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot", "model": "kimi-k2.6",
        "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"
    },
    "Qwen（通义千问）": {
        "provider": "qwen", "model": "qwen-max",
        "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"
    },
    "ChatGPT（GPT-5.6 Luna）": {
        "provider": "openai", "model": "gpt-5.6-luna",
        "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"
    },
    "ChatGPT（GPT-5.6 Terra）": {
        "provider": "openai", "model": "gpt-5.6-terra",
        "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"
    },
    "ChatGPT（GPT-5.6 Sol）": {
        "provider": "openai", "model": "gpt-5.6-sol",
        "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"
    },
    "Google Gemini 3.8 Flash": {
        "provider": "gemini", "model": "gemini-3.8-flash",
        "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"
    },
    "Google Gemini 3 Flash": {
        "provider": "gemini", "model": "gemini-3-flash-preview",
        "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"
    },
    "Google Gemini 3.1 Pro Preview": {
        "provider": "gemini", "model": "gemini-3.1-pro-preview",
        "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"
    },
    "xAI Grok 4.6": {
        "provider": "xai", "model": "grok-4.6",
        "api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY"
    },
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS:
    AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 安全操作辅助函数
# ===============================

def _lock_file(lock_path: Path, timeout=30, poll=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(str(lock_path)) > timeout * 2:
                    os.unlink(str(lock_path))
            except Exception:
                pass
            time.sleep(poll)
    raise TimeoutError(f"获取文件锁超时: {lock_path}")


def _unlock_file(lock_path: Path, fd):
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _make_task_id(file_name: str, row_index: int) -> str:
    return f"{file_name}::row::{row_index + 1}"


def clean_csv_duplicates(file_path: Path):
    if not file_path.exists() or file_path.stat().st_size == 0:
        return
    lock_path = file_path.with_suffix(file_path.suffix + '.maintenance.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=15)
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        changed = False
        key_col = '任务ID' if '任务ID' in df.columns else ('序数' if '序数' in df.columns else None)
        if key_col and df.duplicated(subset=[key_col]).any():
            df = df.drop_duplicates(subset=[key_col], keep='last')
            changed = True
        if '序数' in df.columns:
            df['序数_num'] = pd.to_numeric(df['序数'], errors='coerce')
            if not df['序数_num'].is_monotonic_increasing:
                df = df.sort_values('序数_num')
                changed = True
            df = df.drop(columns=['序数_num'])
        if changed:
            tmp = file_path.with_suffix(file_path.suffix + '.clean.tmp')
            df.to_csv(tmp, index=False, encoding='utf-8-sig')
            os.replace(tmp, file_path)
    except Exception:
        logger.exception(f"清理并排序 CSV 失败: {file_path}")
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)


def append_unique_csv(df: pd.DataFrame, file_path: Path, task_id: str, max_retries=30):
    lock_path = file_path.with_suffix(file_path.suffix + '.write.lock')
    last_error = ''
    for attempt in range(1, max_retries + 1):
        fd = None
        try:
            fd = _lock_file(lock_path, timeout=5)
            if file_path.exists() and file_path.stat().st_size > 0:
                existing_all = pd.read_csv(file_path, encoding='utf-8-sig')
                if '任务ID' not in existing_all.columns:
                    if '序数' in existing_all.columns:
                        existing_all['任务ID'] = existing_all['序数'].apply(
                            lambda x: f'legacy::row::{int(x)}' if pd.notna(x) else f'legacy::row::{x}'
                        )
                    else:
                        existing_all['任务ID'] = [f'legacy::index::{i+1}' for i in range(len(existing_all))]
                existing_ids = set(existing_all['任务ID'].astype(str).tolist())
                if task_id in existing_ids:
                    return True, '该任务ID已经写入，跳过重复写入。', True
                combined = pd.concat([existing_all, df], ignore_index=True)
            else:
                combined = df.copy()

            if '序数' in combined.columns:
                combined['序数_num'] = pd.to_numeric(combined['序数'], errors='coerce')
                combined = combined.drop_duplicates(subset=['任务ID'], keep='last')
                combined = combined.sort_values('序数_num').drop(columns=['序数_num'])

            tmp_path = file_path.with_suffix('.write.tmp')
            combined.to_csv(tmp_path, index=False, encoding='utf-8-sig')
            os.replace(tmp_path, file_path)
            return True, '写入并重排序成功。', False
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"CSV原子合并写入失败（第{attempt}/{max_retries}次）: {last_error}")
            time.sleep(min(2, 0.2 * attempt))
        finally:
            if fd is not None:
                _unlock_file(lock_path, fd)
    return False, f"CSV写入最终失败：{last_error}", False


def is_task_processed(file_path: Path, task_id: str, row_number: int) -> bool:
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        cols = pd.read_csv(file_path, encoding='utf-8-sig', nrows=0).columns.tolist()
        if '任务ID' in cols:
            existing = pd.read_csv(file_path, encoding='utf-8-sig', usecols=['任务ID'])
            return task_id in set(existing['任务ID'].astype(str).tolist())
        if '序数' in cols:
            existing = pd.read_csv(file_path, encoding='utf-8-sig', usecols=['序数'])
            vals = pd.to_numeric(existing['序数'], errors='coerce').dropna().astype(int)
            return row_number in set(vals.tolist())
    except Exception:
        pass
    return False


def spawn_detached(cmd, cwd):
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs['start_new_session'] = True
    return subprocess.Popen(cmd, cwd=cwd, **kwargs)

# ===============================
# 文本解析工具
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        return ""
    try:
        if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
            return resp_json["output_text"]
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] or {}
            message = choice.get("message", {}) or {}
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, dict):
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    pass
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    elif isinstance(part, str) and part.strip():
                        parts.append(part)
                if parts:
                    return "".join(parts)
            delta = choice.get("delta", {}) or {}
            delta_content = delta.get("content", "")
            if isinstance(delta_content, str) and delta_content.strip():
                return delta_content
            for alt_key in ("reasoning_content", "reasoning", "output_text", "text"):
                alt = message.get(alt_key)
                if isinstance(alt, str) and alt.strip() and ("{" in alt or "scores" in alt.lower()):
                    return alt
        candidates = resp_json.get("candidates")
        if isinstance(candidates, list):
            parts = []
            for candidate in candidates:
                content = (candidate or {}).get("content", {}) or {}
                for part in content.get("parts", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            if parts:
                return "".join(parts)
        return ""
    except Exception:
        return ""

def _clean_possible_markdown_json(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text).replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "").strip()
    cleaned = re.sub(r"^\s*```(?:json|JSON|javascript|js)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()

def _repair_json_candidate(candidate: str) -> str:
    if not candidate:
        return candidate
    candidate = candidate.strip()
    candidate = re.sub(r"^\s*```(?:json|JSON)?\s*", "", candidate)
    candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
    candidate = (candidate
                 .replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("「", '"').replace("」", '"'))
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    out = []
    in_string = False
    escaped = False
    for ch in candidate:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 32:
                out.append(" ")
                continue
        else:
            if ch == '"':
                in_string = True
        out.append(ch)
    return "".join(out)

def _iter_balanced_json_objects(text: str):
    if not text:
        return
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, n):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break

def _parse_json_object_candidate(candidate: str):
    try:
        obj = json.loads(candidate, strict=False)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    repaired = _repair_json_candidate(candidate)
    try:
        obj = json.loads(repaired, strict=False)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    if not text:
        return None, text
    cleaned = _clean_possible_markdown_json(text)
    obj = _parse_json_object_candidate(cleaned)
    if isinstance(obj, dict):
        return obj, cleaned
    for candidate in _iter_balanced_json_objects(cleaned):
        obj = _parse_json_object_candidate(candidate)
        if isinstance(obj, dict):
            return obj, candidate
    return None, text

def _gemini_prefer_schema_objects(text: str):
    preferred, others = [], []
    for candidate in _iter_balanced_json_objects(text):
        low = candidate.lower()
        if ("scores" in low or "predicted_pos" in low or "is_dual_category" in low or
                "名词" in candidate or "动词" in candidate):
            preferred.append(candidate)
        else:
            others.append(candidate)
    for c in preferred + others:
        yield c

def _gemini_try_close_truncated(candidate: str) -> str:
    if not candidate:
        return candidate
    opens = candidate.count("{") - candidate.count("}")
    if opens <= 0:
        return candidate
    return candidate + ("}" * min(opens, 8))

def _truthy_token(s: str) -> bool:
    t = (s or "").strip().lower()
    if t in ("true", "yes", "y", "1", "是", "对", "符合", "√", "✓", "match", "positive"):
        return True
    if t in ("false", "no", "n", "0", "否", "不", "不符合", "×", "✗", "x", "mismatch", "negative"):
        return False
    if "不符合" in t or "不能" in t:
        return False
    if "符合" in t or "可以" in t:
        return True
    return False

def _ultra_salvage_from_any_text(text: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return None
    raw = str(text)
    result = {
        "explanation": "（极宽松本地抢救解析）",
        "predicted_pos": "未知",
        "is_dual_category": False,
        "scores": {"名词": {}, "动词": {}, "名动词": {}},
    }
    pos_patterns = [
        r'["\']?predicted_pos["\']?\s*[:=]\s*["\']?\s*(名动词|名词|动词)',
        r'预测词类\s*[：:=]\s*(名动词|名词|动词)',
        r'最可能的?词类(?:是|为)?\s*[：:=]?\s*(名动词|名词|动词)',
        r'(?:属于|判定为|归为)\s*(名动词|名词|动词)',
        r'\b(名动词|名词|动词)\b\s*(?:隶属度最高|得分最高|最典型)',
    ]
    for pat in pos_patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            result["predicted_pos"] = m.group(1)
            break

    m = re.search(
        r'["\']?is_dual_category["\']?\s*[:=]\s*(true|false|True|False|是|否)|'
        r'(?:是否)?兼类\s*[：:=]?\s*(是|否|true|false)',
        raw, re.IGNORECASE
    )
    if m:
        val = (m.group(1) or m.group(2) or "").strip().lower()
        result["is_dual_category"] = val in ("true", "是")

    rule_pat = re.compile(
        r'["\']?\s*((?:NV|N|V)\s*\d+[_\-\u4e00-\u9fa5A-Za-z0-9]*)\s*["\']?\s*[:=：]\s*'
        r'(true|false|True|False|是|否|yes|no|符合|不符合|√|×|✓|✗|1|0)',
        re.IGNORECASE
    )
    narrative_pat = re.compile(
        r'((?:NV|N|V)\s*\d+)\s*[^。；;\n]{0,40}?(符合|不符合|可以|不能|是|否)',
        re.IGNORECASE
    )

    def _bucket(key: str) -> str:
        k = re.sub(r'\s+', '', key).upper()
        if k.startswith("NV"): return "名动词"
        if k.startswith("N"): return "名词"
        if k.startswith("V"): return "动词"
        return "名词"

    for rm in rule_pat.finditer(raw):
        key = re.sub(r'\s+', '', rm.group(1).strip())
        result["scores"][_bucket(key)][key] = _truthy_token(rm.group(2))

    if not any(result["scores"][p] for p in result["scores"]):
        for rm in narrative_pat.finditer(raw):
            key = re.sub(r'\s+', '', rm.group(1).strip())
            result["scores"][_bucket(key)][key] = _truthy_token(rm.group(2))

    for pos in ("名词", "动词", "名动词"):
        block_m = re.search(rf'["\']?{pos}["\']?\s*[:=：]\s*\{{(.*?)\}}', raw, re.DOTALL)
        if not block_m:
            continue
        for rm in rule_pat.finditer(block_m.group(1)):
            key = re.sub(r'\s+', '', rm.group(1).strip())
            result["scores"][pos][key] = _truthy_token(rm.group(2))

    has_scores = any(result["scores"][p] for p in result["scores"])
    if not has_scores and result["predicted_pos"] == "未知":
        return None
    if result["predicted_pos"] == "未知" and has_scores:
        counts = {p: sum(1 for v in result["scores"][p].values() if v) for p in result["scores"]}
        best = max(counts, key=lambda k: counts[k])
        if counts[best] > 0:
            result["predicted_pos"] = best
    return result

def extract_gemini_json(text: str) -> Tuple[Dict[str, Any], str]:
    if not text:
        return None, text
    cleaned = _clean_possible_markdown_json(text)
    obj = _parse_json_object_candidate(cleaned)
    if isinstance(obj, dict) and ("scores" in obj or "predicted_pos" in obj or "is_dual_category" in obj):
        return obj, cleaned
    for candidate in _gemini_prefer_schema_objects(cleaned):
        for cand in (candidate, _gemini_try_close_truncated(candidate)):
            obj = _parse_json_object_candidate(cand)
            if isinstance(obj, dict) and ("scores" in obj or "predicted_pos" in obj or "is_dual_category" in obj):
                return obj, cand
    for candidate in _iter_balanced_json_objects(cleaned):
        for cand in (candidate, _gemini_try_close_truncated(candidate)):
            obj = _parse_json_object_candidate(cand)
            if isinstance(obj, dict):
                return obj, cand
    salvaged = _ultra_salvage_from_any_text(cleaned)
    if salvaged is not None:
        return salvaged, cleaned
    salvaged = _ultra_salvage_from_any_text(text)
    if salvaged is not None:
        return salvaged, text
    return None, text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str):
        return None
    k_clean = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        if re.sub(r'[\s_]+', '', r["name"]).upper() == k_clean:
            return r["name"]
    code_hits = []
    for r in pos_rules:
        code_match = re.match(r'^(NV\d+|N\d+|V\d+)', re.sub(r'[\s_]+', '', r["name"]).upper())
        if not code_match:
            continue
        code = code_match.group(1)
        if k_clean == code:
            code_hits.append((len(code), r["name"]))
        elif k_clean.startswith(code) and not k_clean[len(code):len(code) + 1].isdigit():
            code_hits.append((len(code), r["name"]))
    if code_hits:
        code_hits.sort(key=lambda x: -x[0])
        return code_hits[0][1]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match, mismatch = rule["match_score"], rule["mismatch_score"]
    try:
        if isinstance(raw_val, bool): return match if raw_val else mismatch
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if s in ("yes", "y", "true", "是", "√", "符合"): return match
            if s in ("no", "n", "false", "否", "×", "不符合"): return mismatch
        if isinstance(raw_val, (int, float)):
            return match if int(raw_val) == match else mismatch
    except Exception:
        pass
    return mismatch

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    try:
        for pos, scores in scores_all.items():
            membership[pos] = max(-1.0, min(1.0, sum(scores.values()) / 100))
    except Exception:
        pass
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    try:
        return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        return []

# ===============================
# LLM 调用与词类判定主函数
# ===============================
def get_provider_config(provider, api_key, model, messages, max_tokens, temperature):
    if provider == "openai":
        url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": messages, "max_output_tokens": max_tokens}
        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
        if reasoning_effort in {"none", "low", "medium", "high", "xhigh", "max"}:
            payload["reasoning"] = {"effort": reasoning_effort}
        return url, headers, payload, "responses"

    if provider == "gemini":
        native_base = os.getenv("GEMINI_NATIVE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{native_base}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        contents, system_bits = [], []
        for m in messages or []:
            role = (m.get("role") or "user").lower()
            text = m.get("content") or ""
            if role == "system":
                system_bits.append(str(text))
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": str(text)}]})
            else:
                contents.append({"role": "user", "parts": [{"text": str(text)}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "\n\n".join(system_bits) or "请输出 JSON"}]}]
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "high").strip().lower()
        if thinking_level not in {"low", "medium", "high"}:
            thinking_level = "high"
        payload = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_FIXED_SCHEMA,
                "maxOutputTokens": max(max_tokens, 16384),
                "temperature": 1.0,
                "thinkingConfig": {"includeThoughts": True, "thinkingLevel": thinking_level},
            },
        }
        if system_bits:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_bits)}]}
        return url, headers, payload, "gemini_native"

    base_urls = {
        "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "moonshot": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        "qwen": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "xai": os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    }
    url = f"{base_urls.get(provider, base_urls['deepseek']).rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": provider not in {"moonshot"}}
    if provider == "moonshot":
        payload.update({"max_tokens": 8192, "thinking": {"type": "disabled"}, "temperature": 0.6})
    elif provider == "xai":
        reasoning_effort = os.getenv("XAI_REASONING_EFFORT", "high").strip().lower()
        if reasoning_effort in {"low", "medium", "high", "xhigh"}:
            payload["reasoning_effort"] = reasoning_effort
        payload["max_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    return url, headers, payload, "chat_completions"

def _extract_openai_responses_text(body: Dict[str, Any]) -> str:
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]
    output = body.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                for part in item["content"]:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        return "".join(chunks)
    return ""

def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3, show_ui=True):
    if not _api_key:
        return False, {"error": "API Key 为空"}, "API Key 未提供"
    url, headers, payload, api_style = get_provider_config(_provider, _api_key, _model, messages, max_tokens, temperature)
    streaming_placeholder = st.empty() if show_ui else None
    error_msg = ""

    for attempt in range(max_retries):
        try:
            if api_style == "gemini_native":
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code != 200:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    if response.status_code == 404: error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif response.status_code == 401: error_msg = "Gemini 鉴权失败 (401)。请检查 GEMINI_API_KEY。"
                    elif response.status_code == 429: error_msg = f"Gemini 限流 (429)：{detail}"
                    elif response.status_code in [400, 403]: error_msg = f"Gemini 请求错误 ({response.status_code})：{detail}"
                    else: error_msg = f"Gemini API 错误: {response.status_code} - {detail}"
                    if response.status_code in [400, 401, 403]:
                        break
                    response.raise_for_status()

                body = response.json()
                thought_parts, answer_parts = [], []
                try:
                    for candidate in body.get("candidates") or []:
                        for part in ((candidate or {}).get("content") or {}).get("parts", []) or []:
                            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                                if part.get("thought") is True:
                                    thought_parts.append(part["text"])
                                else:
                                    answer_parts.append(part["text"])
                except Exception:
                    thought_parts, answer_parts = [], []

                final_text = "".join(answer_parts).strip() or extract_text_from_response(body).strip()
                thought_text = "\n\n".join(t.strip() for t in thought_parts if t.strip())
                if final_text:
                    if streaming_placeholder is not None:
                        streaming_placeholder.empty()
                    return True, {"choices": [{"message": {"content": final_text, "reasoning_content": thought_text}}], "_raw_gemini": body}, ""
                error_msg = f"Gemini native 未返回最终 JSON。响应摘要：{json.dumps(body, ensure_ascii=False)[:1200]}"

            elif api_style == "responses":
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code != 200:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    error_msg = f"OpenAI 请求错误 ({response.status_code})：{detail}"
                    if response.status_code in [400, 401, 403]:
                        break
                    response.raise_for_status()
                body = response.json()
                text = _extract_openai_responses_text(body)
                if text:
                    if streaming_placeholder is not None:
                        streaming_placeholder.empty()
                    return True, {"choices": [{"message": {"content": text}}], "_raw_responses": body}, ""
                error_msg = "OpenAI Responses API 未返回有效文本"
            else:
                is_stream = bool(payload.get("stream", False))
                with requests.post(url, headers=headers, json=payload, stream=is_stream, timeout=180) as response:
                    if response.status_code != 200:
                        try:
                            detail = response.json()
                        except Exception:
                            detail = response.text
                        error_msg = f"{_provider.upper()} 请求错误 ({response.status_code})：{detail}"
                        if response.status_code in [400, 401, 403]:
                            break
                        response.raise_for_status()
                    if not is_stream:
                        body = response.json()
                        text = extract_text_from_response(body)
                        if text:
                            if streaming_placeholder is not None:
                                streaming_placeholder.empty()
                            return True, body, ""
                        error_msg = f"{_provider.upper()} 非流式接口未返回文本"
                    else:
                        full_content = ""
                        for line in response.iter_lines():
                            if not line:
                                continue
                            line_text = line.decode("utf-8", errors="ignore").strip()
                            if not line_text.startswith("data:"):
                                continue
                            json_str = line_text[5:].strip()
                            if json_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(json_str)
                                if "choices" in chunk and chunk["choices"]:
                                    delta_text = (chunk["choices"][0].get("delta", {}) or {}).get("content", "") or ""
                                    if delta_text:
                                        full_content += delta_text
                            except json.JSONDecodeError:
                                continue
                        if full_content:
                            if streaming_placeholder is not None:
                                streaming_placeholder.empty()
                            return True, {"choices": [{"message": {"content": full_content}}]}, ""
                        error_msg = "模型未返回有效内容"
        except requests.RequestException as e:
            error_msg = f"网络请求异常（第{attempt + 1}次尝试）: {str(e)}"
        except Exception as e:
            error_msg = f"请求异常（第{attempt + 1}次尝试）: {str(e)}"
        if attempt < max_retries - 1:
            time.sleep(min(8, 2 ** attempt))

    if streaming_placeholder is not None:
        streaming_placeholder.empty()
    return False, {"error": error_msg}, error_msg

def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str, show_ui=True) -> Tuple[Dict[str, Dict[str, int]], str, str, str, bool]:
    if not word:
        return {}, "", "未知", "", False
    full_rules = {pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules]) for pos, rules in RULE_SETS.items()}
    is_gemini = provider == "gemini"
    is_gemini_flash = is_gemini and "flash" in str(model).lower()

    if is_gemini:
        def _short_rules(pos):
            lines = []
            for r in RULE_SETS[pos]:
                code = re.match(r'^(NV\d+|N\d+|V\d+)', r["name"])
                code = code.group(1) if code else r["name"]
                lines.append(f"- {code}: {r['desc']}")
            return "\n".join(lines)

        system_msg = f"""你是一名中文词法与语法方面的专家。现在请严格分析词语「{word}」在名词、动词、名动词三类中的表现。

你的任务有两个层次：
第一层：进行充分的内部思考，逐条检查全部规则，结合真实、自然的现代汉语用法判断。API 会单独返回思考摘要，这部分不写进最终 JSON。
第二层：给出最终结构化结果。最终结果必须严格符合给定 schema。

【名词规则】
{_short_rules("名词")}

【动词规则】
{_short_rules("动词")}

【名动词规则】
{_short_rules("名动词")}

最终 JSON 的 explanation 字段必须是详细、可公开查看的分析依据，不得只写一句笼统结论。请：
1. 按照 N1-N8、V1-V9、NV1-NV10 的顺序逐条说明“为什么符合/不符合”；
2. 每条规则尽量结合该词的自然句法环境、搭配或例句说明判断依据；
3. 不要遗漏任何一条规则；
4. predicted_pos 选择最典型的词类；
5. is_dual_category 仅根据整体词类属性判断；
6. scores 中所有规则必须填写 true 或 false。

注意：不要把分析说明写到 schema 之外；不要输出 Markdown；最终可见答案只能是符合 schema 的 JSON。"""
        user_prompt = f"请严格按照上述规则分析词语「{word}」。先充分思考并核查每一条规则，再输出完整 JSON；不要省略任何规则的判断依据和例证。"
    else:
        system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：
- 需要判断的词类：名词、动词、名动词
- 你只需要判断每一条规则是"符合"还是"不符合"，在 JSON 中的 scores 里给出 true / false，程序自动赋值。
【名词】\n{full_rules["名词"]}\n【动词】\n{full_rules["动词"]}\n【名动词】\n{full_rules["名动词"]}
输出要求：
1. explanation: 逐条规则说明判断依据并举例（写在 JSON 的 explanation 字段中）。
2. scores: 各规则对应 true/false。
3. predicted_pos: 选择最典型词类。
4. is_dual_category: 是否属于兼类（true/false）。

严格直接返回一段合法 JSON，不要输出任何 Markdown 外层文本或开场白。格式：
{{"explanation": "...", "predicted_pos": "...", "is_dual_category": true, "scores": {{"名词": {{...}}, "动词": {{...}}, "名动词": {{...}}}}}}"""
        user_prompt = f"请分析词语「{word}」。只返回一个 JSON 对象；不要输出 Markdown、```、前言、结语或 JSON 之外的任何文字。所有解释文字必须放在 explanation 字段中。"

    with st.spinner(f"正在调用大模型 ({model}) 进行分析...") if show_ui else __import__("contextlib").nullcontext():
        ok, resp_json, err_msg = call_llm_api_cached(
            provider, model, api_key,
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}],
            show_ui=show_ui, max_retries=1
        )

    if not ok:
        if show_ui:
            st.error(f"模型调用失败: {err_msg}")
        if is_gemini_flash:
            empty_scores = {pos: {r["name"]: 0 for r in rules} for pos, rules in RULE_SETS.items()}
            return empty_scores, f"调用失败: {err_msg}", "调用失败", f"失败: {err_msg}", False
        return {}, f"调用失败: {err_msg}", "未知", f"失败: {err_msg}", False

    if provider == "gemini":
        raw_content, gemini_reasoning = "", ""
        try:
            message = ((resp_json.get("choices") or [{}])[0] or {}).get("message", {})
            raw_content = message.get("content", "") or ""
            gemini_reasoning = message.get("reasoning_content", "") or ""
        except Exception:
            pass
        raw_text = str(raw_content).strip()
        raw_text_with_reasoning = (
            f"【Gemini 思考摘要】\n{gemini_reasoning.strip()}\n\n【Gemini 最终 JSON】\n{raw_text}"
            if gemini_reasoning.strip() else raw_text
        )
    else:
        raw_text = extract_text_from_response(resp_json)
        raw_text_with_reasoning = raw_text

    if provider == "gemini":
        parsed_json, _ = extract_gemini_json(raw_text)
        if parsed_json is None:
            parsed_json, _ = extract_json_from_text(raw_text)
    else:
        parsed_json, _ = extract_json_from_text(raw_text)
        if parsed_json is None:
            parsed_json, _ = extract_gemini_json(raw_text)

    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "无推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        is_dual_category = parsed_json.get("is_dual_category", False)
        raw_scores = parsed_json.get("scores", {})
        if not isinstance(raw_scores, dict):
            salv = _ultra_salvage_from_any_text(raw_text or "")
            raw_scores = (salv or {}).get("scores", {}) if salv else {}
    else:
        if show_ui:
            st.error("未能解析有效的 JSON。")
        empty_scores = {pos: {r["name"]: 0 for r in rules} for pos, rules in RULE_SETS.items()}
        return empty_scores, raw_text or "", "解析失败", "JSON 解析失败", False

    scores_out = {pos: {r["name"]: 0 for r in rules} for pos, rules in RULE_SETS.items()}
    for pos, rules in RULE_SETS.items():
        if pos in raw_scores and isinstance(raw_scores[pos], dict):
            for k, v in raw_scores[pos].items():
                norm_k = normalize_key(k, rules)
                if norm_k:
                    rule_def = next(r for r in rules if r["name"] == norm_k)
                    scores_out[pos][norm_k] = map_to_allowed_score(rule_def, v)
    return scores_out, raw_text_with_reasoning, predicted_pos, explanation, is_dual_category

def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm:
        st.warning("无数据。")
        return
    categories = list(scores_norm.keys()) + [list(scores_norm.keys())[0]]
    values = list(scores_norm.values()) + [list(scores_norm.values())[0]]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.22)",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(color="#4F46E5", size=7),
                hovertemplate='<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>'
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#F8FAFC",
            radialaxis=dict(
                visible=True,
                range=[min(min(values), -0.1), max(max(values), 1.0)],
                gridcolor="#CBD5E1",
                linecolor="#CBD5E1",
                tickfont=dict(size=10, color="#64748B")
            ),
            angularaxis=dict(
                gridcolor="#CBD5E1",
                linecolor="#CBD5E1",
                tickfont=dict(size=13, color="#1E293B")
            )
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=40, r=40, t=45, b=25),
        title=dict(text=title, x=0.5, font=dict(size=15, color="#0F172A"))
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 独立进程批处理：Supervisor + Worker
# ===============================

def _write_job_state(state_file: Path, **updates):
    lock_path = state_file.with_suffix(state_file.suffix + '.state.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=10, poll=0.1)
        current = {}
        if state_file.exists():
            try:
                current = json.loads(state_file.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"读取任务状态失败，将以空状态继续：{type(e).__name__}: {e}")
        current.update(updates)
        current['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        tmp = state_file.with_suffix(state_file.suffix + '.tmp')
        tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, state_file)
        return True
    except Exception as e:
        logger.exception(f"写入任务状态失败: {state_file}: {e}")
        return False
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)

def _load_job_state(state_file: Path):
    try:
        return json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    except Exception:
        return {}

def _pid_alive(pid):
    if not pid:
        return False
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, OSError):
        return False

def _kill_pid(pid):
    if not pid:
        return False
    try:
        pid = int(pid)
        if pid <= 0 or not _pid_alive(pid):
            return False
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            return True
        time.sleep(0.4)
        if _pid_alive(pid):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        return True
    except Exception as e:
        logger.warning(f"结束进程失败 pid={pid}: {e}")
        return False

def stop_batch_job(job_state_file: Path) -> str:
    state = _load_job_state(job_state_file)
    supervisor_pid = state.get("supervisor_pid")
    worker_pid = state.get("worker_pid")
    for suffix, key in ((".supervisor.pid", "supervisor"), (".worker.pid", "worker")):
        try:
            p = _service_file(job_state_file, suffix)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt.isdigit():
                    if key == "supervisor": supervisor_pid = int(txt)
                    else: worker_pid = int(txt)
        except Exception:
            pass

    _write_job_state(job_state_file, status="cancelled", error="用户手动停止任务", worker_pid=None)
    killed = []
    if _kill_pid(worker_pid): killed.append(f"worker={worker_pid}")
    if _kill_pid(supervisor_pid): killed.append(f"supervisor={supervisor_pid}")
    for suffix in (".supervisor.pid", ".worker.pid", ".supervisor.lock"):
        try:
            f = _service_file(job_state_file, suffix)
            if f.exists(): f.unlink()
        except Exception:
            pass
    return f"已停止任务，并结束进程：{', '.join(killed)}" if killed else "已标记停止（cancelled）。若界面仍在转，请刷新页面。"

def _state_age_seconds(state):
    try:
        ts = state.get('updated_at')
        return max(0, time.time() - time.mktime(time.strptime(ts, '%Y-%m-%d %H:%M:%S'))) if ts else 0
    except Exception:
        return 0

def _service_file(job_state_file: Path, suffix: str) -> Path:
    return job_state_file.with_name(job_state_file.stem + suffix)

def _batch_worker(df_input, target_col, file_name, backup_file, provider, model, api_key, job_state_file):
    total_rows = len(df_input)
    state = _load_job_state(job_state_file)
    start_row = int(state.get('next_row', 0)) if state else 0
    if state.get('file_name') != file_name or start_row < 0 or start_row > total_rows:
        start_row = 0

    clean_csv_duplicates(backup_file)
    _write_job_state(
        job_state_file, status='running', file_name=file_name, total_rows=total_rows,
        next_row=start_row, current_row=start_row, current_word='', retry_count=0, error='', completed_rows=start_row
    )

    while start_row < total_rows:
        if _load_job_state(job_state_file).get("status") == "cancelled":
            _write_job_state(job_state_file, status="cancelled", error="用户手动停止任务", next_row=start_row, completed_rows=start_row)
            return

        index = start_row
        row_number = index + 1
        word_val = df_input.iloc[index][target_col]
        word = "" if pd.isna(word_val) else str(word_val).strip()
        task_id = _make_task_id(file_name, index)

        if not word:
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row, current_row=index, current_word='', completed_rows=start_row, retry_count=0, error='')
            continue

        if is_task_processed(backup_file, task_id, row_number):
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row, current_row=index, current_word=word, completed_rows=start_row, retry_count=0, error='检测到该 Excel 行已有结果，已跳过重复处理')
            continue

        try:
            _write_job_state(job_state_file, status='running', current_row=index, current_word=word, next_row=index, retry_count=0, error='', completed_rows=index)
            scores, raw_text, pred_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(word, provider, model, api_key, show_ui=False)
            if not scores or str(pred_pos) in {'解析失败', '调用失败', '处理失败', '未知'}:
                detail = f'第 {row_number} 行「{word}」处理失败（{pred_pos}），批次已停止。{explanation or ""}'
                _write_job_state(job_state_file, status='failed', current_row=index, current_word=word, next_row=index, completed_rows=index, retry_count=0, error=detail, model_error=True)
                return
        except BaseException as e:
            detail = f'第 {row_number} 行「{word}」模型请求异常，批次已停止：{type(e).__name__}: {e}'
            _write_job_state(job_state_file, status='failed', current_row=index, current_word=word, next_row=index, completed_rows=index, retry_count=0, error=detail, model_error=True)
            return

        membership = calculate_membership(scores)
        new_row = pd.DataFrame([{
            '任务ID': task_id, '序数': row_number, '词语': word,
            '动词': membership.get('动词', 0.0), '名词': membership.get('名词', 0.0), '名动词': membership.get('名动词', 0.0),
            '差值/距离': round(abs(membership.get('动词', 0.0) - membership.get('名词', 0.0)), 4),
            '预测词类': pred_pos, '是否兼类': '是' if is_dual_category else '否',
            '模型提供商': provider, '模型': model, '原始响应': raw_text, '时间戳': time.strftime('%Y-%m-%d %H:%M:%S')
        }])

        _write_job_state(job_state_file, status='saving', current_row=index, current_word=word, next_row=index, completed_rows=index, retry_count=0, error='')
        write_ok, write_detail, already_exists = append_unique_csv(new_row, backup_file, task_id)
        save_retry = 0
        while not write_ok:
            save_retry += 1
            delay = min(30, max(2, 2 ** min(save_retry - 1, 4)))
            _write_job_state(job_state_file, status='waiting_save_retry', current_row=index, current_word=word, next_row=index, completed_rows=index, retry_count=save_retry, error=write_detail, retry_in_seconds=delay)
            time.sleep(delay)
            write_ok, write_detail, already_exists = append_unique_csv(new_row, backup_file, task_id)

        start_row = index + 1
        _write_job_state(job_state_file, status='running', current_row=index, current_word=word, next_row=start_row, completed_rows=start_row, retry_count=0, error=write_detail if already_exists else '')

    _write_job_state(job_state_file, status='completed', file_name=file_name, total_rows=total_rows, next_row=total_rows, completed_rows=total_rows, current_word='', retry_count=0, error='')

def _worker_entry(job_state_file: Path):
    try:
        spec_file = _service_file(job_state_file, '.spec.json')
        spec = json.loads(spec_file.read_text(encoding='utf-8'))
        api_key = os.getenv(spec['env_var'], '')
        if not api_key:
            raise RuntimeError(f"环境变量 {spec['env_var']} 未配置")
        df_input = pd.read_excel(Path(spec['input_file']))
        _batch_worker(df_input, spec['target_col'], spec['file_name'], Path(spec['backup_file']), spec['provider'], spec['model'], api_key, job_state_file)
    except BaseException as e:
        _write_job_state(job_state_file, status='failed', error=f'{type(e).__name__}: {e}\n{traceback.format_exc()}', worker_fatal_error=True)
        raise

def _acquire_pid_guard(lock_file: Path):
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode('utf-8'))
        return fd
    except FileExistsError:
        try:
            old_pid = int(lock_file.read_text(encoding='utf-8').strip())
        except Exception:
            old_pid = 0
        if not _pid_alive(old_pid):
            try: lock_file.unlink()
            except FileNotFoundError: pass
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode('utf-8'))
            return fd
        raise RuntimeError(f'同一批次已有 Supervisor 在运行（PID {old_pid}）')

def _release_pid_guard(lock_file: Path, fd):
    try: os.close(fd)
    finally:
        try: lock_file.unlink()
        except FileNotFoundError: pass

def _supervisor_entry(job_state_file: Path):
    supervisor_pid_file = _service_file(job_state_file, '.supervisor.pid')
    worker_pid_file = _service_file(job_state_file, '.worker.pid')
    supervisor_lock_file = _service_file(job_state_file, '.supervisor.lock')
    guard_fd = None
    try:
        guard_fd = _acquire_pid_guard(supervisor_lock_file)
        supervisor_pid_file.write_text(str(os.getpid()), encoding='utf-8')
        restart_count = int(_load_job_state(job_state_file).get('supervisor_restart_count', 0) or 0)
        while True:
            state = _load_job_state(job_state_file)
            if state.get('status') in {'completed', 'cancelled'}:
                return
            worker = spawn_detached([sys.executable, str(Path(__file__).resolve()), '--worker', str(job_state_file)], str(Path(__file__).resolve().parent))
            worker_pid_file.write_text(str(worker.pid), encoding='utf-8')
            restart_count += 1
            _write_job_state(job_state_file, status='running', supervisor_pid=os.getpid(), worker_pid=worker.pid, supervisor_restart_count=restart_count, error='')
            while True:
                rc = worker.poll()
                state = _load_job_state(job_state_file)
                if state.get('status') == 'completed': return
                if state.get('status') == 'cancelled':
                    try:
                        if rc is None:
                            worker.terminate()
                            time.sleep(0.3)
                            if worker.poll() is None: worker.kill()
                    except Exception: pass
                    return
                if rc is not None: break
                time.sleep(1)
            current_state = _load_job_state(job_state_file)
            if current_state.get('status') == 'failed' and current_state.get('model_error'):
                return
            delay = min(30, max(2, 2 ** min(restart_count - 1, 4)))
            _write_job_state(job_state_file, status='supervisor_restarting', supervisor_pid=os.getpid(), worker_pid=None, supervisor_restart_count=restart_count, error=f'Worker退出(code={rc})，{delay}秒后自动重启')
            time.sleep(delay)
    except BaseException as e:
        _write_job_state(job_state_file, status='failed', supervisor_pid=os.getpid(), worker_pid=None, error=f'Supervisor异常：{e}', supervisor_fatal_error=True)
        raise
    finally:
        for f in (supervisor_pid_file, worker_pid_file):
            try:
                if f.exists() and f.read_text(encoding='utf-8').strip() in {'', str(os.getpid())}: f.unlink()
            except Exception: pass
        if guard_fd is not None:
            _release_pid_guard(supervisor_lock_file, guard_fd)

def start_or_resume_batch_job(df_input, target_col, uploaded_file, file_name, backup_file, provider, model, env_var, job_state_file):
    state = _load_job_state(job_state_file)
    if _pid_alive(state.get('supervisor_pid')):
        return False, state
    safe_stem = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', job_state_file.stem)
    spec = {
        'input_file': str(BASE_DIR / f"batch_input_{safe_stem}.xlsx"),
        'target_col': target_col, 'file_name': file_name, 'backup_file': str(backup_file),
        'provider': provider, 'model': model, 'env_var': env_var
    }
    Path(spec['input_file']).write_bytes(uploaded_file.getvalue())
    spec_tmp = _service_file(job_state_file, '.spec.json.tmp')
    spec_tmp.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(spec_tmp, _service_file(job_state_file, '.spec.json'))
    resume_row = int(state.get('next_row', 0)) if state.get('file_name') == file_name else 0
    _write_job_state(job_state_file, status='starting', file_name=file_name, total_rows=len(df_input), next_row=resume_row, completed_rows=resume_row, current_row=max(0, resume_row - 1), current_word='', retry_count=0, error='')
    proc = spawn_detached([sys.executable, str(Path(__file__).resolve()), '--supervisor', str(job_state_file)], str(Path(__file__).resolve().parent))
    _write_job_state(job_state_file, status='starting', supervisor_pid=proc.pid, worker_pid=None, error='Supervisor 已启动，等待 Worker')
    return True, _load_job_state(job_state_file)

def _supervisor_or_worker_alive(job_state_file: Path, state: dict = None) -> bool:
    state = state or _load_job_state(job_state_file)
    if _pid_alive(state.get("supervisor_pid")) or _pid_alive(state.get("worker_pid")):
        return True
    for suffix in (".supervisor.pid", ".worker.pid"):
        try:
            p = _service_file(job_state_file, suffix)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt.isdigit() and _pid_alive(int(txt)): return True
        except Exception: pass
    return False

def _reconcile_stale_job_state(job_state_file: Path) -> dict:
    state = _load_job_state(job_state_file)
    active_statuses = {"running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "starting", "supervisor_restarting"}
    if state.get("status") not in active_statuses or _supervisor_or_worker_alive(job_state_file, state):
        return state
    if state.get("status") == "starting" and _state_age_seconds(state) < 15:
        return state
    _write_job_state(job_state_file, status="interrupted", supervisor_pid=None, worker_pid=None, error="检测到后台进程已退出，任务已解锁。可点击「开始处理 / 继续断点任务」继续。")
    return _load_job_state(job_state_file)

def _auto_recover_if_needed(job_state_file: Path):
    state = _reconcile_stale_job_state(job_state_file)
    if os.getenv("BATCH_AUTO_RECOVER", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return state
    active_statuses = {'running', 'retrying', 'waiting_retry', 'saving', 'waiting_save_retry', 'starting', 'supervisor_restarting'}
    if state.get('status') not in active_statuses or _supervisor_or_worker_alive(job_state_file, state):
        return state
    if state.get('status') == 'starting' and _state_age_seconds(state) < 15:
        return state
    spec_file = _service_file(job_state_file, '.spec.json')
    if not spec_file.exists():
        return state
    proc = spawn_detached([sys.executable, str(Path(__file__).resolve()), '--supervisor', str(job_state_file)], str(Path(__file__).resolve().parent))
    _write_job_state(job_state_file, status='starting', supervisor_pid=proc.pid, worker_pid=None, error='系统正在自动重新启动并接续任务')
    return _load_job_state(job_state_file)

# ----------------------------------------------------------------------
# 批量结果清理
# ----------------------------------------------------------------------
def _get_bad_result_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty: return pd.Series(dtype=bool)
    mask = pd.Series(False, index=df.index)
    for col in ("预测词类", "状态", "错误", "说明", "explanation"):
        if col in df.columns:
            mask |= df[col].fillna("").astype(str).str.contains(r"调用失败|解析失败|处理失败|计算错误|JSON\s*解析失败|请求失败|未知", case=False, regex=True, na=False)
    score_cols = [c for c in ("动词", "名词", "名动词") if c in df.columns]
    if len(score_cols) == 3:
        nums = df[score_cols].apply(pd.to_numeric, errors="coerce")
        raw = df.get("原始响应", pd.Series("", index=df.index)).fillna("").astype(str)
        mask |= (nums.fillna(0).eq(0).all(axis=1) & raw.str.contains(r"调用失败|解析失败|计算错误|429|401|403|404", case=False, regex=True, na=False)) | nums.isna().any(axis=1)
    return mask

def remove_bad_batch_rows(backup_file: Path) -> Tuple[bool, int, str]:
    if not backup_file.exists() or backup_file.stat().st_size == 0:
        return True, 0, "当前没有可清理的批量结果。"
    lock_path = backup_file.with_suffix(backup_file.suffix + '.cleanup.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=15)
        df = pd.read_csv(backup_file, encoding='utf-8-sig')
        bad_mask = _get_bad_result_mask(df)
        removed = int(bad_mask.sum())
        if removed == 0: return True, 0, "未发现失败/解析失败/明显计算错误记录。"
        cleaned = df.loc[~bad_mask].copy()
        if '序数' in cleaned.columns:
            cleaned['_seq_num'] = pd.to_numeric(cleaned['序数'], errors='coerce')
            cleaned = cleaned.sort_values('_seq_num').drop(columns=['_seq_num'])
        tmp = backup_file.with_suffix(backup_file.suffix + '.cleanup.tmp')
        cleaned.to_csv(tmp, index=False, encoding='utf-8-sig')
        os.replace(tmp, backup_file)
        return True, removed, f"已删除 {removed} 条失败/解析失败/明显计算错误记录。"
    except Exception as e:
        return False, 0, f"清理失败：{type(e).__name__}: {e}"
    finally:
        if fd is not None: _unlock_file(lock_path, fd)

def remove_selected_batch_rows(backup_file: Path, task_ids: List[str]) -> Tuple[bool, int, str]:
    task_ids = {str(x) for x in (task_ids or []) if str(x).strip()}
    if not task_ids: return True, 0, "未勾选任何记录。"
    if not backup_file.exists() or backup_file.stat().st_size == 0: return True, 0, "当前没有可删除的批量结果。"
    lock_path = backup_file.with_suffix(backup_file.suffix + '.cleanup.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=15)
        df = pd.read_csv(backup_file, encoding='utf-8-sig')
        if '任务ID' not in df.columns: return False, 0, "当前结果文件没有“任务ID”列，无法安全删除。"
        mask = df['任务ID'].astype(str).isin(task_ids)
        removed = int(mask.sum())
        if removed == 0: return True, 0, "未找到需要删除的记录。"
        cleaned = df.loc[~mask].copy()
        if '序数' in cleaned.columns:
            cleaned['_seq_num'] = pd.to_numeric(cleaned['序数'], errors='coerce')
            cleaned = cleaned.sort_values('_seq_num').drop(columns=['_seq_num'])
        tmp = backup_file.with_suffix(backup_file.suffix + '.selected_cleanup.tmp')
        cleaned.to_csv(tmp, index=False, encoding='utf-8-sig')
        os.replace(tmp, backup_file)
        return True, removed, f"已删除 {removed} 条勾选记录。"
    except Exception as e:
        return False, 0, f"删除失败：{type(e).__name__}: {e}"
    finally:
        if fd is not None: _unlock_file(lock_path, fd)

# ----------------------------------------------------------------------
# 实时 UI 刷新组件
# ----------------------------------------------------------------------
def render_live_monitor(job_state_file: Path, backup_file: Path, total_rows_default: int):
    latest_state = _auto_recover_if_needed(job_state_file)
    total = int(latest_state.get("total_rows", total_rows_default) or total_rows_default)
    completed = int(latest_state.get("completed_rows", latest_state.get("next_row", 0)) or 0)
    status_str = str(latest_state.get("status", ""))
    error_str = str(latest_state.get("error", ""))

    live_df = None
    count = 0
    if backup_file.exists():
        try:
            live_df = pd.read_csv(backup_file, encoding='utf-8-sig')
            count = len(live_df)
        except Exception:
            pass

    c1, c2, c3 = st.columns([1.4, 1.4, 1.2], vertical_alignment="center")
    with c1:
        st.metric("已存数据量（去重排序）", f"{count} 条")
    with c2:
        pct = (completed / total * 100) if total else 0.0
        st.metric("当前处理进度", f"{completed} / {total} ({pct:.1f}%)")
    with c3:
        if live_df is not None and count > 0:
            csv_data = live_df.to_csv(index=False, encoding='utf-8-sig')
            model_slug = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', str(live_df.iloc[0].get('模型', 'model')) if '模型' in live_df.columns else 'model')
            st.download_button(
                "⬇ 下载完整结果 (CSV)", data=csv_data,
                file_name=f"{st.session_state.project_code}_{model_slug}_results_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv", type="primary", use_container_width=True
            )
        else:
            st.button("下载结果 (无数据)", disabled=True, use_container_width=True)

    st.progress(min(1.0, max(0.0, completed / total)) if total else 0.0)

    if status_str in {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}:
        spinner_text = {
            "starting": "启动任务中…", "retrying": "调用失败自动重试中…", "waiting_retry": "等待重试…",
            "saving": "安全排序并覆盖保存中…", "supervisor_restarting": "断流重连中…"
        }.get(status_str, "任务正常运行中…")
        details = f"第 {min(int(latest_state.get('current_row', 0)) + 1, total)}/{total} 行 · 当前词语「{latest_state.get('current_word', '')}」"
        if latest_state.get("retry_count", 0):
            details += f" · 重试 {latest_state.get('retry_count')} 次"
        status_html = f'<div class="running-status"><span class="running-spinner"></span><div style="flex:1"><div style="font-size:0.95rem;font-weight:700">{spinner_text}</div><div class="batch-detail">{details}</div>'
        if error_str:
            status_html += f'<div class="batch-detail" style="color: #EF4444;">提示：{error_str}</div>'
        status_html += '</div></div>'
        st.markdown(status_html, unsafe_allow_html=True)
    elif status_str == "completed":
        st.success(f"🎉 任务全部完毕，共 {total} 条。")
    elif status_str == "cancelled":
        st.warning(f"⏹ 任务已手动停止（进度约 {completed}/{total}）。可随时点「开始处理 / 继续断点任务」从断点续跑。")
    elif status_str == "interrupted":
        st.warning(f"⚠ 后台进程已退出，任务已解锁（进度约 {completed}/{total}）。点「开始处理 / 继续断点任务」即可续跑。")
    elif status_str == "failed":
        st.error(f"❌ 任务已停止：{error_str or '未记录到具体异常，请查看 process_log.log。'}")
    else:
        st.info("尚未运行。点击上方「▶ 开始处理 / 继续断点任务」即可启动队列。")

    st.markdown('<div class="section-title"><span class="icon-dot"></span> 实时结果预览（最新结果在最上方）</div>', unsafe_allow_html=True)
    if live_df is not None and count > 0:
        if '序数' in live_df.columns:
            live_df['序数_num'] = pd.to_numeric(live_df['序数'], errors='coerce')
            live_df = live_df.sort_values('序数_num').drop(columns=['序数_num'])
        display_df = live_df.iloc[::-1].copy()
        if '任务ID' in display_df.columns:
            editor_df = display_df.copy()
            editor_df.insert(0, '删除', False)
            edited_df = st.data_editor(
                editor_df, use_container_width=True, height=400, hide_index=True,
                disabled=[c for c in editor_df.columns if c != '删除'],
                column_config={'删除': st.column_config.CheckboxColumn('删除', help='勾选后点击下方“删除勾选记录”')},
                key=f'batch_editor_{job_state_file.stem}'
            )
            selected_ids = edited_df.loc[edited_df['删除'] == True, '任务ID'].astype(str).tolist() if '删除' in edited_df.columns else []
            del_col1, del_col2 = st.columns(2)
            with del_col1:
                if st.button('🗑 删除勾选记录', use_container_width=True, key=f'del_selected_{job_state_file.stem}'):
                    ok_del, _, msg = remove_selected_batch_rows(backup_file, selected_ids)
                    if ok_del: st.success(msg); st.rerun()
                    else: st.error(msg)
            with del_col2:
                if st.button('🧹 删除失败 / 计算错误记录', use_container_width=True, key=f'del_bad_{job_state_file.stem}'):
                    ok_del, _, msg = remove_bad_batch_rows(backup_file)
                    if ok_del: st.success(msg); st.rerun()
                    else: st.error(msg)
        else:
            st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("暂无数据。任务开启后实时数据将在此严格依序显示。")


def main():
    check_optional_auth()

    # ===============================
    # 顶部轻量化品牌栏（彻底告别笨重蓝框与 GitHub 顶栏）
    # ===============================
    st.markdown("""
    <div class="top-hero">
        <div class="top-hero-left">
            <div class="hero-icon-box">汉</div>
            <div>
                <h1>基于大语言模型的汉语词类隶属度检测划类平台</h1>
                <div class="subtitle">Chinese Membership Detection &amp; Classification Platform Based on LLMs</div>
            </div>
        </div>
        <div class="hero-tags">
            <span class="hero-tag blue">多模型支持</span>
            <span class="hero-tag purple">雷达图可视化</span>
            <span class="hero-tag amber">兼类判定</span>
            <span class="hero-tag emerald">抗断流恢复</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # 主体两栏工作台：左侧固定配置面板 + 右侧核心分析区
    # ===============================
    col_left, col_right = st.columns([1, 2.75], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown('<div class="section-title" style="margin-top:0;"><span class="icon-dot"></span> 模型设置（LLM）</div>', unsafe_allow_html=True)
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("未检测到 API Key，请在 Secrets 中配置。")
                st.selectbox("选择大模型", list(MODEL_OPTIONS.keys()), disabled=True, label_visibility="collapsed")
                selected_model_info = {"api_key": "", "provider": "deepseek", "model": "deepseek-chat", "env_var": "DEEPSEEK_API_KEY"}
            else:
                selected_model_display_name = st.selectbox(
                    "选择大模型",
                    list(AVAILABLE_MODEL_OPTIONS.keys()),
                    key="model_select",
                    label_visibility="collapsed"
                )
                selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
                if selected_model_info.get("api_key"):
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.45rem;margin:0.35rem 0 0.65rem 0;flex-wrap:wrap;">'
                        f'<span class="status-badge success">● 已就绪</span>'
                        f'<span style="color:#64748B;font-size:0.78rem;">{selected_model_info["provider"].upper()} · {selected_model_info["model"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<span class="status-badge warn">● 未配置 API Key</span>', unsafe_allow_html=True)

            if st.button("🔌 测试模型连接", type="secondary", use_container_width=True, disabled=not selected_model_info.get("api_key")):
                with st.spinner("测试中..."):
                    test_provider = selected_model_info["provider"]
                    test_max_tokens = 4096 if test_provider == "gemini" else 100
                    test_prompt = "请只输出 pong，不要输出 Markdown、代码块或其他文字。" if test_provider == "gemini" else "请回复'pong'"
                    ok, _, err_msg = call_llm_api_cached(
                        test_provider, selected_model_info["model"], selected_model_info["api_key"],
                        [{"role": "user", "content": test_prompt}], max_tokens=test_max_tokens
                    )
                if ok:
                    st.success("连接正常！")
                else:
                    st.error(f"失败: {err_msg}")

            st.divider()
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 实验配置</div>', unsafe_allow_html=True)
            st.session_state.project_code = st.text_input(
                "实验批次码 (Project Code)",
                value=st.session_state.get("project_code", "default_task"),
                help="用于隔离不同量化分析任务"
            )
            if selected_model_info.get("provider") in {"openai", "gemini", "xai"}:
                st.caption("数据按“批次 + 模型”独立保存，便于多模型横向对比。")

            BACKUP_FILE, PROGRESS_FILE = get_project_files(
                st.session_state.project_code,
                selected_model_info.get("model", "default_model")
            )

            st.divider()
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 评测体系</div>', unsafe_allow_html=True)
            st.caption("• **名词**：8 项句法功能规则\n\n• **动词**：9 项句法功能规则\n\n• **名动词**：10 项兼类判定规则")
            if not _get_app_password():
                st.caption("🔒 提示：可在 Secrets 中配置 `APP_PASSWORD` 启用访问密码。")

    with col_right:
        tab1, tab2 = st.tabs(["🔍 单个词语详细分析", "📊 Excel 批量处理"])

        # ------------------------------------------------------------------
        # Tab 1: 单个词语详细分析
        # ------------------------------------------------------------------
        with tab1:
            with st.form("single_word_form"):
                col_in, col_btn = st.columns([4, 1], vertical_alignment="center")
                with col_in:
                    word = st.text_input(
                        "请输入要分析的汉语词语",
                        value=st.session_state.get("demo_word", ""),
                        placeholder="请输入要分析的汉语词语，例如：苹果、跑、美丽、决定...",
                        max_chars=30,
                        label_visibility="collapsed",
                        key="word_input"
                    )
                with col_btn:
                    analyze_clicked = st.form_submit_button(
                        "开始分析",
                        type="primary",
                        use_container_width=True,
                        disabled=not selected_model_info.get("api_key")
                    )

                ex_cols = st.columns([1.2, 0.75, 0.75, 0.75, 0.75, 3.3], vertical_alignment="center")
                ex_cols[0].caption("快速示例：")
                for idx, sample in enumerate(["苹果", "跑", "美丽", "决定"]):
                    if ex_cols[idx + 1].form_submit_button(sample, use_container_width=True):
                        st.session_state["demo_word"] = sample
                        st.rerun()

            if analyze_clicked:
                clean_word = (word or "").strip()
                if not clean_word:
                    st.warning("请先输入要分析的汉语词语。")
                elif any(c in clean_word for c in ["\n", "\r", "{", "}"]):
                    st.warning("请输入合法的单个汉语词语。")
                else:
                    status_placeholder = st.empty()
                    status_placeholder.info(f"正在为词语「{clean_word}」启动分析...")
                    scores_all, raw_text, predicted_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(
                        clean_word,
                        selected_model_info["provider"],
                        selected_model_info["model"],
                        selected_model_info["api_key"]
                    )
                    status_placeholder.empty()

                    if scores_all:
                        membership = calculate_membership(scores_all)
                        final_membership = membership.get(predicted_pos, 0.0)
                        jianlei_badge_color = "#F59E0B" if is_dual_category else "#2563EB"
                        jianlei_text = "属于兼类词" if is_dual_category else "非兼类词"

                        st.markdown(
                            f'<div class="result-success-card">'
                            f'<div style="font-size: 1.02rem; font-weight: 700; color: #065F46;">✅ 分析完成</div>'
                            f'<div style="margin-top: 0.4rem; font-size: 0.96rem; color: #065F46; line-height: 1.8;">'
                            f'词语「<strong>{clean_word}</strong>」最可能的词类是 '
                            f'<span style="background: #10B981; color: white; padding: 0.16rem 0.6rem; border-radius: 6px; font-weight: 600;">{predicted_pos}</span>'
                            f'，隶属度为 <strong>{final_membership:.4f}</strong> &nbsp;|&nbsp; '
                            f'兼类属性：<span style="background: {jianlei_badge_color}; color: white; padding: 0.16rem 0.6rem; border-radius: 6px; font-weight: 600;">{jianlei_text}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True
                        )

                        k1, k2, k3 = st.columns(3)
                        with k1:
                            st.markdown(f'<div class="kpi-card verb"><div class="kpi-label">动词隶属度</div><div class="kpi-value">{membership.get("动词", 0.0):.4f}</div></div>', unsafe_allow_html=True)
                        with k2:
                            st.markdown(f'<div class="kpi-card noun"><div class="kpi-label">名词隶属度</div><div class="kpi-value">{membership.get("名词", 0.0):.4f}</div></div>', unsafe_allow_html=True)
                        with k3:
                            st.markdown(f'<div class="kpi-card nv"><div class="kpi-label">名动词隶属度</div><div class="kpi-value">{membership.get("名动词", 0.0):.4f}</div></div>', unsafe_allow_html=True)

                        col_results_1, col_results_2 = st.columns(2, gap="medium")
                        with col_results_1:
                            st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度排名</div>', unsafe_allow_html=True)
                            top10 = get_top_10_positions(membership)
                            for i, (pos, score) in enumerate(top10):
                                rank_class = f"top-{i+1}" if i < 3 else ""
                                st.markdown(
                                    f'<div class="rank-card {rank_class}">'
                                    f'<div style="display: flex; align-items: center; gap: 0.7rem;">'
                                    f'<div class="rank-num">{i+1}</div>'
                                    f'<span style="font-weight: 600; color: #1E293B;">{pos}</span>'
                                    f'</div>'
                                    f'<span style="font-weight: 700; color: #2563EB; font-size: 1.02rem;">{score:.4f}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

                            st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度雷达图</div>', unsafe_allow_html=True)
                            plot_radar_chart_streamlit(dict(top10), f"「{clean_word}」隶属度分布")

                        with col_results_2:
                            st.markdown('<div class="section-title"><span class="icon-dot"></span> 各词类详细得分</div>', unsafe_allow_html=True)
                            pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in scores_all.keys()}
                            for pos in sorted(pos_total_scores.keys(), key=lambda p: pos_total_scores[p], reverse=True):
                                total_score = pos_total_scores[pos]
                                max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                                with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                                    rule_data = [
                                        {
                                            "规则代码": k,
                                            "规则描述": next((r["desc"] for r in RULE_SETS.get(pos, []) if r["name"] == k), ""),
                                            "得分": v
                                        }
                                        for k, v in scores_all[pos].items()
                                    ]
                                    rule_df = pd.DataFrame(sorted(rule_data, key=lambda x: x["得分"], reverse=True))
                                    st.dataframe(
                                        rule_df.style.map(
                                            lambda x: "color: #EF4444; font-weight: bold" if isinstance(x, (int, float)) and x < 0 else "",
                                            subset=["得分"]
                                        ),
                                        use_container_width=True,
                                        height=min(len(rule_df) * 35 + 45, 400)
                                    )

                            if explanation:
                                st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型判定依据</div>', unsafe_allow_html=True)
                                with st.expander("展开查看逐条规则分析依据", expanded=True):
                                    st.write(explanation)

                            st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型原始响应</div>', unsafe_allow_html=True)
                            with st.expander("展开查看原始响应", expanded=False):
                                st.code(raw_text, language="text")
            else:
                # 未提交分析时的精致引导卡片，避免页面大片留白
                st.markdown('<div class="section-title"><span class="icon-dot"></span> 判定维度说明</div>', unsafe_allow_html=True)
                g1, g2, g3 = st.columns(3, gap="medium")
                with g1:
                    st.markdown("""
                    <div class="guide-card noun">
                        <h4>名词（8 项规则）</h4>
                        <p>考察数量词修饰、副词排斥、主宾语功能、定语/中心语、的字结构与方位词搭配等句法特征。</p>
                    </div>
                    """, unsafe_allow_html=True)
                with g2:
                    st.markdown("""
                    <div class="guide-card verb">
                        <h4>动词（9 项规则）</h4>
                        <p>考察否定副词修饰、时体助词“着/了/过”、真宾语、重叠形式及谓语核心充当能力。</p>
                    </div>
                    """, unsafe_allow_html=True)
                with g3:
                    st.markdown("""
                    <div class="guide-card nv">
                        <h4>名动词（10 项规则）</h4>
                        <p>综合评估兼具名词性与动词性的兼类特征，计算三类隶属度并输出直观的雷达图对比。</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Tab 2: Excel 批量处理
        # ------------------------------------------------------------------
        with tab2:
            with st.container(border=True):
                col_h1, col_h2 = st.columns([2.2, 1.5], vertical_alignment="center")
                with col_h1:
                    st.markdown(
                        f'<div class="section-title" style="margin:0;"><span class="icon-dot"></span> 批量任务管理 (批次: <code>{st.session_state.project_code}</code>)</div>',
                        unsafe_allow_html=True
                    )
                with col_h2:
                    confirm_clear = st.checkbox("确认清空本批次（不可恢复）", key="confirm_clear_batch")
                    if st.button("🗑 清空本批次记录", use_container_width=True, type="secondary", disabled=not confirm_clear):
                        if os.path.exists(BACKUP_FILE):
                            try:
                                os.remove(BACKUP_FILE)
                                st.success(f"已清空批次 {st.session_state.project_code} 记录")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"清空失败: {e}")

                uploaded_file = st.file_uploader("选择待处理的 Excel 文件（建议 10MB 以内）", type=["xlsx", "xls"])

            if uploaded_file:
                try:
                    if uploaded_file.size > 10 * 1024 * 1024:
                        st.error("文件大小超过 10MB 限制，请拆分后上传。")
                        st.stop()

                    df_input = pd.read_excel(uploaded_file)
                    target_col = next((col for col in df_input.columns if "词" in str(col) or "word" in str(col).lower()), None)
                    if target_col:
                        st.markdown(
                            f'<div class="info-highlight">'
                            f'<div style="font-weight: 600; color: #1E40AF;">✅ 识别到目标列: <code>{target_col}</code> &nbsp;|&nbsp; 待分析总数: <strong>{len(df_input)}</strong> 条</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        model_namespace = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', selected_model_info.get("model", "default_model"))
                        safe_proj = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', st.session_state.project_code)
                        job_state_file = BASE_DIR / f"batch_job_{safe_proj}__{model_namespace}.json"
                        current_state = _auto_recover_if_needed(job_state_file)

                        really_running = _supervisor_or_worker_alive(job_state_file, current_state)
                        can_start = not really_running
                        can_stop = really_running or current_state.get("status") in {
                            "starting", "running", "retrying", "waiting_retry", "saving",
                            "waiting_save_retry", "supervisor_restarting",
                        }

                        btn_col1, btn_col2, btn_col3 = st.columns([2.5, 1, 1])
                        with btn_col1:
                            if st.button("▶ 开始处理 / 继续断点任务", type="primary", use_container_width=True, disabled=not can_start):
                                if not selected_model_info["api_key"]:
                                    st.error("请配置有效的 API Key")
                                else:
                                    try:
                                        started, _ = start_or_resume_batch_job(
                                            df_input, target_col, uploaded_file,
                                            f"{st.session_state.project_code}_{uploaded_file.name}",
                                            BACKUP_FILE, selected_model_info["provider"],
                                            selected_model_info["model"], selected_model_info["env_var"],
                                            job_state_file,
                                        )
                                        if started:
                                            st.success("守护任务已启动，将在后台持续处理并自动恢复。")
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.warning("任务似乎仍在运行。若按钮一直灰，请先点「停止任务」或「强制解锁」。")
                                    except Exception as e:
                                        st.error(f"启动失败：{e}")
                        with btn_col2:
                            if st.button("⏹ 停止任务", type="secondary", use_container_width=True, disabled=not can_stop and not really_running):
                                msg = stop_batch_job(job_state_file)
                                st.warning(msg)
                                time.sleep(0.5)
                                st.rerun()
                        with btn_col3:
                            if st.button("🔓 强制解锁", type="secondary", use_container_width=True, help="进程已死但按钮仍灰时点这个"):
                                stop_batch_job(job_state_file)
                                _write_job_state(job_state_file, status="interrupted", supervisor_pid=None, worker_pid=None, error="用户强制解锁，可重新开始/继续")
                                st.info("已强制解锁，请再点「开始处理 / 继续断点任务」。")
                                time.sleep(0.4)
                                st.rerun()

                        st.divider()
                        if hasattr(st, "fragment"):
                            @st.fragment(run_every="2s")
                            def _live_batch_status_view():
                                render_live_monitor(job_state_file, BACKUP_FILE, len(df_input))
                            _live_batch_status_view()
                        else:
                            render_live_monitor(job_state_file, BACKUP_FILE, len(df_input))
                    else:
                        st.error("未在表格中识别到包含 '词' 或 'word' 字段的目标列")
                except Exception as e:
                    st.error(f"读取Excel文件失败: {e}")

if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_entry(Path(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--supervisor" in sys.argv:
        _supervisor_entry(Path(sys.argv[sys.argv.index("--supervisor") + 1]))
    else:
        main()

if not SERVICE_MODE:
    st.markdown(
        '<div style="text-align: center; color: #94A3B8; font-size: 13px; line-height: 1.8; padding: 18px 0 6px 0;">'
        '汉语词类隶属度检测划类平台 · © 2026 Ryan'
        '</div>',
        unsafe_allow_html=True
    )

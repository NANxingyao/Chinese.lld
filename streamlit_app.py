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
# 自定义CSS样式（深度UI优化版）
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
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
}

/* ===== 主容器卡片化 ===== */
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
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 40px rgba(15, 41, 66, 0.35);
    position: relative;
    overflow: hidden;
}
.title-header-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.title-header-card h1 {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    padding: 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}
.title-header-card .subtitle {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 0.95rem !important;
    margin-top: 0.5rem !important;
    font-style: italic;
    position: relative;
    z-index: 1;
}
.title-header-card .badges {
    margin-top: 1rem;
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.title-header-card .badge {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    color: #fff;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* ===== 模块高亮卡片 ===== */
.module-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(200, 210, 225, 0.5);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

/* ===== 解决问题一与三：通过 :has 伪类直接包裹 Streamlit 原生 Column，修复背景色分离断层与下拉框无背景色的问题 ===== */
div[data-testid="column"]:has(#model-settings-section) {
    background: linear-gradient(135deg, #fbfcfe 0%, #f0f4fa 100%) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    border: 1.5px solid rgba(45, 90, 135, 0.15) !important;
    box-shadow: 0 4px 16px rgba(45, 90, 135, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04) !important;
}
div[data-testid="column"]:has(#connection-test-section) {
    background: linear-gradient(135deg, #f5f9ff 0%, #e8f0fe 100%) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    border: 1.5px solid rgba(59, 130, 246, 0.2) !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1), 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    text-align: center;
}

/* 其他卡片样式 */
.result-success-card { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #10b981; margin: 1rem 0; }
.warning-highlight { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #f59e0b; margin: 0.75rem 0; }
.error-highlight { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #ef4444; margin: 0.75rem 0; }
.info-highlight { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #3b82f6; margin: 0.75rem 0; }

/* ===== 子标题样式 ===== */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
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
    flex-shrink: 0;
}

/* ===== 解决问题二：标签页样式优化（彻底修复无圆角问题） ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem !important;
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"], .stTabs button[role="tab"] {
    height: 3rem !important;
    border-radius: 16px !important; /* 强制圆角 */
    padding: 0 2rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"]:hover, .stTabs button[role="tab"]:hover {
    color: #1e4d7b !important;
    border-color: #2d6cb8 !important;
    background: #f5f9ff !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(30, 77, 123, 0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important; /* 必须隐藏原生的底边高亮线条，否则破坏圆角外观 */
}

/* ===== 其他通用组件样式优化 ===== */
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 50%, #3d8bd6 100%) !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 700 !important; font-size: 1rem !important; color: white !important; box-shadow: 0 6px 20px rgba(30, 77, 123, 0.4), 0 2px 6px rgba(0, 0, 0, 0.1) !important; transition: all 0.3s ease !important; width: 100%; letter-spacing: 0.5px; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #0f2942 0%, #1e4d7b 50%, #2d6cb8 100%) !important; box-shadow: 0 10px 30px rgba(30, 77, 123, 0.5), 0 4px 10px rgba(0, 0, 0, 0.15) !important; transform: translateY(-2px); }
.stButton > button[kind="secondary"] { background: #ffffff !important; border: 2px solid #cbd5e1 !important; border-radius: 12px !important; padding: 0.6rem 1.5rem !important; font-weight: 600 !important; color: #475569 !important; transition: all 0.3s ease !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04); }
.stButton > button[kind="secondary"]:hover { border-color: #2d6cb8 !important; color: #1e4d7b !important; background: #f5f9ff !important; transform: translateY(-1px); }
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div { border-radius: 12px !important; border: 2px solid #e2e8f0 !important; transition: all 0.3s ease !important; background: #fefefe !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus, .stSelectbox > div > div > div:focus { border-color: #2d6cb8 !important; box-shadow: 0 0 0 4px rgba(45, 108, 184, 0.1) !important; background: #ffffff !important; }
.stDataFrame { border-radius: 16px; overflow: hidden; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06); border: 1px solid #e2e8f0; }
.dataframe { font-size: 13px; }
.dataframe th { background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important; color: white !important; font-weight: 600 !important; padding: 0.85rem 1rem !important; }
.dataframe td { padding: 0.7rem 1rem !important; }
.dataframe tr:nth-child(even) { background: #f8fafc; }
.dataframe tr:hover { background: #eff6ff !important; }
.metric-card { background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%); border-radius: 16px; padding: 1.25rem 1.5rem; border: 1.5px solid rgba(45, 108, 184, 0.15); text-align: center; box-shadow: 0 4px 12px rgba(45, 108, 184, 0.08); }
.metric-card .metric-value { font-size: 2rem; font-weight: 800; color: #1e4d7b; }
.metric-card .metric-label { font-size: 0.9rem; color: #64748b; margin-top: 0.5rem; font-weight: 500; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #1e4d7b 0%, #2d6cb8 50%, #3d8bd6 100%) !important; border-radius: 10px !important; height: 12px !important; }
.stProgress > div > div > div { border-radius: 10px !important; background: #e2e8f0 !important; height: 12px !important; }
.streamlit-expanderHeader { background: #f8fafc; border-radius: 12px !important; padding: 0.85rem 1.25rem !important; font-weight: 600 !important; color: #1e4d7b !important; border: 1.5px solid #e2e8f0; transition: all 0.3s ease; }
.streamlit-expanderHeader:hover { background: #f0f7ff; border-color: #2d6cb8; }
.streamlit-expanderContent { background: #ffffff; border: 1.5px solid #e2e8f0; border-top: none; border-radius: 0 0 12px 12px !important; padding: 1.25rem !important; }
.stCodeBlock { border-radius: 12px !important; border: 1.5px solid #e2e8f0 !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important; }
.stFileUploader { background: #ffffff; border-radius: 16px; padding: 1.5rem; border: 2px dashed #cbd5e1; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03); }
.stFileUploader:hover { border-color: #2d6cb8; background: #f5f9ff; box-shadow: 0 4px 12px rgba(45, 108, 184, 0.1); }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%); margin: 1.5rem 0; }
.footer-text { text-align: center; color: #64748b; font-size: 0.85rem; padding: 1.5rem 0; margin-top: 2rem; border-top: 1px solid #e5e7eb; }
.status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; }
.status-badge.success { background: #d1fae5; color: #065f46; }
.status-badge.warning { background: #fef3c7; color: #92400e; }
.status-badge.error { background: #fee2e2; color: #991b1b; }
.rank-card { display: flex; align-items: center; justify-content: space-between; padding: 0.85rem 1.25rem; border-radius: 12px; margin-bottom: 0.6rem; background: linear-gradient(135deg, #fefefe 0%, #f5f7fa 100%); border: 1.5px solid #e2e8f0; transition: all 0.3s ease; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03); }
.rank-card:hover { background: linear-gradient(135deg, #f5f9ff 0%, #e8f0fe 100%); border-color: #2d6cb8; transform: translateX(6px); box-shadow: 0 4px 12px rgba(45, 108, 184, 0.12); }
.rank-card .rank-num { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%); color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08); }
.rank-card.top-1 .rank-num { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; box-shadow: 0 3px 8px rgba(251, 191, 36, 0.4); }
.rank-card.top-2 .rank-num { background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%); color: white; box-shadow: 0 3px 8px rgba(156, 163, 175, 0.4); }
.rank-card.top-3 .rank-num { background: linear-gradient(135deg, #d97706 0%, #b45309 100%); color: white; box-shadow: 0 3px 8px rgba(217, 119, 6, 0.4); }
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
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": True,
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}", 
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable",
            "Accept": "text/event-stream"
        },
        "payload": lambda model, messages, **kw: {
            "model": model, "input": {"messages": messages}, 
            "parameters": {
                "max_tokens": kw.get("max_tokens", 4096), 
                "temperature": kw.get("temperature", 0.0),
                "result_format": "message",
                "incremental_output": True 
            },
        },
    },
}

# ===============================
# 模型选项
# ===============================
MODEL_OPTIONS = {
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o（推荐）": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Google Gemini 1.5 Pro": {"provider": "gemini", "model": "models/gemini-1.5-pro", "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"},
    "Google Gemini 1.5 Flash": {"provider": "gemini", "model": "models/gemini-1.5-flash", "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"},
    "Moonshot（Kimi）": {"provider": "moonshot", "model": "moonshot-v1-32k", "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"},
    "Qwen（通义千问）": {"provider": "qwen", "model": "qwen-max", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS:
    AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict): return ""
    try:
        if "output" in resp_json and "text" in resp_json["output"]: return resp_json["output"]["text"]
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]: return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        logger.error(f"提取响应文本失败: {e}")
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    if not match: return None, text
    json_text = match.group(1).strip()
    try:
        return json.loads(json_text), json_text
    except json.JSONDecodeError as e:
        logger.error(f"解析JSON失败: {e}, 原始文本: {json_text[:100]}")
        return None, json_text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_norm = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        r_norm = re.sub(r'[\s_]+', '', r["name"]).upper()
        if r_norm == k_norm: return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    try:
        if isinstance(raw_val, bool): return match_score if raw_val else mismatch_score
        if isinstance(raw_val, str):
            s = raw_val.strip().lower()
            if s in ("yes", "y", "true", "是", "√", "符合"): return match_score
            if s in ("no", "n", "false", "否", "×", "不符合"): return mismatch_score
        if isinstance(raw_val, (int, float)):
            if int(raw_val) == match_score: return match_score
            if int(raw_val) == mismatch_score: return mismatch_score
    except Exception as e:
        logger.error(f"映射得分失败: {e}")
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    try:
        for pos, scores in scores_all.items():
            normalized = sum(scores.values()) / 100
            membership[pos] = max(-1.0, min(1.0, normalized))
    except Exception as e:
        logger.error(f"计算隶属度失败: {e}")
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

def get_history_count(backup_file):
    if not os.path.exists(backup_file): return 0
    try:
        return len(pd.read_csv(backup_file, encoding='utf-8-sig'))
    except Exception as e:
        return 0

def safe_write_csv(df, file_path, mode='a', header=False, encoding='utf-8-sig', max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            with open(file_path, mode, encoding=encoding) as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                df.to_csv(f, mode=mode, header=header, index=False)
                fcntl.flock(f, fcntl.LOCK_UN)
            return True
        except Exception as e:
            retry_count += 1
            time.sleep(1)
    return False

def save_process_progress(file_name, current_row, total_rows):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"file_name": file_name, "current_row": current_row, "total_rows": total_rows, "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存进度失败: {e}")

def load_process_progress():
    if not os.path.exists(PROGRESS_FILE): return None
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception:
        return None

def clear_process_progress():
    if os.path.exists(PROGRESS_FILE):
        try: os.remove(PROGRESS_FILE)
        except Exception: pass

# ===============================
# LLM API 调用
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3):
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    if _provider not in MODEL_CONFIGS: return False, {"error": f"未知提供商 {_provider}"}, f"未知提供商 {_provider}"
    
    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}/{cfg['endpoint'].lstrip('/')}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    
    streaming_placeholder = st.empty()
    full_content = ""
    error_msg = "未知错误"
    
    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as response:
                if response.status_code != 200:
                    status_code = response.status_code
                    detail = response.json() if "application/json" in response.headers.get("Content-Type", "") else response.text
                    if status_code == 404: error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif status_code == 401: error_msg = "鉴权失败 (401)。请检查 API Key 权限。"
                    else: error_msg = f"API 错误: {status_code} - {detail}"
                    if status_code in [404, 401]: break
                    response.raise_for_status()
                    
                for line in response.iter_lines():
                    if not line: continue
                    line_text = line.decode('utf-8').strip()
                    json_str = line_text[5:].strip() if line_text.startswith("data:") else line_text
                    if json_str == "[DONE]": break
                    
                    try:
                        chunk = json.loads(json_str)
                        delta_text = ""
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            if "delta" in choice: delta_text = choice["delta"].get("content", "")
                            elif "message" in choice: delta_text = choice["message"].get("content", "")
                        elif "output" in chunk:
                            output = chunk["output"]
                            if "choices" in output: delta_text = output["choices"][0].get("message", {}).get("content", "")
                            elif "text" in output: delta_text = output["text"]
                        if delta_text: full_content += delta_text
                    except json.JSONDecodeError:
                        continue
            if full_content:
                streaming_placeholder.empty()
                return True, {"choices": [{"message": {"content": full_content}}]}, ""
            else:
                error_msg = "模型未返回有效文本内容。"
        except Exception as e:
            error_msg = f"请求异常（第{attempt+1}次尝试）: {str(e)}"
            time.sleep(2 ** attempt)
            
    streaming_placeholder.empty()
    return False, {"error": error_msg}, error_msg

def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str, str]:
    if not word: return {}, "", "未知", ""
    full_rules_by_pos = {pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules]) for pos, rules in RULE_SETS.items()}
    system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：
- 需要判断的词类：名词、动词、名动词
- 评分规则已经由系统定义，你**不要**自己设计分值，也**不要**在 JSON 中给出具体数字分数。程序将根据你的判断（true/false）自动赋值。
- 你只需要判断每一条规则是"符合"还是"不符合"。

【各词类的规则说明（仅供你判断使用）】
【名词】\n{full_rules_by_pos["名词"]}
【动词】\n{full_rules_by_pos["动词"]}
【名动词】\n{full_rules_by_pos["名动词"]}

【输出要求】
1. 在 explanation 字段中，必须**逐条规则**说明判断依据，并举例（可以自己造句）：
   - 格式示例：「名词-N1_可受数量词修饰：符合。理由：……。例句：……。」
2. 在 JSON 中的 scores 字段里：
   - 每一类下的每一条规则，只能给出 **布尔值 true / false**，表示是否符合该规则
   - 严禁在 scores 里使用数值分数（例如 0, 5, 10 等）
3. predicted_pos：选择「名词」「动词」「名动词」之一，作为最典型的词类。
4. **最后输出时，先写详细的文字推理，最后单独且完整地给出一段合法的 JSON（不要再加注释）。**
"""
    user_prompt = f"请严格按照上述要求分析词语「{word}」。\n- 在 JSON 的 scores 部分，只能用 true/false 表示\"是否符合规则\"。\n请先给出详细推理过程，然后在最后单独输出一个 JSON 对象。"

    with st.spinner(f"正在调用大模型 ({model}) 进行分析，请稍候..."):
        ok, resp_json, err_msg = call_llm_api_cached(_provider=provider, _model=model, _api_key=api_key, messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}])

    if not ok:
        st.error(f"模型调用失败: {err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", f"模型调用失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)

    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
    else:
        st.error("未能从模型响应中解析出有效的JSON。")
        explanation = "无法解析模型输出。"
        predicted_pos, raw_scores = "未知", {}

    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)
        for rule in rules:
            if rule["name"] not in scores_out[pos]: scores_out[pos][rule["name"]] = 0
            
    return scores_out, raw_text, predicted_pos, explanation

# ===============================
# UI绘图与其他功能
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm: return st.warning("无法绘制雷达图：没有有效数据。")
    categories = list(scores_norm.keys())
    if not categories: return st.warning("无法绘制雷达图：没有有效词类。")
    values = list(scores_norm.values())
    categories += [categories[0]]
    values += [values[0]]
    axis_min, axis_max = min(min(values), -0.1), max(max(values), 1.0)
    
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度", hovertemplate='<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>')])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[axis_min, axis_max])), showlegend=False, title=dict(text=title, x=0.5, font=dict(size=16)))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 主页面逻辑（UI优化版）
# ===============================
def main():
    st.markdown("""
    <div class="title-header-card">
        <h1>基于大语言模型的汉语词类隶属度检测划类平台</h1>
        <div class="subtitle">Chinese Membership Detection and Classification Platform Based on Large Language Models (LLMs)</div>
        <div class="badges">
            <span class="badge">多模型支持</span><span class="badge">隶属度分析</span><span class="badge">可视化展示</span><span class="badge">批量处理</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 修复第一和第三点：移除打断DOM的空包裹，改用特定ID供 :has() 选取整个列进行样式渲染 =====
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div id="model-settings-section" class="section-title"><span class="icon-dot"></span> 模型设置（LLM）</div>', unsafe_allow_html=True)
        if not AVAILABLE_MODEL_OPTIONS:
            st.error("找不到可用的 API Key！请设置以下任意一个环境变量来启用模型:")
            for name, info in MODEL_OPTIONS.items(): st.code(f"export {info['env_var']}='你的API Key'", language="bash")
            selected_model_display_name = list(MODEL_OPTIONS.keys())[0]
            selected_model_info = MODEL_OPTIONS[selected_model_display_name]
            st.selectbox("选择大模型 (不可用)", list(MODEL_OPTIONS.keys()), disabled=True)
        else:
            selected_model_display_name = st.selectbox("选择大模型", list(AVAILABLE_MODEL_OPTIONS.keys()), key="model_select")
            selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                <span class="status-badge success">● 已配置</span>
                <span style="color: #64748b; font-size: 0.85rem;">提供商: {selected_model_info['provider'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div id="connection-test-section" class="section-title" style="justify-content: center;"><span class="icon-dot"></span> 连接测试</div>', unsafe_allow_html=True)
        st.write("")
        if not selected_model_info["api_key"]:
            st.button("测试模型链接 (不可用)", type="secondary", disabled=True)
        else:
            if st.button("测试模型链接", type="secondary"):
                with st.spinner("正在测试连接..."):
                    ok, _, err_msg = call_llm_api_cached(_provider=selected_model_info["provider"], _model=selected_model_info["model"], _api_key=selected_model_info["api_key"], messages=[{"role": "user", "content": "请回复'pong'"}], max_tokens=10)
                if ok: st.success("成功！")
                else: st.error(f"失败: {err_msg}")

    st.markdown("---")

    # ===== 分页 =====
    tab1, tab2 = st.tabs(["单个词语详细分析", "Excel 批量处理"])

    with tab1:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon-dot"></span> 词语输入</div>', unsafe_allow_html=True)
        word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
        analyze_button = st.button("开始分析", type="primary", disabled=not (selected_model_info["api_key"] and word))
        
        with st.expander("ℹ️ 使用说明", expanded=False):
            st.markdown('<div class="info-highlight">', unsafe_allow_html=True)
            st.info("1. 配置 API Key 2. 输入词语 3. 开始分析 4. 查看结果")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_button and word and selected_model_info["api_key"]:
            status_placeholder = st.empty()
            status_placeholder.info(f"正在为词语「{word}」启动分析，使用模型：{selected_model_display_name}...")

            scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(word=word, provider=selected_model_info["provider"], model=selected_model_info["model"], api_key=selected_model_info["api_key"])
            status_placeholder.empty()
            
            if scores_all:
                membership = calculate_membership(scores_all)
                final_membership = membership.get(predicted_pos, 0)
                
                st.markdown(f"""
                <div class="result-success-card">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #065f46;">分析完成</div>
                    <div style="margin-top: 0.5rem; font-size: 1rem; color: #065f46;">
                        词语「<strong>{word}</strong>」最可能的词类是 <span style="background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600;">{predicted_pos}</span>，隶属度为 <strong>{final_membership:.4f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_results_1, col_results_2 = st.columns(2)
                with col_results_1:
                    st.markdown('<div class="module-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度排名</div>', unsafe_allow_html=True)
                    top10 = get_top_10_positions(membership)
                    for i, (pos, score) in enumerate(top10):
                        rank_class = f"top-{i+1}" if i < 3 else ""
                        st.markdown(f"""
                        <div class="rank-card {rank_class}">
                            <div style="display: flex; align-items: center; gap: 0.75rem;"><div class="rank-num">{i+1}</div><span style="font-weight: 600; color: #1e3a5f;">{pos}</span></div>
                            <span style="font-weight: 700; color: #2d5a87; font-size: 1.1rem;">{score:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="module-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度雷达图</div>', unsafe_allow_html=True)
                    plot_radar_chart_streamlit(dict(top10), f"「{word}」的词类隶属度分布")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_results_2:
                    st.markdown('<div class="module-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 各词类详细得分</div>', unsafe_allow_html=True)
                    pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in scores_all.keys()}
                    sorted_pos_names = sorted(pos_total_scores.keys(), key=lambda pos: pos_total_scores[pos], reverse=True)
                    for pos in sorted_pos_names:
                        total_score = pos_total_scores[pos]
                        max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                        with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                            rule_data = []
                            for rule_name, rule_score in scores_all[pos].items():
                                rule_desc = next((r["desc"] for r in RULE_SETS.get(pos, []) if r["name"] == rule_name), "")
                                rule_data.append({"规则代码": rule_name, "规则描述": rule_desc, "得分": rule_score})
                            rule_df = pd.DataFrame(sorted(rule_data, key=lambda x: x["得分"], reverse=True))
                            styled_df = rule_df.style.map(lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, (int, float)) and x < 0 else "", subset=["得分"])
                            st.dataframe(styled_df, use_container_width=True, height=min(len(rule_df) * 30 + 50, 400))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="module-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型原始响应</div>', unsafe_allow_html=True)
                    with st.expander("点击展开查看原始响应", expanded=False): st.code(raw_text, language="text")
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon-dot"></span> 批量任务实时监控</div>', unsafe_allow_html=True)
        
        st.markdown("#### 控制面板")
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            metric_placeholder = st.empty()
            history_count = get_history_count(BACKUP_FILE)
            metric_placeholder.metric("已存数据量", f"{history_count} 条")
            if os.path.exists(BACKUP_FILE): st.caption(f"存储位置: `{BACKUP_FILE}`")
        
        with ctrl_col2:
            if os.path.exists(BACKUP_FILE):
                with open(BACKUP_FILE, "rb") as f:
                    st.download_button("下载历史文件(CSV)", data=f, file_name=f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
            else:
                st.button("下载历史文件", disabled=True, use_container_width=True)
        with ctrl_col3:
            if st.button("清空本地记录", use_container_width=True, type="secondary"):
                if os.path.exists(BACKUP_FILE):
                    try:
                        os.remove(BACKUP_FILE)
                        clear_process_progress()
                        st.success("已清空本地记录和进度")
                        metric_placeholder.metric("已存数据量", "0 条")
                        st.rerun()
                    except Exception as e: st.error(f"清空记录失败: {e}")
                else: st.info("暂无本地记录可清空")
        
        st.divider()
        st.markdown("#### 运行状态")
        progress_bar = st.progress(0)
        status_info = st.empty()
        
        st.markdown("#### 实时结果预览")
        table_placeholder = st.empty()
        if os.path.exists(BACKUP_FILE):
            try: table_placeholder.dataframe(pd.read_csv(BACKUP_FILE, encoding='utf-8-sig'), use_container_width=True, height=300)
            except Exception as e: table_placeholder.error(f"显示历史记录失败: {e}")
        else:
            table_placeholder.info("暂无数据。上传文件并点击开始后，结果将在此逐行实时显示。")
        
        st.divider()
        st.markdown("#### 上传新任务")
        uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df_input = pd.read_excel(uploaded_file)
                target_col = next((col for col in df_input.columns if "词" in str(col) or "word" in str(col).lower()), None)
                
                if target_col:
                    st.markdown(f"""
                    <div class="info-highlight">
                        <div style="font-weight: 600; color: #1e40af;">文件信息</div>
                        <div style="margin-top: 0.5rem;">识别到目标列: <code>{target_col}</code> | 待分析总数: <strong>{len(df_input)}</strong> 条</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("开始处理", type="primary", use_container_width=True):
                        if not selected_model_info["api_key"]: st.error("请先在上方配置有效的 API Key")
                        else:
                            existing_words = set()
                            if os.path.exists(BACKUP_FILE):
                                try:
                                    existing_df = pd.read_csv(BACKUP_FILE, encoding='utf-8-sig')
                                    if "词语" in existing_df.columns: existing_words = set(existing_df["词语"].astype(str).tolist())
                                    st.info(f"已跳过 {len(existing_words)} 条已处理记录")
                                except Exception as e: st.warning(f"读取记录失败: {e}")
                            
                            total_rows = len(df_input)
                            try:
                                for index, row in df_input.iterrows():
                                    word = str(row[target_col]).strip()
                                    if not word:
                                        status_info.write(f"**跳过空值**: 第 {index+1}/{total_rows} 行")
                                        progress_bar.progress((index + 1) / total_rows)
                                        continue
                                    
                                    pct = int((index + 1) / total_rows * 100)
                                    progress_bar.progress((index + 1) / total_rows)
                                    
                                    if word in existing_words:
                                        status_info.write(f" **跳过已处理**: {word} ({index+1}/{total_rows}) | 进度: {pct}%")
                                        continue
                                    
                                    status_info.write(f" **正在分析**: `{word}` | 进度: {index+1}/{total_rows} ({pct}%)")
                                    
                                    max_retries = 3
                                    success = False
                                    scores, raw_text, pred_pos, explanation = {}, "", "处理失败", "无响应"
                                    for attempt in range(max_retries):
                                        try:
                                            scores, raw_text, pred_pos, explanation = ask_model_for_pos_and_scores(word=word, provider=selected_model_info["provider"], model=selected_model_info["model"], api_key=selected_model_info["api_key"])
                                            success = bool(scores)
                                            if success: break
                                            time.sleep(2)
                                        except Exception as e:
                                            explanation = f"调用异常: {str(e)}"
                                            time.sleep(2)
                                    
                                    membership = calculate_membership(scores) if success else {}
                                    new_row = {
                                        "序数": index + 1, "词语": word, "动词": membership.get("动词", 0.0),
                                        "名词": membership.get("名词", 0.0), "名动词": membership.get("名动词", 0.0),
                                        "差值/距离": round(abs(membership.get("动词", 0.0) - membership.get("名词", 0.0)), 4),
                                        "预测词类": pred_pos, "原始响应": raw_text if success else f"错误: {explanation}",
                                        "时间戳": time.strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    try:
                                        temp_df = pd.DataFrame([new_row])
                                        write_success = safe_write_csv(temp_df, BACKUP_FILE, mode='a', header=not os.path.exists(BACKUP_FILE))
                                        if write_success:
                                            existing_words.add(word)
                                            metric_placeholder.metric("已存数据量", f"{get_history_count(BACKUP_FILE)} 条")
                                    except Exception as csv_err:
                                        st.error(f"保存第 {index+1} 条记录失败: {csv_err}")
                                    
                                    try:
                                        table_placeholder.dataframe(pd.read_csv(BACKUP_FILE, encoding='utf-8-sig'), use_container_width=True, height=300)
                                    except Exception as read_err: st.warning(f"刷新表格失败: {read_err}")
                                    time.sleep(0.5)
                                
                                progress_bar.progress(100)
                                status_info.success(f"批量处理完成！总处理量: {total_rows} 条，已保存到 {BACKUP_FILE}")
                                clear_process_progress()
                                st.rerun()
                            except Exception as batch_err:
                                status_info.error(f" 批量处理中断: {batch_err}，下次可从断点继续")
                else:
                    st.error("未识别到包含'词'或'word'的列，请检查Excel文件结构")
            except Exception as e:
                st.error(f"读取Excel文件失败: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

st.markdown("---")
st.markdown('<div class="footer-text">© 2025 汉语词类隶属度检测划类平台 | Ryan</div>', unsafe_allow_html=True)

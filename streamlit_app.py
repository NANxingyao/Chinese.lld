import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import time
import logging
import subprocess
import sys
from typing import Tuple, Dict, Any, List
from pathlib import Path

SERVICE_MODE = "--worker" in sys.argv or "--supervisor" in sys.argv

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
        page_title="基于大语言模型的汉语隶属度检测划类平台",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None
    )

# ===============================
# 自定义CSS样式
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

/* ===== 主按钮样式优化 ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 50%, #3d8bd6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: white !important;
    box-shadow: 0 6px 20px rgba(30, 77, 123, 0.4), 0 2px 6px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
    width: 100%;
    letter-spacing: 0.5px;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f2942 0%, #1e4d7b 50%, #2d6cb8 100%) !important;
    box-shadow: 0 10px 30px rgba(30, 77, 123, 0.5), 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    transform: translateY(-2px);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0);
    box-shadow: 0 3px 10px rgba(30, 77, 123, 0.3) !important;
}

/* ===== 次要按钮样式 ===== */
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.stButton > button[kind="secondary"]:hover {
    border-color: #2d6cb8 !important;
    color: #1e4d7b !important;
    background: #f5f9ff !important;
    box-shadow: 0 4px 12px rgba(45, 108, 184, 0.15);
    transform: translateY(-1px);
}

/* ===== 输入框样式 ===== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
    background: #fefefe !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div > div:focus {
    border-color: #2d6cb8 !important;
    box-shadow: 0 0 0 4px rgba(45, 108, 184, 0.1) !important;
    background: #ffffff !important;
}

/* ===== 标签页样式优化 ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #f1f5f9;
    padding: 0.6rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.04);
}
.stTabs [data-baseweb="tab"] {
    height: 2.8rem;
    border-radius: 12px !important;
    padding: 0 1.75rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    transition: all 0.3s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1e4d7b !important;
    background: rgba(45, 108, 184, 0.08);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(30, 77, 123, 0.3);
}

/* ===== 其他通用样式 ===== */
.result-success-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #10b981;
    margin: 1rem 0;
}
.info-highlight {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #3b82f6;
    margin: 0.75rem 0;
}
.error-highlight {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #ef4444;
    margin: 0.75rem 0;
}
.rank-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    background: linear-gradient(135deg, #fefefe 0%, #f5f7fa 100%);
    border: 1.5px solid #e2e8f0;
    transition: all 0.3s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.rank-card:hover {
    background: linear-gradient(135deg, #f5f9ff 0%, #e8f0fe 100%);
    border-color: #2d6cb8;
    transform: translateX(6px);
    box-shadow: 0 4px 12px rgba(45, 108, 184, 0.12);
}
.rank-card .rank-num {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.rank-card.top-1 .rank-num {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    color: white;
}
.rank-card.top-2 .rank-num {
    background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
    color: white;
}
.rank-card.top-3 .rank-num {
    background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    color: white;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.status-badge.success {
    background: #d1fae5;
    color: #065f46;
}

/* ===== 后台任务运行转动圈 ===== */
.running-status{display:flex;align-items:center;gap:.75rem;padding:.9rem 1.1rem;margin:.5rem 0 1rem 0;border-radius:12px;background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border:1px solid #bfdbfe;color:#1e40af;font-weight:600;}
.running-spinner{width:18px;height:18px;border:3px solid rgba(45,108,184,.2);border-top-color:#2d6cb8;border-radius:50%;animation:batch-spin .9s linear infinite;flex:0 0 auto;}
@keyframes batch-spin{to{transform:rotate(360deg);}}
.batch-detail{margin-top:.35rem;font-size:.9rem;font-weight:500;color:#475569;}
</style>
"""
if not SERVICE_MODE:
    st.markdown(custom_css, unsafe_allow_html=True)

# ===============================
# 全局常量与动态文件路径生成
# ===============================
BASE_DIR = Path(__file__).parent

def get_project_files(project_code: str) -> Tuple[Path, Path]:
    """根据实验批次码动态生成隔离的数据文件路径"""
    safe_code = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', project_code)
    if not safe_code:
        safe_code = "default_task"
    backup_file = BASE_DIR / f"batch_history_{safe_code}.csv"
    progress_file = BASE_DIR / f"process_progress_{safe_code}.json"
    return backup_file, progress_file

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
    "OpenAI GPT-4o（推荐）": {
        "provider": "openai", "model": "gpt-4o-mini", 
        "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"
    },
    "Google Gemini 1.5 Pro": {
        "provider": "gemini", "model": "gemini-1.5-pro", 
        "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"
    },
    "Google Gemini 1.5 Flash": {
        "provider": "gemini", "model": "gemini-1.5-flash", 
        "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot", "model": "kimi-k2.6", 
        "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"
    },
    "Qwen（通义千问）": {
        "provider": "qwen", "model": "qwen-max", 
        "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"
    },
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS: AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 增强型工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict): return ""
    try:
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        logger.error(f"提取响应文本失败: {e}")
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    if not text: return None, text
    
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        try: return json.loads(code_block_match.group(1)), code_block_match.group(1)
        except json.JSONDecodeError: pass

    last_bracket = text.rfind('}')
    if last_bracket != -1:
        first_bracket = text.rfind('{', 0, last_bracket)
        while first_bracket != -1:
            candidate = text[first_bracket:last_bracket+1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and ("scores" in parsed or "predicted_pos" in parsed):
                    return parsed, candidate
            except json.JSONDecodeError:
                pass
            first_bracket = text.rfind('{', 0, first_bracket)

    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    if match:
        try: return json.loads(match.group(1)), match.group(1)
        except json.JSONDecodeError: pass

    return None, text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_clean = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        r_clean = re.sub(r'[\s_]+', '', r["name"]).upper()
        if r_clean == k_clean: return r["name"]
    for r in pos_rules:
        r_clean = re.sub(r'[\s_]+', '', r["name"]).upper()
        code_match = re.match(r'^(NV\d+|N\d+|V\d+)', r_clean)
        if code_match:
            code = code_match.group(1)
            if k_clean == code or k_clean.startswith(code) or code in k_clean:
                return r["name"]
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
            raw_val_int = int(raw_val)
            if raw_val_int == match_score: return match_score
            if raw_val_int == mismatch_score: return mismatch_score
    except Exception as e:
        logger.error(f"映射得分失败: {e}")
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    try:
        for pos, scores in scores_all.items():
            total_score = sum(scores.values())
            normalized = total_score / 100
            membership[pos] = max(-1.0, min(1.0, normalized))
    except Exception as e:
        logger.error(f"计算隶属度失败: {e}")
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    try:
        return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception as e:
        logger.error(f"排序隶属度失败: {e}")
        return []

def get_history_count(backup_file):
    if not os.path.exists(backup_file): return 0
    try:
        temp_history = pd.read_csv(backup_file, encoding='utf-8-sig')
        return len(temp_history)
    except Exception as e:
        logger.warning(f"读取历史记录数量失败: {e}")
        return 0

def safe_write_csv(df, file_path, mode='a', header=False, encoding='utf-8-sig', max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            with open(file_path, mode, encoding=encoding) as f:
                df.to_csv(f, mode=mode, header=header, index=False)
            return True
        except Exception as e:
            retry_count += 1
            logger.warning(f"写入CSV失败（重试{retry_count}/{max_retries}）: {e}")
            time.sleep(1)
    logger.error(f"写入CSV最终失败: {file_path}")
    return False

# ===============================
# 进度管理
# ===============================
def save_process_progress(file_name, current_row, total_rows, progress_file):
    try:
        progress_data = {
            "file_name": file_name,
            "current_row": current_row,
            "total_rows": total_rows,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存进度失败: {e}")

def load_process_progress(progress_file):
    if not os.path.exists(progress_file): return None
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载进度失败: {e}")
        return None

def clear_process_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
        except Exception as e:
            logger.error(f"清除进度文件失败: {e}")

# ===============================
# 进程辅助
# ===============================
def spawn_detached(cmd, cwd):
    """跨平台脱离式进程启动，防止 Streamlit 重启误杀子进程"""
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True
    return subprocess.Popen(cmd, cwd=cwd, **kwargs)

# ===============================
# LLM调用与词类判定主函数
# ===============================
def get_provider_config(provider, api_key, model, messages, max_tokens, temperature):
    """生成标准化的大模型统一请求配置（全部对齐 OpenAI SSE 格式）"""
    base_urls = {
        "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        # 使用兼容层保障 SSE 解析统一不异常
        "gemini": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"), 
        "moonshot": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        "qwen": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    }
    
    url = f"{base_urls.get(provider, base_urls['openai']).rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": provider != "moonshot"  # 保留 kimi 的非流式限制
    }
    
    if provider == "moonshot":
        payload["max_tokens"] = 8192
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = 0.6
    else:
        payload["max_tokens"] = max_tokens
        
    return url, headers, payload

def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3, show_ui=True):
    if not _api_key:
        return False, {"error": "API Key 为空"}, "API Key 未提供"
        
    url, headers, payload = get_provider_config(_provider, _api_key, _model, messages, max_tokens, temperature)
    streaming_placeholder = st.empty() if show_ui else None
    error_msg = "未知错误"
    is_kimi = _provider == "moonshot"

    for attempt in range(max_retries):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=not is_kimi, timeout=120
            ) as response:
                if response.status_code != 200:
                    status_code = response.status_code
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text

                    if status_code == 404:
                        error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif status_code == 401:
                        error_msg = "鉴权失败 (401)。请检查 API Key 权限。"
                    elif status_code == 403:
                        error_msg = f"没有 API 权限 (403)：{detail}"
                    elif status_code == 400:
                        error_msg = f"请求参数错误 (400)：{detail}"
                    else:
                        error_msg = f"API 错误: {status_code} - {detail}"
                    
                    if status_code in [400, 401, 403, 404]:
                        break
                    response.raise_for_status()

                # 非流式 (Kimi)
                if is_kimi:
                    try: body = response.json()
                    except Exception: body = None
                    if isinstance(body, dict):
                        text = extract_text_from_response(body)
                        if text:
                            if streaming_placeholder is not None: streaming_placeholder.empty()
                            return True, body, ""
                    error_msg = f"接口返回 200，但没有解析到文本内容。\n原始响应：{response.text[:2000]}"
                    break

                # 标准化 OpenAI 格式 SSE 解析流式 (支持 DeepSeek, OpenAI, Qwen, Gemini)
                full_content = ""
                for line in response.iter_lines():
                    if not line: continue
                    line_text = line.decode('utf-8').strip()
                    if not line_text.startswith("data:"): continue
                    
                    json_str = line_text[5:].strip()
                    if json_str == "[DONE]": break

                    try:
                        chunk = json.loads(json_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            delta_text = choice.get("delta", {}).get("content", "") or ""
                            if delta_text:
                                full_content += delta_text
                    except json.JSONDecodeError:
                        continue

                if full_content:
                    if streaming_placeholder is not None: streaming_placeholder.empty()
                    return True, {"choices": [{"message": {"content": full_content}}]}, ""
                else:
                    error_msg = "模型未返回有效文本内容或由于格式不兼容导致解析失败。"

        except Exception as e:
            error_msg = f"请求异常（第{attempt+1}次尝试）: {str(e)}"
            time.sleep(2 ** attempt)

    if streaming_placeholder is not None:
        streaming_placeholder.empty()
    return False, {"error": error_msg}, error_msg

def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str, show_ui=True) -> Tuple[Dict[str, Dict[str, int]], str, str, str, bool]:
    if not word: return {}, "", "未知", "", False
    
    full_rules_by_pos = {
        pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules])
        for pos, rules in RULE_SETS.items()
    }
    
    system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：
- 需要判断的词类：名词、动词、名动词
- 评分规则已经由系统定义，你**不要**自己设计分值，也**不要**在 JSON 中给出具体数字分数。程序将根据你的判断（true/false）自动赋值。
- 你只需要判断每一条规则是"符合"还是"不符合"。

【各词类的规则说明（仅供你判断使用）】
【名词】\n{full_rules_by_pos["名词"]}
【动词】\n{full_rules_by_pos["动词"]}
【名动词】\n{full_rules_by_pos["名动词"]}

【输出要求】
1. 在 explanation 字段中，必须**逐条规则**说明判断依据，并举例。explanation 里要覆盖 **三个词类的所有规则**。
2. 在 JSON 中的 scores 字段里，每一类下的每一条规则，只能给出 **布尔值 true / false**，表示是否符合该规则。
3. predicted_pos：请选择「名词」「动词」「名动词」之一，作为该词语最典型的词类。
4. 判断兼类：请结合词类典型性特征，判断该词在现代汉语中是否属于“兼类词”（即具备多种词类的句法功能，如兼具动词和名词特征）。在 JSON 中新增字段 `is_dual_category`，用 true 或 false 表示。
5. 最后单独且完整地给出一段合法的 JSON。

JSON 结构示例：
{{"explanation": "...", "predicted_pos": "...", "is_dual_category": true, "scores": {{"名词": {{...}}, "动词": {{...}}, "名动词": {{...}}}}}}
"""

    user_prompt = f"""请严格按照上述要求分析词语「{word}」。
特别注意：scores 部分只能用 true/false；explanation 必须包含理由和例句。
请先给出详细推理过程，然后在最后单独输出一个 JSON 对象。"""

    if show_ui:
        ui_ctx = st.spinner(f"正在调用大模型 ({model}) 进行分析，请稍候...")
    else:
        ui_ctx = __import__("contextlib").nullcontext()

    with ui_ctx:
        ok, resp_json, err_msg = call_llm_api_cached(
            _provider=provider, _model=model, _api_key=api_key,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}],
            show_ui=show_ui
        )
        
    if not ok:
        if show_ui:
            st.error(f"模型调用失败: {err_msg}")
        logger.error(f"模型调用失败 - 词语:{word}, 错误:{err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", f"模型调用失败: {err_msg}", False

    raw_text = extract_text_from_response(resp_json)
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)
    is_dual_category = False
    
    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        is_dual_category = parsed_json.get("is_dual_category", False)
        raw_scores = parsed_json.get("scores", {})
        if predicted_pos not in RULE_SETS:
             if show_ui:
                 st.warning(f"模型预测的词类 '{predicted_pos}' 不在分析范围内 ('名词', '动词', '名动词')。")
    else:
        if show_ui:
            st.error(" 未能从模型响应中解析出有效的JSON。请检查模型输出是否符合要求。")
        explanation = "无法解析模型输出。原始响应：\n" + raw_text
        predicted_pos = "未知"
        raw_scores = {}

    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    try:
        for pos, rules in RULE_SETS.items():
            raw_pos_scores = raw_scores.get(pos, {})
            if isinstance(raw_pos_scores, dict):
                for k, v in raw_pos_scores.items():
                    normalized_key = normalize_key(k, rules)
                    if normalized_key:
                        rule_def = next(r for r in rules if r["name"] == normalized_key)
                        scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)
        for pos, rules in RULE_SETS.items():
            for rule in rules:
                rule_name = rule["name"]
                if rule_name not in scores_out[pos]:
                    scores_out[pos][rule_name] = 0
    except Exception as e:
        logger.error(f"处理得分失败: {e}")
        scores_out = {}
        
    return scores_out, raw_text, predicted_pos, explanation, is_dual_category

def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm:
        st.warning("无法绘制雷达图：没有有效数据。")
        return
    categories = list(scores_norm.keys())
    values = list(scores_norm.values())
    categories += [categories[0]]
    values += [values[0]]
    
    min_val, max_val = min(values), max(values)
    axis_min, axis_max = min(min_val, -0.1), max(max_val, 1.0)
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values, theta=categories, fill="toself", name="隶属度",
            hovertemplate = '<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>'
        )
    ])
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[axis_min, axis_max],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0] if axis_min >= 0 else [-1.0, -0.5, 0, 0.5, 1.0]
            )
        ),
        showlegend=False, title=dict(text=title, x=0.5, font=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 独立进程批处理：Supervisor + Worker
# ===============================
def _write_job_state(state_file: Path, **updates):
    try:
        current = {}
        if state_file.exists():
            try: current = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception: current = {}
        current.update(updates)
        current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, state_file)
    except Exception as e:
        logger.exception(f"写入任务状态失败: {e}")

def _load_job_state(state_file: Path):
    try:
        if state_file.exists():
            return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception(f"读取任务状态失败: {e}")
    return None

def _pid_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0: return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _service_file(job_state_file: Path, suffix: str) -> Path:
    return job_state_file.with_name(job_state_file.stem + suffix)

def _batch_worker(df_input, target_col, file_name, backup_file, progress_file,
                  provider, model, api_key, job_state_file):
    total_rows = len(df_input)
    existing_words = set()
    try:
        if backup_file.exists():
            existing_df = pd.read_csv(backup_file, encoding="utf-8-sig")
            if "词语" in existing_df.columns:
                existing_words = set(existing_df["词语"].astype(str).tolist())
    except Exception as e:
        logger.warning(f"读取历史记录失败，将继续处理：{e}")

    state = _load_job_state(job_state_file) or {}
    start_row = int(state.get("next_row", 0))
    if start_row < 0 or start_row > total_rows: start_row = 0

    _write_job_state(
        job_state_file,
        status="running", file_name=file_name, total_rows=total_rows,
        next_row=start_row, current_row=start_row, current_word="",
        retry_count=0, error="", completed_rows=start_row
    )

    while start_row < total_rows:
        index = start_row
        word = str(df_input.iloc[index][target_col]).strip()

        if not word:
            start_row = index + 1
            save_process_progress(file_name, start_row, total_rows, progress_file)
            _write_job_state(job_state_file, status="running", next_row=start_row,
                             current_row=index, current_word="", retry_count=0,
                             error="", completed_rows=start_row)
            continue

        if word in existing_words:
            start_row = index + 1
            save_process_progress(file_name, start_row, total_rows, progress_file)
            _write_job_state(job_state_file, status="running", next_row=start_row,
                             current_row=index, current_word=word, retry_count=0,
                             error="已存在", completed_rows=start_row)
            continue

        success = False
        last_error = ""  # [FIX] 修复初始化为未知错误导致的闪现问题
        retry_count = 0
        scores, raw_text, pred_pos, explanation, is_dual_category = {}, "", "处理失败", "无响应", False

        while not success:
            retry_count += 1
            _write_job_state(
                job_state_file,
                status="retrying" if retry_count > 1 else "running",
                current_row=index, current_word=word, next_row=index,
                retry_count=retry_count, error=last_error, completed_rows=index
            )
            try:
                scores, raw_text, pred_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(
                    word=word, provider=provider, model=model, api_key=api_key, show_ui=False
                )
                if scores:
                    success = True
                    break
                last_error = explanation or "模型未返回有效结果"
            except BaseException as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.exception(f"Worker处理词语 {word} 异常")

            wait_seconds = min(60, max(2, 2 ** min(retry_count - 1, 5)))
            _write_job_state(
                job_state_file, status="waiting_retry", current_row=index,
                current_word=word, next_row=index, retry_count=retry_count,
                error=last_error, retry_in_seconds=wait_seconds, completed_rows=index
            )
            time.sleep(wait_seconds)

        membership = calculate_membership(scores)
        new_row = {
            "序数": index + 1, "词语": word,
            "动词": membership.get("动词", 0.0),
            "名词": membership.get("名词", 0.0),
            "名动词": membership.get("名动词", 0.0),
            "差值/距离": round(abs(membership.get("动词", 0.0) - membership.get("名词", 0.0)), 4),
            "预测词类": pred_pos,
            "是否兼类": "是" if is_dual_category else "否",
            "原始响应": raw_text,
            "时间戳": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        save_retry = 0
        while True:
            try:
                save_retry += 1
                _write_job_state(job_state_file, status="saving", current_row=index,
                                 current_word=word, next_row=index,
                                 retry_count=save_retry, error="", completed_rows=index)
                temp_df = pd.DataFrame([new_row])
                header_needed = not backup_file.exists()
                if safe_write_csv(temp_df, backup_file, mode="a", header=header_needed):
                    break
                raise IOError("CSV写入失败")
            except BaseException as e:
                last_error = f"保存结果失败：{type(e).__name__}: {e}"
                logger.exception(f"Worker保存词语 {word} 结果失败")
                wait_seconds = min(60, max(2, 2 ** min(save_retry - 1, 5)))
                _write_job_state(
                    job_state_file, status="waiting_save_retry", current_row=index,
                    current_word=word, next_row=index, retry_count=save_retry,
                    error=last_error, retry_in_seconds=wait_seconds, completed_rows=index
                )
                time.sleep(wait_seconds)

        existing_words.add(word)
        start_row = index + 1
        save_process_progress(file_name, start_row, total_rows, progress_file)
        _write_job_state(
            job_state_file, status="running", current_row=index,
            current_word=word, next_row=start_row, completed_rows=start_row,
            retry_count=0, error=""
        )

    clear_process_progress(progress_file)
    _write_job_state(
        job_state_file, status="completed", file_name=file_name,
        total_rows=total_rows, next_row=total_rows,
        completed_rows=total_rows, current_word="", retry_count=0, error=""
    )

def _worker_entry(job_state_file: Path):
    spec_file = _service_file(job_state_file, ".spec.json")
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    input_file = Path(spec["input_file"])
    df_input = pd.read_excel(input_file)
    target_col = spec["target_col"]
    provider = spec["provider"]
    model = spec["model"]
    env_var = spec["env_var"]
    api_key = os.getenv(env_var, "")
    if not api_key: raise RuntimeError(f"环境变量 {env_var} 未配置")
    _batch_worker(
        df_input=df_input, target_col=target_col,
        file_name=spec["file_name"], backup_file=Path(spec["backup_file"]),
        progress_file=Path(spec["progress_file"]),
        provider=provider, model=model, api_key=api_key,
        job_state_file=job_state_file
    )

def _supervisor_entry(job_state_file: Path):
    spec_file = _service_file(job_state_file, ".spec.json")
    supervisor_pid_file = _service_file(job_state_file, ".supervisor.pid")
    worker_pid_file = _service_file(job_state_file, ".worker.pid")
    supervisor_pid_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        restart_count = 0
        while True:
            state = _load_job_state(job_state_file) or {}
            status = state.get("status", "")
            if status == "completed": return
            if not spec_file.exists():
                _write_job_state(job_state_file, status="failed", error="任务配置文件不存在")
                return

            worker = None
            try:
                worker_cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_state_file)]
                worker = spawn_detached(worker_cmd, str(Path(__file__).resolve().parent))
                worker_pid_file.write_text(str(worker.pid), encoding="utf-8")
                restart_count += 1
                _write_job_state(
                    job_state_file, status="running", supervisor_pid=os.getpid(),
                    worker_pid=worker.pid, supervisor_restart_count=restart_count
                )
                
                while True:
                    rc = worker.poll()
                    state = _load_job_state(job_state_file) or {}
                    if state.get("status") == "completed": return
                    if rc is not None:
                        logger.warning(f"Worker PID={worker.pid} 已退出，返回码={rc}")
                        break
                    time.sleep(2)

                state = _load_job_state(job_state_file) or {}
                if state.get("status") != "completed":
                    delay = min(30, max(2, 2 ** min(restart_count - 1, 4)))
                    _write_job_state(
                        job_state_file, status="supervisor_restarting",
                        supervisor_pid=os.getpid(), worker_pid=None,
                        supervisor_restart_count=restart_count,
                        error=f"Worker意外退出（返回码 {rc}），{delay} 秒后自动启动"
                    )
                    time.sleep(delay)
            except BaseException as e:
                logger.exception("Supervisor循环异常")
                _write_job_state(
                    job_state_file, status="supervisor_restarting",
                    supervisor_pid=os.getpid(), worker_pid=None,
                    error=f"Supervisor异常：{type(e).__name__}: {e}"
                )
                time.sleep(5)
    finally:
        for f in (supervisor_pid_file, worker_pid_file):
            try:
                if f.exists(): f.unlink()
            except Exception: pass

def _create_job_spec(job_state_file, uploaded_file, target_col, file_name,
                     backup_file, progress_file, provider, model, env_var):
    project_safe = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', job_state_file.stem.replace("batch_job_", ""))
    input_file = BASE_DIR / f"batch_input_{project_safe}.xlsx"
    input_file.write_bytes(uploaded_file.getvalue())
    spec = {
        "input_file": str(input_file), "target_col": target_col,
        "file_name": file_name, "backup_file": str(backup_file),
        "progress_file": str(progress_file), "provider": provider,
        "model": model, "env_var": env_var
    }
    spec_file = _service_file(job_state_file, ".spec.json")
    tmp = spec_file.with_suffix(spec_file.suffix + ".tmp")
    tmp.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, spec_file)
    return spec

def start_or_resume_batch_job(df_input, target_col, uploaded_file, file_name,
                              backup_file, progress_file, provider, model,
                              env_var, job_state_file):
    state = _load_job_state(job_state_file) or {}
    supervisor_pid = state.get("supervisor_pid")
    if _pid_alive(supervisor_pid): return False, state

    supervisor_pid_file = _service_file(job_state_file, ".supervisor.pid")
    if supervisor_pid_file.exists():
        try:
            old_pid = int(supervisor_pid_file.read_text(encoding="utf-8").strip())
            if _pid_alive(old_pid): return False, state
        except Exception: pass

    _create_job_spec(job_state_file, uploaded_file, target_col, file_name,
                     backup_file, progress_file, provider, model, env_var)

    same_file = state.get("file_name") == file_name
    resume_row = int(state.get("next_row", 0) or 0) if same_file else 0
    resume_completed = int(state.get("completed_rows", state.get("next_row", 0) or 0)) if same_file else 0
    _write_job_state(
        job_state_file, status="starting", file_name=file_name,
        total_rows=len(df_input), next_row=resume_row,
        completed_rows=resume_completed, current_word="", retry_count=0, error=""
    )

    cmd = [sys.executable, str(Path(__file__).resolve()), "--supervisor", str(job_state_file)]
    spawn_detached(cmd, str(Path(__file__).resolve().parent))
    return True, _load_job_state(job_state_file) or {}

def _auto_recover_if_needed(job_state_file: Path):
    state = _load_job_state(job_state_file) or {}
    if state.get("status") in {"running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "starting", "supervisor_restarting"}:
        supervisor_file = _service_file(job_state_file, ".supervisor.pid")
        file_pid_alive = False
        if supervisor_file.exists():
            try: file_pid_alive = _pid_alive(int(supervisor_file.read_text(encoding="utf-8").strip()))
            except Exception: file_pid_alive = False
        
        if not _pid_alive(state.get("supervisor_pid")) and not file_pid_alive:
            spec_file = _service_file(job_state_file, ".spec.json")
            if spec_file.exists():
                cmd = [sys.executable, str(Path(__file__).resolve()), "--supervisor", str(job_state_file)]
                spawn_detached(cmd, str(Path(__file__).resolve().parent))
                _write_job_state(job_state_file, status="starting", error="检测到后台守护退出，系统已全自动原地接续。")
                state = _load_job_state(job_state_file) or state
    return state

def main():

    st.markdown("""
    <div class="title-header-card">
        <h1>基于大语言模型的汉语词类隶属度检测划类平台</h1>
        <div class="subtitle">Chinese Membership Detection and Classification Platform Based on Large Language Models (LLMs)</div>
        <div class="badges">
            <span class="badge">多模型支持</span>
            <span class="badge">隶属度分析</span>
            <span class="badge">可视化展示</span>
            <span class="badge">批量处理</span>
            <span class="badge" style="background: rgba(251, 191, 36, 0.4); border-color: rgba(251, 191, 36, 0.8);">兼类判定支持</span>
            <span class="badge" style="background: rgba(16, 185, 129, 0.4); border-color: rgba(16, 185, 129, 0.8);">抗断流自动恢复</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== 顶部控制区 =====
    control_container = st.container()
    with control_container:
        col1, col2, col3 = st.columns([5, 3, 2])
        
        with col1:
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型设置（LLM）</div>', unsafe_allow_html=True)
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("找不到可用的 API Key！请设置环境变量。")
                selected_model_display_name = list(MODEL_OPTIONS.keys())[0]
                selected_model_info = MODEL_OPTIONS[selected_model_display_name]
                st.selectbox("选择大模型 (不可用)", list(MODEL_OPTIONS.keys()), disabled=True)
            else:
                selected_model_display_name = st.selectbox(
                    "选择大模型", list(AVAILABLE_MODEL_OPTIONS.keys()), key="model_select"
                )
                selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                    <span class="status-badge success">● 已配置</span>
                    <span style="color: #64748b; font-size: 0.85rem;">提供商: {selected_model_info['provider'].upper()}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 实验配置</div>', unsafe_allow_html=True)
            if "project_code" not in st.session_state:
                st.session_state.project_code = "default_task"
            
            project_code_input = st.text_input(
                "实验批次码 (Project Code)", 
                value=st.session_state.project_code, 
                help="用于隔离不同词表集的量化分析任务。在不同设备输入相同的批次码即可接续处理进度。"
            )
            st.session_state.project_code = project_code_input
            
            BACKUP_FILE, PROGRESS_FILE = get_project_files(st.session_state.project_code)
                
        with col3:
            st.markdown('<div class="section-title" style="justify-content: center;"><span class="icon-dot"></span> 连接测试</div>', unsafe_allow_html=True)
            st.write("")
            if not selected_model_info["api_key"]:
                st.button("测试模型链接 (不可用)", type="secondary", disabled=True, use_container_width=True)
            else:
                if st.button("测试模型链接", type="secondary", use_container_width=True):
                    with st.spinner("正在测试连接..."):
                        ok, _, err_msg = call_llm_api_cached(
                            _provider=selected_model_info["provider"],
                            _model=selected_model_info["model"],
                            _api_key=selected_model_info["api_key"],
                            messages=[{"role": "user", "content": "请回复'pong'"}],
                            max_tokens=10
                        )
                    if ok: st.success("成功！")
                    else: st.error(f"失败: {err_msg}")

    st.markdown("---")

    # ===== 分页 =====
    tab1, tab2 = st.tabs(["单个词语详细分析", "Excel 批量处理"])

    # ===== 单个词语分析 =====
    with tab1:
        st.markdown('<div class="section-title"><span class="icon-dot"></span> 词语输入</div>', unsafe_allow_html=True)
        word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
        analyze_button = st.button(
            "开始分析", type="primary", disabled=not (selected_model_info["api_key"] and word)
        )
        
        with st.expander("ℹ️ 使用说明", expanded=False):
            st.info("在上方输入词语，点击开始分析。系统将显示隶属度、兼类属性、雷达图和详细规则得分。")

        if analyze_button and word and selected_model_info["api_key"]:
            status_placeholder = st.empty()
            status_placeholder.info(f"正在为词语「{word}」启动分析，使用模型：{selected_model_display_name}...")

            scores_all, raw_text, predicted_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(
                word=word, provider=selected_model_info["provider"],
                model=selected_model_info["model"], api_key=selected_model_info["api_key"]
            )
            
            status_placeholder.empty()
            
            if scores_all:
                membership = calculate_membership(scores_all)
                final_membership = membership.get(predicted_pos, 0)
                
                jianlei_badge_color = "#f59e0b" if is_dual_category else "#3b82f6"
                jianlei_text = "属于兼类词" if is_dual_category else "非兼类词"
                
                st.markdown(f"""
                <div class="result-success-card">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #065f46;">分析完成</div>
                    <div style="margin-top: 0.5rem; font-size: 1rem; color: #065f46; line-height: 1.8;">
                        词语「<strong>{word}</strong>」最可能的词类是 
                        <span style="background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600;">{predicted_pos}</span>
                        ，隶属度为 <strong>{final_membership:.4f}</strong>
                        <br/>
                        多类属性判定： <span style="background: {jianlei_badge_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600;">{jianlei_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                col_results_1, col_results_2 = st.columns(2)
                
                with col_results_1:
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度排名</div>', unsafe_allow_html=True)
                    top10 = get_top_10_positions(membership)
                    for i, (pos, score) in enumerate(top10):
                        rank_class = f"top-{i+1}" if i < 3 else ""
                        st.markdown(f"""
                        <div class="rank-card {rank_class}">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <div class="rank-num">{i+1}</div>
                                <span style="font-weight: 600; color: #1e3a5f;">{pos}</span>
                            </div>
                            <span style="font-weight: 700; color: #2d5a87; font-size: 1.1rem;">{score:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度雷达图</div>', unsafe_allow_html=True)
                    plot_radar_chart_streamlit(dict(top10), f"「{word}」的词类隶属度分布")

                with col_results_2:
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
                            styled_df = rule_df.style.map(
                                lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, (int, float)) and x < 0 else "",
                                subset=["得分"]
                            )
                            st.dataframe(styled_df, use_container_width=True, height=min(len(rule_df) * 30 + 50, 400))
                    
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型原始响应</div>', unsafe_allow_html=True)
                    with st.expander("点击展开查看原始响应", expanded=False):
                        st.code(raw_text, language="text")

    # ===== 批量处理 =====
    with tab2:
        st.markdown(f'<div class="section-title"><span class="icon-dot"></span> 批量任务实时监控 (当前批次: <code>{st.session_state.project_code}</code>)</div>', unsafe_allow_html=True)
        
        st.markdown("#### 控制面板")
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            metric_placeholder = st.empty()
            history_count = get_history_count(BACKUP_FILE)
            metric_placeholder.metric("已存数据量", f"{history_count} 条")
            
            has_history = os.path.exists(BACKUP_FILE)
            if has_history:
                st.caption(f"当前存储表: `{BACKUP_FILE.name}`")
        
        with ctrl_col2:
            if os.path.exists(BACKUP_FILE):
                with open(BACKUP_FILE, "rb") as f:
                    st.download_button(
                        label="下载历史文件(CSV)", data=f,
                        file_name=f"{st.session_state.project_code}_results_{time.strftime('%Y%m%d')}.csv",
                        mime="text/csv", use_container_width=True
                    )
            else:
                st.button("下载历史文件", disabled=True, use_container_width=True)
                
        with ctrl_col3:
            if st.button("清空本批次记录", use_container_width=True, type="secondary"):
                if os.path.exists(BACKUP_FILE):
                    try:
                        os.remove(BACKUP_FILE)
                        clear_process_progress(PROGRESS_FILE) 
                        st.success(f"已清空批次 {st.session_state.project_code} 的本地记录")
                        metric_placeholder.metric("已存数据量", "0 条")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空记录失败: {e}")
                else:
                    st.info("当前批次暂无记录可清空")
        
        st.divider()
        st.markdown("#### 运行状态")
        progress_bar = st.progress(0)
        status_info = st.empty()
        
        st.markdown("#### 实时结果预览")
        table_placeholder = st.empty()
        if os.path.exists(BACKUP_FILE):
            try:
                table_placeholder.dataframe(
                    pd.read_csv(BACKUP_FILE, encoding='utf-8-sig'), 
                    use_container_width=True, height=300
                )
            except Exception as e:
                table_placeholder.error(f"显示历史记录失败: {e}")
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
                        <div style="margin-top: 0.5rem;">
                            识别到目标列: <code>{target_col}</code> | 待分析总数: <strong>{len(df_input)}</strong> 条
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    job_state_file = BASE_DIR / f"batch_job_{re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', st.session_state.project_code)}.json"
                    current_state = _auto_recover_if_needed(job_state_file)
                    supervisor_alive = _pid_alive(current_state.get("supervisor_pid"))

                    state_status = current_state.get("status", "")
                    if state_status in {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}:
                        current_row = int(current_state.get("current_row", current_state.get("next_row", 0)) or 0)
                        total_now = int(current_state.get("total_rows", len(df_input)))
                        retry_now = int(current_state.get("retry_count", 0) or 0)
                        word_now = current_state.get("current_word", "")
                        label = "守护任务运行中" if supervisor_alive else "正在恢复守护任务"
                        st.info(
                            f"{label}：第 {current_row + 1}/{total_now} 行"
                            + (f"，当前词语「{word_now}」" if word_now else "")
                            + (f"，当前重试 {retry_now} 次" if retry_now else "")
                        )
                        if current_state.get("error"):
                            st.warning(f"最近状态：{current_state['error']}")
                    elif state_status == "completed":
                        st.success(f"🎉 批量任务已完成，共 {current_state.get('completed_rows', len(df_input))} 条")
                    elif state_status == "failed":
                        st.error(f"任务失败：{current_state.get('error', '未知错误')}")

                    can_start = not supervisor_alive
                    if st.button("开始处理 / 继续任务", type="primary", use_container_width=True, disabled=not can_start):
                        if not selected_model_info["api_key"]:
                            st.error("请先在上方配置有效的 API Key")
                        else:
                            file_name = f"{st.session_state.project_code}_{uploaded_file.name}"
                            try:
                                started, state = start_or_resume_batch_job(
                                    df_input=df_input, target_col=target_col,
                                    uploaded_file=uploaded_file, file_name=file_name,
                                    backup_file=BACKUP_FILE, progress_file=PROGRESS_FILE,
                                    provider=selected_model_info["provider"],
                                    model=selected_model_info["model"],
                                    env_var=selected_model_info["env_var"],
                                    job_state_file=job_state_file
                                )
                                if started:
                                    st.success("守护任务已启动。Worker 即使异常退出，Supervisor 也会自动重新启动并从断点继续。")
                                else:
                                    st.info("任务已经在后台运行，无需重复启动。")
                            except Exception as e:
                                st.error(f"启动后台任务失败：{e}")

                    latest_state = _auto_recover_if_needed(job_state_file)
                    completed = int(latest_state.get("completed_rows", latest_state.get("next_row", 0) or 0))
                    total_display = int(latest_state.get("total_rows", len(df_input)))
                    current_row_display = int(latest_state.get("current_row", latest_state.get("next_row", 0)) or 0)
                    current_word_display = str(latest_state.get("current_word", "") or "")
                    retry_display = int(latest_state.get("retry_count", 0) or 0)
                    state_display = str(latest_state.get("status", "") or "")

                    if total_display:
                        progress_bar.progress(min(1.0, completed / total_display))

                    active_states = {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}
                    if state_display in active_states:
                        spinner_text = {
                            "starting": "正在启动守护任务…",
                            "running": "正在处理…",
                            "retrying": "当前词语请求失败，正在重试…",
                            "waiting_retry": "正在等待下一次重试…",
                            "saving": "正在保存结果…",
                            "waiting_save_retry": "保存失败，正在重试…",
                            "supervisor_restarting": "Worker 已退出，Supervisor 正在自动恢复…"
                        }.get(state_display, "任务运行中…")
                        detail_parts = [f"第 {current_row_display + 1}/{total_display} 行"]
                        if current_word_display: detail_parts.append(f"当前词语「{current_word_display}」")
                        if retry_display: detail_parts.append(f"当前重试 {retry_display} 次")
                        detail = " · ".join(detail_parts)
                        status_info.markdown(
                            '<div class="running-status">'
                            '<span class="running-spinner"></span>'
                            '<div><div>' + spinner_text + '</div>'
                            '<div class="batch-detail">' + detail + '</div></div></div>',
                            unsafe_allow_html=True
                        )
                    elif state_display == "completed":
                        progress_bar.progress(1.0)
                        status_info.success(f"🎉 批量处理已完成，共 {total_display} 条")
                    elif state_display == "failed":
                        status_info.error(f"❌ 批量任务停止：{latest_state.get('error', '未知错误')}")

                else:
                    st.markdown('<div class="error-highlight">', unsafe_allow_html=True)
                    st.error("未识别到包含'词'或'word'的列，请检查Excel文件结构")
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"读取Excel文件失败: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_entry(Path(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--supervisor" in sys.argv:
        _supervisor_entry(Path(sys.argv[sys.argv.index("--supervisor") + 1]))
    else:
        main()

# ===============================
# 页面底部说明
# ===============================
if not SERVICE_MODE:
    st.markdown("---")
    st.markdown(
        """
        <div style="
            text-align: center;
            color: #999;
            font-size: 13px;
            line-height: 1.8;
            padding: 10px 0 5px 0;
        ">
            汉语词类隶属度检测划类平台 · © 2025 Ryan<br>
            <a href="mailto:shenrui26@gmail.com"
               style="color: #999; text-decoration: none;">
                ✉ shenrui26@gmail.com
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

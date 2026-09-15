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
import traceback
from typing import Tuple, Dict, Any, List
from pathlib import Path

SERVICE_MODE = "--worker" in sys.argv or "--supervisor" in sys.argv
# 版本戳：若界面/日志看不到此字符串，说明仍在跑旧进程，必须 kill 后重启
CODE_VERSION = "2026-09-15-stop-on-model-error-v1"

# Gemini 原生 responseSchema：把输出格式锁死为固定 JSON（短规则码 + boolean）
# normalize_key 可将 N1/V1/NV1 映射回完整规则名
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
.stApp {background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);}

/* ===== 主容器卡片化 ===== */
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95% !important;}

/* ===== 标题高亮卡片 ===== */
.title-header-card {
    background: linear-gradient(135deg, #0f2942 0%, #1e4d7b 40%, #2d6cb8 75%, #3d8bd6 100%);
    padding: 2rem 2.5rem; border-radius: 20px; margin-bottom: 1.5rem;
    box-shadow: 0 12px 40px rgba(15, 41, 66, 0.35); position: relative; overflow: hidden;
}
.title-header-card::before {
    content: ''; position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); border-radius: 50%;
}
.title-header-card h1 {
    color: #ffffff !important; font-size: 1.8rem !important; font-weight: 700 !important;
    margin: 0 !important; padding: 0 !important; text-shadow: 0 2px 4px rgba(0,0,0,0.2); position: relative; z-index: 1;
}
.title-header-card .subtitle {
    color: rgba(255, 255, 255, 0.8) !important; font-size: 0.95rem !important; margin-top: 0.5rem !important;
    font-style: italic; position: relative; z-index: 1;
}
.title-header-card .badges {margin-top: 1rem; display: flex; gap: 0.75rem; flex-wrap: wrap; position: relative; z-index: 1;}
.title-header-card .badge {
    background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px); color: #fff;
    padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.8rem; border: 1px solid rgba(255, 255, 255, 0.2);
}

/* ===== 子标题样式 ===== */
.section-title {
    display: flex; align-items: center; gap: 0.5rem; font-size: 1.15rem !important;
    font-weight: 600 !important; color: #1e3a5f !important; margin-bottom: 1rem !important;
    padding-bottom: 0.5rem; border-bottom: 2px solid #e2e8f0;
}
.section-title .icon-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #2d6cb8 0%, #3d8bd6 100%);
    box-shadow: 0 2px 6px rgba(45, 108, 184, 0.4); flex-shrink: 0;
}

/* ===== 按钮样式 ===== */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 50%, #3d8bd6 100%) !important;
    border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-weight: 700 !important; font-size: 1rem !important; color: white !important;
    box-shadow: 0 6px 20px rgba(30, 77, 123, 0.4), 0 2px 6px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important; width: 100%; letter-spacing: 0.5px;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f2942 0%, #1e4d7b 50%, #2d6cb8 100%) !important;
    box-shadow: 0 10px 30px rgba(30, 77, 123, 0.5), 0 4px 10px rgba(0, 0, 0, 0.15) !important; transform: translateY(-2px);
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important; border: 2px solid #cbd5e1 !important; border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important; font-weight: 600 !important; color: #475569 !important;
    transition: all 0.3s ease !important; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* ===== 输入框及标签页 ===== */
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
    border-radius: 12px !important; border: 2px solid #e2e8f0 !important; transition: all 0.3s ease !important; background: #fefefe !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus, .stSelectbox > div > div > div:focus {
    border-color: #2d6cb8 !important; box-shadow: 0 0 0 4px rgba(45, 108, 184, 0.1) !important; background: #ffffff !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem; background: #f1f5f9; padding: 0.6rem; border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid #e2e8f0; box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.04);
}
.stTabs [data-baseweb="tab"] {height: 2.8rem; border-radius: 12px !important; padding: 0 1.75rem !important; font-weight: 600 !important; color: #64748b !important;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #1e4d7b 0%, #2d6cb8 100%) !important; color: white !important;}

/* ===== 通用小组件 ===== */
.result-success-card {background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #10b981; margin: 1rem 0;}
.info-highlight {background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #3b82f6; margin: 0.75rem 0;}
.error-highlight {background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-radius: 10px; padding: 1rem 1.25rem; border-left: 4px solid #ef4444; margin: 0.75rem 0;}
.rank-card {display: flex; align-items: center; justify-content: space-between; padding: 0.85rem 1.25rem; border-radius: 12px; margin-bottom: 0.6rem; background: linear-gradient(135deg, #fefefe 0%, #f5f7fa 100%); border: 1.5px solid #e2e8f0;}
.rank-card .rank-num {width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%); color: #64748b; display: flex; align-items: center; justify-content: center; font-weight: 700;}
.rank-card.top-1 .rank-num {background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white;}
.status-badge {display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500;}
.status-badge.success {background: #d1fae5; color: #065f46;}

/* ===== 任务状态圈 ===== */
.running-status{display:flex;align-items:center;gap:.75rem;padding:.9rem 1.1rem;margin:.5rem 0 1rem 0;border-radius:12px;background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border:1px solid #bfdbfe;color:#1e40af;font-weight:600;}
.running-spinner{display:inline-block;width:18px;height:18px;min-width:18px;border:3px solid rgba(45,108,184,.22);border-top-color:#2d6cb8;border-radius:50%;animation:batch-spin .75s linear infinite;flex:0 0 auto;}
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

def get_project_files(project_code: str, model_key: str = "") -> Tuple[Path, Path]:
    """按实验批次 + 模型隔离结果，避免不同模型的国外/国内数据互相覆盖。"""
    safe_code = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', project_code)
    if not safe_code:
        safe_code = "default_task"
    safe_model = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', model_key or "default_model")
    if not safe_model:
        safe_model = "default_model"
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
    # ===================== 国内模型（保留原有） =====================
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

    # ===================== 国外模型：OpenAI =====================
    # ChatGPT 本身是产品名；API 实际调用的是 OpenAI 的 GPT 模型。
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

    # ===================== 国外模型：Google Gemini =====================
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

    # ===================== 国外模型：xAI Grok =====================
    "xAI Grok 4.6": {
        "provider": "xai", "model": "grok-4.6",
        "api_key": os.getenv("XAI_API_KEY"), "env_var": "XAI_API_KEY"
    },
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS: AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 安全操作辅助函数
# ===============================

def _lock_file(lock_path: Path, timeout=30, poll=0.2):
    """跨进程排他锁：基于 O_EXCL，包含超时死锁自愈机制。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            try:
                # 死锁解脱：如果锁文件存在且已严重超时（例如强杀进程遗留），强行清理
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
    """批次内每个 Excel 行的稳定唯一 ID。"""
    return f"{file_name}::row::{row_index + 1}"


def clean_csv_duplicates(file_path: Path):
    """确保全表严格去重并严格按【序数】递增排序。"""
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
    except Exception as e:
        logger.exception(f"清理并排序 CSV 失败: {file_path}")
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)


def append_unique_csv(df: pd.DataFrame, file_path: Path, task_id: str, max_retries=30):
    """
    原子写入：读取原表合并新行 -> 严格按ID去重 -> 严格按序数重新排序 -> 原子覆盖写入。
    彻底解决多进程读写导致的错序和重复。
    """
    lock_path = file_path.with_suffix(file_path.suffix + '.write.lock')
    last_error = ''
    for attempt in range(1, max_retries + 1):
        fd = None
        try:
            fd = _lock_file(lock_path, timeout=5)
            if file_path.exists() and file_path.stat().st_size > 0:
                existing_all = pd.read_csv(file_path, encoding='utf-8-sig')

                # 兼容旧版本：旧 CSV 没有任务ID时，先补全ID列
                if '任务ID' not in existing_all.columns:
                    if '序数' in existing_all.columns:
                        existing_all['任务ID'] = existing_all['序数'].apply(
                            lambda x: f'legacy::row::{int(x)}' if pd.notna(x) else f'legacy::row::{x}'
                        )
                    else:
                        existing_all['任务ID'] = [f'legacy::index::{i+1}' for i in range(len(existing_all))]

                # 如果明确已经存在最新结果，跳过
                existing_ids = set(existing_all['任务ID'].astype(str).tolist())
                if task_id in existing_ids:
                    return True, '该任务ID已经写入，跳过重复写入。', True

                combined = pd.concat([existing_all, df], ignore_index=True)
            else:
                combined = df.copy()

            # 强制剔除任何重复项并强制按序数排序
            if '序数' in combined.columns:
                combined['序数_num'] = pd.to_numeric(combined['序数'], errors='coerce')
                combined = combined.drop_duplicates(subset=['任务ID'], keep='last')
                combined = combined.sort_values('序数_num').drop(columns=['序数_num'])

            # 写入临时文件后覆盖，防止崩溃导致文件损坏
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
    """判断具体 Excel 行是否已经处理"""
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
    except Exception as e:
        pass
    return False


def spawn_detached(cmd, cwd):
    """跨平台脱离式进程启动，防止 Streamlit 页面刷新误杀子进程。"""
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
    """兼容 OpenAI / Gemini / xAI 等返回结构，尽可能提取最终文本。"""
    if not isinstance(resp_json, dict):
        return ""
    try:
        # OpenAI Responses API
        if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
            return resp_json["output_text"]

        # OpenAI-compatible Chat Completions
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] or {}
            message = choice.get("message", {}) or {}
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, dict):
                # 某些 Gemini/OpenAI 兼容层可能直接把结构化结果放入 content。
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    pass
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        # Gemini 3.x 偶发：reasoning / thought 与 text 分 part
                        text = part.get("text") or part.get("content")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    elif isinstance(part, str) and part.strip():
                        parts.append(part)
                if parts:
                    return "".join(parts)
            # 某些兼容层可能把文本放进 delta
            delta = choice.get("delta", {}) or {}
            delta_content = delta.get("content", "")
            if isinstance(delta_content, str) and delta_content.strip():
                return delta_content
            # Gemini Flash 偶发：最终 JSON 在 refusal / reasoning 之外的扩展字段
            for alt_key in ("reasoning_content", "reasoning", "output_text", "text"):
                alt = message.get(alt_key)
                if isinstance(alt, str) and alt.strip() and ("{" in alt or "scores" in alt.lower()):
                    return alt

        # Gemini 原生/代理偶见结构：candidates[].content.parts[].text
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
    """清理 Gemini 常见的 Markdown/不可见字符，但不破坏 JSON 内部文本。"""
    if not text:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    cleaned = cleaned.strip()

    # 去除首尾 Markdown JSON 代码围栏
    cleaned = re.sub(r"^\s*```(?:json|JSON|javascript|js)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _repair_json_candidate(candidate: str) -> str:
    """对已经定位到的 JSON 对象做最小侵入式修复。
    重点处理：字符串中的真实换行/制表符、尾逗号、Markdown 残留、中文弯引号。
    """
    if not candidate:
        return candidate

    candidate = candidate.strip()
    candidate = re.sub(r"^\s*```(?:json|JSON)?\s*", "", candidate)
    candidate = re.sub(r"\s*```\s*$", "", candidate).strip()

    # 常见中文/弯引号 → 标准 JSON 双引号（对 Gemini 更稳）
    candidate = (candidate
                 .replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("「", '"').replace("」", '"'))

    # 去除对象/数组结尾前的尾逗号：{"a": 1,} / [1,]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # JSON 字符串内若出现真实控制字符，转成合法 JSON 转义。
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
    """按 JSON 字符串语义寻找完整 {...} 对象，避免简单 find/rfind 误截取。"""
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
    """严格解析 -> 最小修复后解析。"""
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
    """通用 JSON 提取器。
    保留原有模型兼容性，同时解决代码块、前后说明、嵌套对象、控制字符和尾逗号问题。
    """
    if not text:
        return None, text

    cleaned = _clean_possible_markdown_json(text)

    # 1. 整段就是 JSON
    obj = _parse_json_object_candidate(cleaned)
    if isinstance(obj, dict):
        return obj, cleaned

    # 2. 在任意位置寻找完整 JSON 对象
    for candidate in _iter_balanced_json_objects(cleaned):
        obj = _parse_json_object_candidate(candidate)
        if isinstance(obj, dict):
            return obj, candidate

    return None, text


def _gemini_prefer_schema_objects(text: str):
    """优先返回看起来像本任务 schema 的完整 JSON 对象（含 scores / predicted_pos）。"""
    preferred = []
    others = []
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
    """Gemini 长 explanation 偶发截断：尝试补全缺失的右括号。"""
    if not candidate:
        return candidate
    opens = candidate.count("{") - candidate.count("}")
    if opens <= 0:
        return candidate
    return candidate + ("}" * min(opens, 8))


def _truthy_token(s: str) -> bool:
    """把各种『符合/不符合』写法统一成 bool。"""
    t = (s or "").strip().lower()
    if t in ("true", "yes", "y", "1", "是", "对", "符合", "√", "✓", "match", "positive"):
        return True
    if t in ("false", "no", "n", "0", "否", "不", "不符合", "×", "✗", "x", "mismatch", "negative"):
        return False
    # 兜底：含“符合”且不含“不”视为 true
    if "不符合" in t or "不能" in t:
        return False
    if "符合" in t or "可以" in t:
        return True
    return False


def _ultra_salvage_from_any_text(text: str) -> Dict[str, Any]:
    """极宽松抢救：不管模型输出多乱，只要能抠出规则 true/false 或词类判断就返回。
    目标：能解析出来就算成功，避免因格式问题判失败。
    """
    if not text or not str(text).strip():
        return None

    raw = str(text)
    result = {
        "explanation": "（极宽松本地抢救解析）",
        "predicted_pos": "未知",
        "is_dual_category": False,
        "scores": {"名词": {}, "动词": {}, "名动词": {}},
    }
    found_any = False

    # ---- predicted_pos：多种中英文写法 ----
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
            found_any = True
            break

    # ---- is_dual_category ----
    m = re.search(
        r'["\']?is_dual_category["\']?\s*[:=]\s*(true|false|True|False|是|否)|'
        r'(?:是否)?兼类\s*[：:=]?\s*(是|否|true|false)',
        raw, re.IGNORECASE
    )
    if m:
        val = (m.group(1) or m.group(2) or "").strip().lower()
        result["is_dual_category"] = val in ("true", "是")
        found_any = True

    # ---- 规则分：兼容 N1 / N1_xxx / "N1_可受..." : true / 符合 ----
    # 1) 标准 key: value
    rule_pat = re.compile(
        r'["\']?\s*((?:NV|N|V)\s*\d+[_\-\u4e00-\u9fa5A-Za-z0-9]*)\s*["\']?\s*[:=：]\s*'
        r'(true|false|True|False|是|否|yes|no|符合|不符合|√|×|✓|✗|1|0)',
        re.IGNORECASE
    )
    # 2) 叙述式：N1 ... 符合/不符合
    narrative_pat = re.compile(
        r'((?:NV|N|V)\s*\d+)\s*[^。；;\n]{0,40}?(符合|不符合|可以|不能|是|否)',
        re.IGNORECASE
    )

    def _bucket(key: str) -> str:
        k = re.sub(r'\s+', '', key).upper()
        if k.startswith("NV"):
            return "名动词"
        if k.startswith("N"):
            return "名词"
        if k.startswith("V"):
            return "动词"
        return "名词"

    for rm in rule_pat.finditer(raw):
        key = re.sub(r'\s+', '', rm.group(1).strip())
        is_match = _truthy_token(rm.group(2))
        result["scores"][_bucket(key)][key] = is_match
        found_any = True

    if not any(result["scores"][p] for p in result["scores"]):
        for rm in narrative_pat.finditer(raw):
            key = re.sub(r'\s+', '', rm.group(1).strip())
            is_match = _truthy_token(rm.group(2))
            result["scores"][_bucket(key)][key] = is_match
            found_any = True

    # ---- 按词类块再扫一遍（嵌套更深时）----
    for pos in ("名词", "动词", "名动词"):
        block_m = re.search(
            rf'["\']?{pos}["\']?\s*[:=：]\s*\{{(.*?)\}}',
            raw, re.DOTALL
        )
        if not block_m:
            continue
        block = block_m.group(1)
        for rm in rule_pat.finditer(block):
            key = re.sub(r'\s+', '', rm.group(1).strip())
            is_match = _truthy_token(rm.group(2))
            result["scores"][pos][key] = is_match
            found_any = True

    # 若完全抠不到任何信号，返回 None
    has_scores = any(result["scores"][p] for p in result["scores"])
    if not has_scores and result["predicted_pos"] == "未知":
        return None

    # 没有 predicted_pos 时，用得分条目数粗略推断
    if result["predicted_pos"] == "未知" and has_scores:
        counts = {p: sum(1 for v in result["scores"][p].values() if v) for p in result["scores"]}
        best = max(counts, key=lambda k: counts[k])
        if counts[best] > 0:
            result["predicted_pos"] = best

    return result


def extract_gemini_json(text: str) -> Tuple[Dict[str, Any], str]:
    """极宽松解析器：无论模型输出多乱，尽可能抠出可用结果。
    策略顺序：
      1. 清理 Markdown / 不可见字符
      2. 标准 JSON loads
      3. 截断补全后再 loads
      4. 从文本任意位置找 JSON 对象
      5. 极宽松正则抢救规则 true/false、词类、兼类
    只要第 5 步能抠到一点信号就视为成功。
    """
    if not text:
        return None, text

    cleaned = _clean_possible_markdown_json(text)

    # 1) 整段尝试
    obj = _parse_json_object_candidate(cleaned)
    if isinstance(obj, dict) and ("scores" in obj or "predicted_pos" in obj or "is_dual_category" in obj):
        return obj, cleaned

    # 2) 优先 schema 相关对象 + 截断补全
    for candidate in _gemini_prefer_schema_objects(cleaned):
        for cand in (candidate, _gemini_try_close_truncated(candidate)):
            obj = _parse_json_object_candidate(cand)
            if isinstance(obj, dict) and ("scores" in obj or "predicted_pos" in obj or "is_dual_category" in obj):
                return obj, cand

    # 3) 任意 balanced 对象
    for candidate in _iter_balanced_json_objects(cleaned):
        for cand in (candidate, _gemini_try_close_truncated(candidate)):
            obj = _parse_json_object_candidate(cand)
            if isinstance(obj, dict):
                # 即使没有 scores 字段，只要有嵌套 dict 也先返回，后面 ask_model 会映射
                return obj, cand

    # 4) 把全文当“半结构化文本”极宽松抢救
    salvaged = _ultra_salvage_from_any_text(cleaned)
    if salvaged is not None:
        return salvaged, cleaned

    # 5) 对原始未清理文本再抢救一次（防止清理误伤）
    salvaged = _ultra_salvage_from_any_text(text)
    if salvaged is not None:
        return salvaged, text

    return None, text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str):
        return None
    k_clean = re.sub(r'[\s_]+', '', k).upper()
    # 1) 全名完全匹配
    for r in pos_rules:
        if re.sub(r'[\s_]+', '', r["name"]).upper() == k_clean:
            return r["name"]
    # 2) 短码精确匹配（N1/V6/NV10），按码长度从长到短，避免 NV1 误匹配 NV10
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
    except Exception: pass
    return mismatch

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    try:
        for pos, scores in scores_all.items():
            membership[pos] = max(-1.0, min(1.0, sum(scores.values()) / 100))
    except Exception: pass
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    try: return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception: return []

# ===============================
# LLM调用与词类判定主函数
# ===============================
def get_provider_config(provider, api_key, model, messages, max_tokens, temperature):
    """统一构造各厂商请求配置。OpenAI 用 Responses API；Gemini/xAI 用 Chat Completions。"""
    if provider == "openai":
        url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
        }
        # GPT-5.6 系列支持 reasoning_effort；批量词类判定使用 low，避免无谓增加成本。
        reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
        if reasoning_effort in {"none", "low", "medium", "high", "xhigh", "max"}:
            payload["reasoning"] = {"effort": reasoning_effort}
        return url, headers, payload, "responses"

    # ===== Gemini：官方 generateContent + responseSchema + Thought Summary =====
    # Gemini 3 的“思考”与最终答案是分开的：思考摘要通过 thought=true 的 parts 返回，
    # 最终答案仍严格限制为 JSON。这样既保留推理摘要，又不会把推理文本混进 JSON 导致解析失败。
    if provider == "gemini":
        native_base = os.getenv(
            "GEMINI_NATIVE_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        url = f"{native_base}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        contents = []
        system_bits = []
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

        # Gemini 3：优先 high，确保获得较完整的思考摘要；可通过环境变量下调。
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "high").strip().lower()
        if thinking_level not in {"low", "medium", "high"}:
            thinking_level = "high"

        payload = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_FIXED_SCHEMA,
                # 思考 token 也会占用输出预算；给足空间，避免 explanation / thought summary 被截断。
                "maxOutputTokens": max(max_tokens, 16384),
                # Gemini 3 官方建议保持 temperature 默认值 1.0。
                "temperature": 1.0,
                "thinkingConfig": {
                    "includeThoughts": True,
                    "thinkingLevel": thinking_level,
                },
            },
        }
        if system_bits:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_bits)}]
            }
        return url, headers, payload, "gemini_native"

    base_urls = {
        "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "moonshot": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        "qwen": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "xai": os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    }
    url = f"{base_urls.get(provider, base_urls['deepseek']).rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": provider not in {"moonshot"},
    }

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
    """解析 OpenAI Responses API 的 output_text / output.content.text。"""
    if not isinstance(body, dict):
        return ""
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]
    output = body.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        return "".join(chunks)
    return ""


def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3, show_ui=True):
    if not _api_key:
        return False, {"error": "API Key 为空"}, "API Key 未提供"

    url, headers, payload, api_style = get_provider_config(
        _provider, _api_key, _model, messages, max_tokens, temperature
    )
    streaming_placeholder = st.empty() if show_ui else None
    error_msg = ""

    for attempt in range(max_retries):
        try:
            if api_style == "gemini_native":
                # 官方 generateContent + responseMimeType=application/json
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code != 200:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    if response.status_code == 404:
                        error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif response.status_code == 401:
                        error_msg = "Gemini 鉴权失败 (401)。请检查 GEMINI_API_KEY。"
                    elif response.status_code == 429:
                        error_msg = f"Gemini 限流 (429)：{detail}"
                    elif response.status_code in [400, 403]:
                        error_msg = f"Gemini 请求错误 ({response.status_code})：{detail}"
                    else:
                        error_msg = f"Gemini API 错误: {response.status_code} - {detail}"
                    if response.status_code in [400, 401, 403]:
                        break
                    # 429 等可重试错误：抛出让外层退避
                    response.raise_for_status()

                body = response.json()
                logger.info("Gemini native 响应摘要：%s", json.dumps(body, ensure_ascii=False)[:4000])

                # 分离 Gemini Thought Summary 与最终 JSON。
                thought_parts = []
                answer_parts = []
                try:
                    candidates = body.get("candidates") or []
                    for candidate in candidates:
                        content = (candidate or {}).get("content") or {}
                        for part in content.get("parts", []) or []:
                            if not isinstance(part, dict):
                                continue
                            part_text = part.get("text")
                            if not isinstance(part_text, str) or not part_text.strip():
                                continue
                            if part.get("thought") is True:
                                thought_parts.append(part_text)
                            else:
                                answer_parts.append(part_text)
                except Exception:
                    thought_parts = []
                    answer_parts = []

                final_text = "".join(answer_parts).strip()
                if not final_text:
                    # 兼容极少数代理/版本将最终文本放在标准 choices 结构中的情况。
                    final_text = extract_text_from_response(body).strip()

                thought_text = "\n\n".join(t.strip() for t in thought_parts if t.strip())
                if final_text:
                    if streaming_placeholder is not None:
                        streaming_placeholder.empty()
                    # 保留原始 response，同时把 thought summary 单独放在 reasoning_content，
                    # 解析 JSON 时永远只取 content，因此不会把推理混进 JSON。
                    wrapped = {
                        "choices": [{
                            "message": {
                                "content": final_text,
                                "reasoning_content": thought_text,
                            }
                        }],
                        "_raw_gemini": body,
                    }
                    return True, wrapped, ""

                error_msg = f"Gemini native 未返回最终 JSON。思考摘要={bool(thought_text)}；响应摘要：{json.dumps(body, ensure_ascii=False)[:1200]}"

            elif api_style == "responses":
                # OpenAI Responses：非流式读取，解析稳定；不影响后台 Worker。
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                if response.status_code != 200:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    if response.status_code == 404:
                        error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif response.status_code == 401:
                        error_msg = "OpenAI 鉴权失败 (401)。请检查 OPENAI_API_KEY。"
                    elif response.status_code in [400, 403, 429]:
                        error_msg = f"OpenAI 请求错误 ({response.status_code})：{detail}"
                    else:
                        error_msg = f"OpenAI API 错误: {response.status_code} - {detail}"
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
                        if response.status_code == 404:
                            error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                        elif response.status_code == 401:
                            error_msg = f"{_provider.upper()} 鉴权失败 (401)。请检查 {_provider.upper()} API Key。"
                        elif response.status_code in [400, 403, 429]:
                            error_msg = f"{_provider.upper()} 请求错误 ({response.status_code})：{detail}"
                        else:
                            error_msg = f"API 错误: {response.status_code} - {detail}"
                        if response.status_code in [400, 401, 403]:
                            break
                        response.raise_for_status()

                    if not is_stream:
                        body = response.json()
                        if _provider == "gemini":
                            logger.info("Gemini 非流式原始响应摘要：%s", json.dumps(body, ensure_ascii=False)[:2000])
                        text = extract_text_from_response(body)
                        if text:
                            if streaming_placeholder is not None:
                                streaming_placeholder.empty()
                            return True, body, ""
                        # 不要把整段响应直接扔掉；保留安全摘要，方便定位 Gemini/xAI 兼容层问题。
                        try:
                            safe_body = json.dumps(body, ensure_ascii=False)[:1200]
                            error_msg = f"{_provider.upper()} 非流式接口未返回文本。响应摘要：{safe_body}"
                        except Exception:
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
                                    delta = chunk["choices"][0].get("delta", {}) or {}
                                    delta_text = delta.get("content", "") or ""
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
    if not word: return {}, "", "未知", "", False
    full_rules = {pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules]) for pos, rules in RULE_SETS.items()}
    
    is_gemini = provider == "gemini"
    is_gemini_flash = is_gemini and "flash" in str(model).lower()

    if is_gemini:
        # Gemini 专用：保留用户给出的原版提示词逻辑，但把“推理过程”从最终 JSON 中分离。
        # explanation = 可公开、逐条、带例子的分析依据；thought summary = API 返回的思考摘要。
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

    # 模型请求只允许 1 次；失败后由批处理层直接停止整个任务，不再重复请求。
    api_retries = 1
    with st.spinner(f"正在调用大模型 ({model}) 进行分析...") if show_ui else __import__("contextlib").nullcontext():
        ok, resp_json, err_msg = call_llm_api_cached(
            provider, model, api_key,
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}],
            show_ui=show_ui, max_retries=api_retries
        )
        
    if not ok:
        if show_ui: st.error(f"模型调用失败: {err_msg}")
        # Gemini Flash：即使接口失败也返回占位 scores，避免 Worker 再重入调用
        if is_gemini_flash:
            empty_scores = {pos: {r["name"]: 0 for r in rules} for pos, rules in RULE_SETS.items()}
            return empty_scores, f"调用失败: {err_msg}", "调用失败", f"失败: {err_msg}", False
        return {}, f"调用失败: {err_msg}", "未知", f"失败: {err_msg}", False

    # Gemini：将“最终 JSON”与“思考摘要”分开。最终 JSON 才参与解析。
    if provider == "gemini":
        raw_content = ""
        gemini_reasoning = ""
        try:
            choices = resp_json.get("choices") or []
            message = (choices[0] or {}).get("message", {}) if choices else {}
            raw_content = message.get("content", "") or ""
            gemini_reasoning = message.get("reasoning_content", "") or ""
        except Exception:
            pass
        raw_text = str(raw_content).strip()
        if gemini_reasoning.strip():
            raw_text_with_reasoning = (
                "【Gemini 思考摘要】\n" + gemini_reasoning.strip() +
                "\n\n【Gemini 最终 JSON】\n" + raw_text
            )
        else:
            raw_text_with_reasoning = raw_text
    else:
        raw_text = extract_text_from_response(resp_json)
        raw_text_with_reasoning = raw_text

    # 解析时只使用最终 JSON，避免 Gemini Thought Summary 破坏 JSON 解析。
    if provider == "gemini":
        parsed_json, _ = extract_gemini_json(raw_text)
        if parsed_json is None:
            parsed_json, _ = extract_json_from_text(raw_text)
    else:
        parsed_json, _ = extract_json_from_text(raw_text)
        if parsed_json is None:
            # 其它模型也允许极宽松抢救，能抠出来就算成功
            parsed_json, _ = extract_gemini_json(raw_text)

    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "无推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        is_dual_category = parsed_json.get("is_dual_category", False)
        raw_scores = parsed_json.get("scores", {})
        # scores 可能是 list 等异常结构，尽量拉平
        if not isinstance(raw_scores, dict):
            salv = _ultra_salvage_from_any_text(raw_text or "")
            raw_scores = (salv or {}).get("scores", {}) if salv else {}
    else:
        if show_ui: st.error("未能解析有效的 JSON。")
        # 返回占位 scores（非空 dict），Worker 写盘前进，不重试
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
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself", hovertemplate='<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>')])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[min(min(values), -0.1), max(max(values), 1.0)])), showlegend=False, title=dict(text=title, x=0.5))
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
    except Exception as e:
        return {}


def _pid_alive(pid):
    if not pid: return False
    try:
        pid = int(pid)
        if pid <= 0: return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError: return False
    except PermissionError: return True
    except (ValueError, OSError): return False


def _kill_pid(pid):
    """尝试结束指定进程（先 SIGTERM，再 SIGKILL）。"""
    if not pid:
        return False
    try:
        pid = int(pid)
        if pid <= 0 or not _pid_alive(pid):
            return False
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            return True
        time.sleep(0.4)
        if _pid_alive(pid):
            try:
                os.kill(pid, 9)  # SIGKILL
            except ProcessLookupError:
                pass
        return True
    except Exception as e:
        logger.warning(f"结束进程失败 pid={pid}: {e}")
        return False


def stop_batch_job(job_state_file: Path) -> str:
    """停止当前批次：标记 cancelled，并结束 supervisor / worker。"""
    state = _load_job_state(job_state_file)
    supervisor_pid = state.get("supervisor_pid")
    worker_pid = state.get("worker_pid")

    # 从 pid 文件再读一次，防止 state 里 pid 过期
    for suffix, key in ((".supervisor.pid", "supervisor"), (".worker.pid", "worker")):
        try:
            p = _service_file(job_state_file, suffix)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt.isdigit():
                    if key == "supervisor":
                        supervisor_pid = int(txt)
                    else:
                        worker_pid = int(txt)
        except Exception:
            pass

    _write_job_state(
        job_state_file,
        status="cancelled",
        error="用户手动停止任务",
        worker_pid=None,
    )

    killed = []
    if _kill_pid(worker_pid):
        killed.append(f"worker={worker_pid}")
    if _kill_pid(supervisor_pid):
        killed.append(f"supervisor={supervisor_pid}")

    # 清理锁/pid 文件，避免下次启动被挡
    for suffix in (".supervisor.pid", ".worker.pid", ".supervisor.lock"):
        try:
            f = _service_file(job_state_file, suffix)
            if f.exists():
                f.unlink()
        except Exception:
            pass

    if killed:
        return f"已停止任务，并结束进程：{', '.join(killed)}"
    return "已标记停止（cancelled）。若界面仍在转，请刷新页面。"


def _state_age_seconds(state):
    try:
        ts = state.get('updated_at')
        if not ts: return 0
        return max(0, time.time() - time.mktime(time.strptime(ts, '%Y-%m-%d %H:%M:%S')))
    except Exception:
        return 0


def _service_file(job_state_file: Path, suffix: str) -> Path:
    return job_state_file.with_name(job_state_file.stem + suffix)


def _batch_worker(df_input, target_col, file_name, backup_file, provider, model, api_key, job_state_file):
    total_rows = len(df_input)
    state = _load_job_state(job_state_file)
    start_row = int(state.get('next_row', 0)) if state else 0
    if state.get('file_name') != file_name:
        start_row = 0
    if start_row < 0 or start_row > total_rows:
        start_row = 0

    clean_csv_duplicates(backup_file)
    _write_job_state(
        job_state_file,
        status='running', file_name=file_name, total_rows=total_rows,
        next_row=start_row, current_row=start_row, current_word='', retry_count=0,
        error='', completed_rows=start_row
    )

    while start_row < total_rows:
        # 用户点击「停止任务」后优雅退出
        cur_state = _load_job_state(job_state_file)
        if cur_state.get("status") == "cancelled":
            _write_job_state(job_state_file, status="cancelled", error="用户手动停止任务",
                             next_row=start_row, completed_rows=start_row)
            logger.info("Worker 收到 cancelled，退出。next_row=%s", start_row)
            return

        index = start_row
        row_number = index + 1
        
        # 兼容处理 Excel 空行现象 (NaN)
        word_val = df_input.iloc[index][target_col]
        word = "" if pd.isna(word_val) else str(word_val).strip()
        task_id = _make_task_id(file_name, index)

        if not word:
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row,
                             current_row=index, current_word='', completed_rows=start_row,
                             retry_count=0, error='')
            continue

        if is_task_processed(backup_file, task_id, row_number):
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row,
                             current_row=index, current_word=word, completed_rows=start_row,
                             retry_count=0, error='检测到该 Excel 行已有结果，已跳过重复处理')
            continue

        success = False
        last_error = ''
        retry_count = 0
        scores, raw_text, pred_pos, explanation, is_dual_category = {}, '', '处理失败', '无响应', False

        # 重要：模型调用失败后立即停止整个批次。
        # 不再对同一个词持续重试，也不写入占位失败结果。
        try:
            _write_job_state(
                job_state_file,
                status='running',
                current_row=index,
                current_word=word,
                next_row=index,
                retry_count=0,
                error='',
                completed_rows=index
            )
            scores, raw_text, pred_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(
                word, provider, model, api_key, show_ui=False
            )

            if not scores:
                last_error = explanation or '模型调用失败或返回结果为空'
                detail = f'第 {row_number} 行「{word}」处理失败，批次已停止。{last_error}'
                logger.error(detail)
                _write_job_state(
                    job_state_file,
                    status='failed',
                    current_row=index,
                    current_word=word,
                    next_row=index,
                    completed_rows=index,
                    retry_count=0,
                    error=detail,
                    model_error=True
                )
                return

            # 解析失败/调用失败不能当成 0 分正常结果保存。
            if str(pred_pos) in {'解析失败', '调用失败', '处理失败', '未知'}:
                detail = f'第 {row_number} 行「{word}」结果无效（{pred_pos}），批次已停止。{explanation or "未获得有效结果"}'
                logger.error(detail)
                _write_job_state(
                    job_state_file,
                    status='failed',
                    current_row=index,
                    current_word=word,
                    next_row=index,
                    completed_rows=index,
                    retry_count=0,
                    error=detail,
                    model_error=True
                )
                return

            success = True

        except BaseException as e:
            detail = f'第 {row_number} 行「{word}」模型请求异常，批次已停止：{type(e).__name__}: {e}'
            logger.exception(detail)
            _write_job_state(
                job_state_file,
                status='failed',
                current_row=index,
                current_word=word,
                next_row=index,
                completed_rows=index,
                retry_count=0,
                error=detail,
                model_error=True
            )
            return

        if not success:
            return

        membership = calculate_membership(scores)
        new_row = pd.DataFrame([{
            '任务ID': task_id,
            '序数': row_number,
            '词语': word,
            '动词': membership.get('动词', 0.0),
            '名词': membership.get('名词', 0.0),
            '名动词': membership.get('名动词', 0.0),
            '差值/距离': round(abs(membership.get('动词', 0.0) - membership.get('名词', 0.0)), 4),
            '预测词类': pred_pos,
            '是否兼类': '是' if is_dual_category else '否',
            '模型提供商': provider,
            '模型': model,
            '原始响应': raw_text,
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S')
        }])

        _write_job_state(job_state_file, status='saving', current_row=index,
                         current_word=word, next_row=index, completed_rows=index,
                         retry_count=0, error='')
        
        write_ok, write_detail, already_exists = append_unique_csv(new_row, backup_file, task_id)
        if not write_ok:
            save_retry = 0
            while not write_ok:
                save_retry += 1
                _write_job_state(job_state_file, status='waiting_save_retry', current_row=index,
                                 current_word=word, next_row=index, completed_rows=index,
                                 retry_count=save_retry, error=write_detail,
                                 retry_in_seconds=min(30, max(2, 2 ** min(save_retry - 1, 4))))
                time.sleep(min(30, max(2, 2 ** min(save_retry - 1, 4))))
                write_ok, write_detail, already_exists = append_unique_csv(new_row, backup_file, task_id)
            _write_job_state(job_state_file, status='saving', current_row=index,
                             current_word=word, next_row=index, completed_rows=index,
                             retry_count=0, error='')

        start_row = index + 1
        _write_job_state(job_state_file, status='running', current_row=index,
                         current_word=word, next_row=start_row, completed_rows=start_row,
                         retry_count=0, error=write_detail if already_exists else '')

    _write_job_state(job_state_file, status='completed', file_name=file_name,
                     total_rows=total_rows, next_row=total_rows, completed_rows=total_rows,
                     current_word='', retry_count=0, error='')


def _worker_entry(job_state_file: Path):
    try:
        logger.info("Worker 启动 CODE_VERSION=%s pid=%s", CODE_VERSION, os.getpid())
        spec_file = _service_file(job_state_file, '.spec.json')
        if not spec_file.exists():
            raise FileNotFoundError(f'找不到任务配置文件：{spec_file}')
        spec = json.loads(spec_file.read_text(encoding='utf-8'))
        api_key = os.getenv(spec['env_var'], '')
        if not api_key:
            raise RuntimeError(f"环境变量 {spec['env_var']} 未配置")
        df_input = pd.read_excel(Path(spec['input_file']))
        _batch_worker(df_input, spec['target_col'], spec['file_name'], Path(spec['backup_file']),
                      spec['provider'], spec['model'], api_key, job_state_file)
    except BaseException as e:
        detail = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        logger.error(f'Worker致命退出：\n{detail}')
        _write_job_state(job_state_file, status='failed', error=detail,
                         worker_fatal_error=True)
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
            if state.get('status') == 'completed': return
            if state.get('status') == 'cancelled': return

            worker = spawn_detached(
                [sys.executable, str(Path(__file__).resolve()), '--worker', str(job_state_file)],
                str(Path(__file__).resolve().parent)
            )
            worker_pid_file.write_text(str(worker.pid), encoding='utf-8')
            restart_count += 1
            _write_job_state(job_state_file, status='running', supervisor_pid=os.getpid(),
                             worker_pid=worker.pid, supervisor_restart_count=restart_count,
                             error='')

            while True:
                rc = worker.poll()
                state = _load_job_state(job_state_file)
                if state.get('status') == 'completed':
                    return
                if state.get('status') == 'cancelled':
                    # 用户停止：结束 worker 后退出 supervisor
                    try:
                        if rc is None:
                            worker.terminate()
                            time.sleep(0.3)
                            if worker.poll() is None:
                                worker.kill()
                    except Exception:
                        pass
                    return
                if rc is not None:
                    break
                time.sleep(1)

            current_state = _load_job_state(job_state_file)
            # Worker 因模型/API 错误主动把任务标记为 failed 时，Supervisor 必须立即停止，禁止自动拉起。
            if current_state.get('status') == 'failed' and current_state.get('model_error'):
                logger.error('检测到模型错误导致的 failed 状态，Supervisor 停止自动重启。')
                return

            if rc == 0: detail = 'Worker正常退出，但任务状态未标记完成。将自动检查并继续。'
            else: detail = f'Worker退出，状态码 {rc}。{("错误详情：" + str(current_state.get("error"))) if current_state.get("error") else "请查看日志。"}'
            delay = min(30, max(2, 2 ** min(restart_count - 1, 4)))
            _write_job_state(job_state_file, status='supervisor_restarting', supervisor_pid=os.getpid(),
                             worker_pid=None, supervisor_restart_count=restart_count,
                             error=f'{detail} {delay} 秒后自动重启 Worker')
            time.sleep(delay)
    except BaseException as e:
        detail = f'Supervisor异常：{type(e).__name__}: {e}\n{traceback.format_exc()}'
        logger.error(detail)
        _write_job_state(job_state_file, status='failed', supervisor_pid=os.getpid(),
                         worker_pid=None, error=detail, supervisor_fatal_error=True)
        raise
    finally:
        for f in (supervisor_pid_file, worker_pid_file):
            try:
                if f.exists() and f.read_text(encoding='utf-8').strip() in {'', str(os.getpid())}:
                    f.unlink()
            except Exception: pass
        if guard_fd is not None:
            _release_pid_guard(supervisor_lock_file, guard_fd)


def start_or_resume_batch_job(df_input, target_col, uploaded_file, file_name, backup_file,
                              provider, model, env_var, job_state_file):
    state = _load_job_state(job_state_file)
    if _pid_alive(state.get('supervisor_pid')):
        return False, state

    spec = {
        'input_file': str(BASE_DIR / f"batch_input_{re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', job_state_file.stem)}.xlsx"),
        'target_col': target_col,
        'file_name': file_name,
        'backup_file': str(backup_file),
        'provider': provider,
        'model': model,
        'env_var': env_var
    }
    Path(spec['input_file']).write_bytes(uploaded_file.getvalue())
    spec_tmp = _service_file(job_state_file, '.spec.json.tmp')
    spec_tmp.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(spec_tmp, _service_file(job_state_file, '.spec.json'))

    resume_row = int(state.get('next_row', 0)) if state.get('file_name') == file_name else 0
    _write_job_state(job_state_file, status='starting', file_name=file_name,
                     total_rows=len(df_input), next_row=resume_row, completed_rows=resume_row,
                     current_row=max(0, resume_row - 1), current_word='', retry_count=0, error='')

    try:
        proc = spawn_detached([sys.executable, str(Path(__file__).resolve()), '--supervisor', str(job_state_file)],
                              str(Path(__file__).resolve().parent))
    except Exception as e:
        detail = f'启动 Supervisor 失败：{type(e).__name__}: {e}'
        _write_job_state(job_state_file, status='failed', error=detail)
        raise

    _write_job_state(job_state_file, status='starting', supervisor_pid=proc.pid,
                     worker_pid=None, error='Supervisor 已启动，等待 Worker')
    return True, _load_job_state(job_state_file)


def _supervisor_or_worker_alive(job_state_file: Path, state: dict = None) -> bool:
    """只有真实进程还在，才算任务在跑（不看过期 status 文案）。"""
    state = state or _load_job_state(job_state_file)
    if _pid_alive(state.get("supervisor_pid")) or _pid_alive(state.get("worker_pid")):
        return True
    for suffix in (".supervisor.pid", ".worker.pid"):
        try:
            p = _service_file(job_state_file, suffix)
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt.isdigit() and _pid_alive(int(txt)):
                    return True
        except Exception:
            pass
    return False


def _reconcile_stale_job_state(job_state_file: Path) -> dict:
    """状态写着 running 但进程已死 → 解锁，允许再次点「开始/继续」。"""
    state = _load_job_state(job_state_file)
    active_statuses = {
        "running", "retrying", "waiting_retry", "saving",
        "waiting_save_retry", "starting", "supervisor_restarting",
    }
    if state.get("status") not in active_statuses:
        return state
    if _supervisor_or_worker_alive(job_state_file, state):
        return state
    # 刚启动 15 秒内先不判定为僵死
    if state.get("status") == "starting" and _state_age_seconds(state) < 15:
        return state
    _write_job_state(
        job_state_file,
        status="interrupted",
        supervisor_pid=None,
        worker_pid=None,
        error="检测到后台进程已退出，任务已解锁。可点击「开始处理 / 继续断点任务」继续。",
    )
    return _load_job_state(job_state_file)


def _auto_recover_if_needed(job_state_file: Path):
    """默认不再自动拉起 supervisor（避免用户停不掉 / 按钮按不了）。
    仅当环境变量 BATCH_AUTO_RECOVER=1 时恢复旧的自动重连行为。
    """
    state = _reconcile_stale_job_state(job_state_file)
    if os.getenv("BATCH_AUTO_RECOVER", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return state

    active_statuses = {'running', 'retrying', 'waiting_retry', 'saving', 'waiting_save_retry', 'starting', 'supervisor_restarting'}
    if state.get('status') not in active_statuses:
        return state
    if _supervisor_or_worker_alive(job_state_file, state):
        return state

    if state.get('status') == 'starting' and _state_age_seconds(state) < 15:
        return state
    spec_file = _service_file(job_state_file, '.spec.json')
    if not spec_file.exists():
        return state

    try:
        proc = spawn_detached([sys.executable, str(Path(__file__).resolve()), '--supervisor', str(job_state_file)],
                              str(Path(__file__).resolve().parent))
        _write_job_state(job_state_file, status='starting', supervisor_pid=proc.pid, worker_pid=None,
                         error='检测到 Supervisor 已退出，系统正在自动重新启动并接续任务')
        return _load_job_state(job_state_file)
    except Exception as e:
        detail = f'自动恢复 Supervisor 失败：{type(e).__name__}: {e}'
        _write_job_state(job_state_file, status='failed', error=detail)
        return _load_job_state(job_state_file)


# ----------------------------------------------------------------------
# 批量结果清理
# ----------------------------------------------------------------------
def _get_bad_result_mask(df: pd.DataFrame) -> pd.Series:
    """识别批量结果中的失败/解析失败/明显计算错误记录。"""
    if df is None or df.empty:
        return pd.Series(dtype=bool)

    mask = pd.Series(False, index=df.index)

    # 预测词类/说明字段中的明确失败标记
    for col in ("预测词类", "状态", "错误", "说明", "explanation"):
        if col in df.columns:
            text = df[col].fillna("").astype(str)
            mask |= text.str.contains(
                r"调用失败|解析失败|处理失败|计算错误|JSON\s*解析失败|请求失败|未知",
                case=False, regex=True, na=False
            )

    # 数值列明显异常：三类隶属度全部为 0 且属于失败/空响应记录
    score_cols = [c for c in ("动词", "名词", "名动词") if c in df.columns]
    if len(score_cols) == 3:
        nums = df[score_cols].apply(pd.to_numeric, errors="coerce")
        all_zero = nums.fillna(0).eq(0).all(axis=1)
        invalid_num = nums.isna().any(axis=1)
        raw = df.get("原始响应", pd.Series("", index=df.index)).fillna("").astype(str)
        suspicious_raw = raw.str.contains(r"调用失败|解析失败|计算错误|429|401|403|404", case=False, regex=True, na=False)
        mask |= all_zero & suspicious_raw
        mask |= invalid_num

    return mask


def remove_bad_batch_rows(backup_file: Path) -> Tuple[bool, int, str]:
    """删除失败/解析失败/明显计算错误记录，并原子覆盖原结果文件。"""
    if not backup_file.exists() or backup_file.stat().st_size == 0:
        return True, 0, "当前没有可清理的批量结果。"
    lock_path = backup_file.with_suffix(backup_file.suffix + '.cleanup.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=15)
        df = pd.read_csv(backup_file, encoding='utf-8-sig')
        if df.empty:
            return True, 0, "当前没有可清理的批量结果。"
        bad_mask = _get_bad_result_mask(df)
        removed = int(bad_mask.sum())
        if removed == 0:
            return True, 0, "未发现失败/解析失败/明显计算错误记录。"
        cleaned = df.loc[~bad_mask].copy()
        if '序数' in cleaned.columns:
            cleaned['_seq_num'] = pd.to_numeric(cleaned['序数'], errors='coerce')
            cleaned = cleaned.sort_values('_seq_num').drop(columns=['_seq_num'])
        tmp = backup_file.with_suffix(backup_file.suffix + '.cleanup.tmp')
        cleaned.to_csv(tmp, index=False, encoding='utf-8-sig')
        os.replace(tmp, backup_file)
        return True, removed, f"已删除 {removed} 条失败/解析失败/明显计算错误记录。"
    except Exception as e:
        logger.exception(f"清理失败/错误批量记录失败: {backup_file}")
        return False, 0, f"清理失败：{type(e).__name__}: {e}"
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)


def remove_selected_batch_rows(backup_file: Path, task_ids: List[str]) -> Tuple[bool, int, str]:
    """按任务ID删除用户在结果预览中勾选的记录。"""
    task_ids = {str(x) for x in (task_ids or []) if str(x).strip()}
    if not task_ids:
        return True, 0, "未勾选任何记录。"
    if not backup_file.exists() or backup_file.stat().st_size == 0:
        return True, 0, "当前没有可删除的批量结果。"
    lock_path = backup_file.with_suffix(backup_file.suffix + '.cleanup.lock')
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=15)
        df = pd.read_csv(backup_file, encoding='utf-8-sig')
        if '任务ID' not in df.columns:
            return False, 0, "当前结果文件没有“任务ID”列，无法安全删除。"
        mask = df['任务ID'].astype(str).isin(task_ids)
        removed = int(mask.sum())
        if removed == 0:
            return True, 0, "未找到需要删除的记录。"
        cleaned = df.loc[~mask].copy()
        if '序数' in cleaned.columns:
            cleaned['_seq_num'] = pd.to_numeric(cleaned['序数'], errors='coerce')
            cleaned = cleaned.sort_values('_seq_num').drop(columns=['_seq_num'])
        tmp = backup_file.with_suffix(backup_file.suffix + '.selected_cleanup.tmp')
        cleaned.to_csv(tmp, index=False, encoding='utf-8-sig')
        os.replace(tmp, backup_file)
        return True, removed, f"已删除 {removed} 条勾选记录。"
    except Exception as e:
        logger.exception(f"删除勾选批量记录失败: {backup_file}")
        return False, 0, f"删除失败：{type(e).__name__}: {e}"
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)


# ----------------------------------------------------------------------
# 实时UI刷新组件（将所有状态同步绑定在一个片段内，杜绝延迟和不同步问题）
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
            
    # 1. 顶部统计 & 下载
    c1, c2 = st.columns([3, 1])
    with c1:
        st.metric("已存数据量 (完美去重排序)", f"{count} 条")
    with c2:
        if live_df is not None and count > 0:
            csv_data = live_df.to_csv(index=False, encoding='utf-8-sig')
            model_slug = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', str(live_df.iloc[0].get('模型', 'model')) if '模型' in live_df.columns else 'model')
            st.download_button(
                "下载完整结果(CSV)",
                data=csv_data,
                file_name=f"{st.session_state.project_code}_{model_slug}_results_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("下载结果 (无数据)", disabled=True, use_container_width=True)

    # 2. 进度条
    st.progress(min(1.0, max(0.0, completed / total)) if total else 0.0)

    # 3. 运行状态
    if status_str in {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}:
        spinner_text = {"starting": "启动任务中…", "retrying": "调用失败自动重试中…", "waiting_retry": "等待重试…", "saving": "安全排序并覆盖保存中…", "supervisor_restarting": "断流重连中…"}.get(status_str, "任务正常运行中…")
        details = f"第 {min(int(latest_state.get('current_row', 0)) + 1, total)}/{total} 行 · 当前词语「{latest_state.get('current_word', '')}」"
        if latest_state.get("retry_count", 0):
            details += f" · 重试 {latest_state.get('retry_count')} 次"
        status_html = f'<div class="running-status"><span class="running-spinner"></span><div style="flex:1"><div style="font-size:1rem;font-weight:700">{spinner_text}</div><div class="batch-detail">{details}</div>'
        if error_str:
            status_html += f'<div class="batch-detail" style="color: #ef4444;">提示：{error_str}</div>'
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
        st.caption("本次失败不会写入占位数据。修复模型/API 后，可点击“开始处理 / 继续断点任务”从失败词项继续。")
    else:
        st.info("尚未运行。点击上方“开始处理 / 继续任务”即可启动队列。")

    # 4. 实时数据表 (展示全部数据)
    st.markdown("#### 实时结果预览 (全部数据，最新结果在最上方)")
    if live_df is not None and count > 0:
        if '序数' in live_df.columns:
            live_df['序数_num'] = pd.to_numeric(live_df['序数'], errors='coerce')
            live_df = live_df.sort_values('序数_num').drop(columns=['序数_num'])
        # 全部结果倒序展示；增加“删除”勾选列，支持手动删除不满意记录。
        display_df = live_df.iloc[::-1].copy()
        if '任务ID' in display_df.columns:
            editor_df = display_df.copy()
            editor_df.insert(0, '删除', False)
            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                height=420,
                hide_index=True,
                disabled=[c for c in editor_df.columns if c != '删除'],
                column_config={
                    '删除': st.column_config.CheckboxColumn('删除', help='勾选后点击下方“删除勾选记录”')
                },
                key=f'batch_editor_{job_state_file.stem}'
            )
            selected_ids = edited_df.loc[edited_df['删除'] == True, '任务ID'].astype(str).tolist() if '删除' in edited_df.columns else []
            del_col1, del_col2 = st.columns(2)
            with del_col1:
                if st.button('🗑 删除勾选记录', use_container_width=True, key=f'del_selected_{job_state_file.stem}'):
                    ok_del, removed, msg = remove_selected_batch_rows(backup_file, selected_ids)
                    if ok_del:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with del_col2:
                if st.button('🧹 删除失败 / 计算错误记录', use_container_width=True, key=f'del_bad_{job_state_file.stem}'):
                    ok_del, removed, msg = remove_bad_batch_rows(backup_file)
                    if ok_del:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("暂无数据。任务开启后实时数据将在此严格依序显示。")


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

    with st.container():
        col1, col2, col3 = st.columns([5, 3, 2])
        with col1:
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型设置（LLM）</div>', unsafe_allow_html=True)
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("找不到可用的 API Key！请设置环境变量。")
                st.selectbox("选择大模型 (不可用)", list(MODEL_OPTIONS.keys()), disabled=True)
                selected_model_info = {"api_key": ""}
            else:
                selected_model_display_name = st.selectbox("选择大模型", list(AVAILABLE_MODEL_OPTIONS.keys()), key="model_select")
                selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
                st.markdown(
                    f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap;">'
                    f'<span class="status-badge success">● 已配置</span>'
                    f'<span style="color: #64748b; font-size: 0.85rem;">提供商: {selected_model_info["provider"].upper()}</span>'
                    f'<span style="color: #64748b; font-size: 0.85rem;">模型: {selected_model_info["model"]}</span>'
                    f'</div>', unsafe_allow_html=True
                )
        with col2:
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 实验配置</div>', unsafe_allow_html=True)
            if selected_model_info.get("provider") in {"openai", "gemini", "xai"}:
                st.caption("国外模型数据将按“批次 + 模型”独立保存，可分别跑出 Gemini / ChatGPT / Grok 数据后进行横向比较。")
            st.session_state.project_code = st.text_input("实验批次码 (Project Code)", value=st.session_state.get("project_code", "default_task"), help="隔离不同量化分析任务")
            BACKUP_FILE, PROGRESS_FILE = get_project_files(st.session_state.project_code, selected_model_info.get("model", "default_model"))
        with col3:
            st.markdown('<div class="section-title" style="justify-content: center;"><span class="icon-dot"></span> 连接测试</div>', unsafe_allow_html=True)
            st.write("")
            if st.button("测试模型链接", type="secondary", use_container_width=True, disabled=not selected_model_info["api_key"]):
                with st.spinner("正在测试连接..."):
                    # 仅 Gemini 使用更大的测试预算；其他模型保持原来的测试参数。
                    test_provider = selected_model_info["provider"]
                    test_max_tokens = 4096 if test_provider == "gemini" else 100
                    test_prompt = "请只输出 pong，不要输出 Markdown、代码块或其他文字。" if test_provider == "gemini" else "请回复'pong'"
                    ok, _, err_msg = call_llm_api_cached(
                        test_provider,
                        selected_model_info["model"],
                        selected_model_info["api_key"],
                        [{"role": "user", "content": test_prompt}],
                        max_tokens=test_max_tokens
                    )
                if ok: st.success("成功！")
                else: st.error(f"失败: {err_msg}")

    st.markdown("---")
    tab1, tab2 = st.tabs(["单个词语详细分析", "Excel 批量处理"])

    with tab1:
        st.markdown('<div class="section-title"><span class="icon-dot"></span> 词语输入</div>', unsafe_allow_html=True)
        word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
        if st.button("开始分析", type="primary", disabled=not (selected_model_info["api_key"] and word)):
            status_placeholder = st.empty()
            status_placeholder.info(f"正在为词语「{word}」启动分析...")
            scores_all, raw_text, predicted_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(word, selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"])
            status_placeholder.empty()
            if scores_all:
                membership = calculate_membership(scores_all)
                final_membership = membership.get(predicted_pos, 0)
                jianlei_badge_color = "#f59e0b" if is_dual_category else "#3b82f6"
                jianlei_text = "属于兼类词" if is_dual_category else "非兼类词"
                
                st.markdown(f'<div class="result-success-card"><div style="font-size: 1.1rem; font-weight: 600; color: #065f46;">分析完成</div><div style="margin-top: 0.5rem; font-size: 1rem; color: #065f46; line-height: 1.8;">词语「<strong>{word}</strong>」最可能的词类是 <span style="background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600;">{predicted_pos}</span>，隶属度为 <strong>{final_membership:.4f}</strong><br/>多类属性判定： <span style="background: {jianlei_badge_color}; color: white; padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 600;">{jianlei_text}</span></div></div>', unsafe_allow_html=True)
                
                col_results_1, col_results_2 = st.columns(2)
                with col_results_1:
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度排名</div>', unsafe_allow_html=True)
                    top10 = get_top_10_positions(membership)
                    for i, (pos, score) in enumerate(top10):
                        rank_class = f"top-{i+1}" if i < 3 else ""
                        st.markdown(f'<div class="rank-card {rank_class}"><div style="display: flex; align-items: center; gap: 0.75rem;"><div class="rank-num">{i+1}</div><span style="font-weight: 600; color: #1e3a5f;">{pos}</span></div><span style="font-weight: 700; color: #2d5a87; font-size: 1.1rem;">{score:.4f}</span></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 词类隶属度雷达图</div>', unsafe_allow_html=True)
                    plot_radar_chart_streamlit(dict(top10), f"「{word}」隶属度分布")
                with col_results_2:
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 各词类详细得分</div>', unsafe_allow_html=True)
                    pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in scores_all.keys()}
                    for pos in sorted(pos_total_scores.keys(), key=lambda p: pos_total_scores[p], reverse=True):
                        total_score = pos_total_scores[pos]
                        max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                        with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                            rule_data = [{"规则代码": k, "规则描述": next((r["desc"] for r in RULE_SETS.get(pos, []) if r["name"] == k), ""), "得分": v} for k, v in scores_all[pos].items()]
                            rule_df = pd.DataFrame(sorted(rule_data, key=lambda x: x["得分"], reverse=True))
                            st.dataframe(rule_df.style.map(lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, (int, float)) and x < 0 else "", subset=["得分"]), use_container_width=True, height=min(len(rule_df) * 30 + 50, 400))
                    st.markdown('<div class="section-title"><span class="icon-dot"></span> 模型原始响应</div>', unsafe_allow_html=True)
                    with st.expander("展开查看原始响应", expanded=False):
                        st.code(raw_text, language="text")

    with tab2:
        # 将静态按钮与实时刷新组件严格分离
        st.markdown(f'<div class="section-title"><span class="icon-dot"></span> 批量任务管理 (当前批次: <code>{st.session_state.project_code}</code>)</div>', unsafe_allow_html=True)
        st.caption(f"代码版本：{CODE_VERSION}（若日志无此字符串=仍在跑旧进程，请先 kill 再重启）")
        col_c1, col_c2 = st.columns([3, 1])
        with col_c2:
            # 清空缓存被放在外侧，以防止和 fragment 内的自动重刷发生冲突
            if st.button("清空本批次记录", use_container_width=True, type="secondary"):
                if os.path.exists(BACKUP_FILE):
                    try:
                        os.remove(BACKUP_FILE)
                        st.success(f"已清空批次 {st.session_state.project_code} 记录")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e: st.error(f"清空失败: {e}")

        st.divider()
        st.markdown("#### 任务配置")
        uploaded_file = st.file_uploader("选择待处理的 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df_input = pd.read_excel(uploaded_file)
                target_col = next((col for col in df_input.columns if "词" in str(col) or "word" in str(col).lower()), None)
                if target_col:
                    st.markdown(f'<div class="info-highlight"><div style="font-weight: 600; color: #1e40af;">识别到目标列: <code>{target_col}</code> | 待分析总数: <strong>{len(df_input)}</strong> 条</div></div>', unsafe_allow_html=True)
                    
                    model_namespace = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', selected_model_info.get("model", "default_model"))
                    job_state_file = BASE_DIR / f"batch_job_{re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', st.session_state.project_code)}__{model_namespace}.json"
                    # 先清理「状态写着 running 但进程已死」的僵死状态，避免开始按钮永远灰掉
                    current_state = _auto_recover_if_needed(job_state_file)

                    really_running = _supervisor_or_worker_alive(job_state_file, current_state)
                    # 只有真实进程在跑时才禁用开始按钮；不再被过期 status 卡住
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
                                    started, state = start_or_resume_batch_job(
                                        df_input, target_col, uploaded_file,
                                        f"{st.session_state.project_code}_{uploaded_file.name}",
                                        BACKUP_FILE,
                                        selected_model_info["provider"],
                                        selected_model_info["model"],
                                        selected_model_info["env_var"],
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
                        if st.button("强制解锁", type="secondary", use_container_width=True, help="进程已死但按钮仍灰时点这个"):
                            stop_batch_job(job_state_file)
                            _write_job_state(
                                job_state_file,
                                status="interrupted",
                                supervisor_pid=None,
                                worker_pid=None,
                                error="用户强制解锁，可重新开始/继续",
                            )
                            st.info("已强制解锁，请再点「开始处理 / 继续断点任务」。")
                            time.sleep(0.4)
                            st.rerun()

                    st.divider()
                    
                    # 使用 @st.fragment 将所有需刷新的指标统统打包，确保每次读取CSV时进度条和表格完美同步
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
    if "--worker" in sys.argv: _worker_entry(Path(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--supervisor" in sys.argv: _supervisor_entry(Path(sys.argv[sys.argv.index("--supervisor") + 1]))
    else: main()

if not SERVICE_MODE:
    st.markdown('<div style="text-align: center; color: #999; font-size: 13px; line-height: 1.8; padding: 10px 0 5px 0;">汉语词类隶属度检测划类平台 · © 2026 Ryan<br><a href="mailto:shenrui26@gmail.com" style="color: #999; text-decoration: none;">✉ shenrui26@gmail.com</a></div>', unsafe_allow_html=True)

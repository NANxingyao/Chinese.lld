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

def get_project_files(project_code: str) -> Tuple[Path, Path]:
    safe_code = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', project_code)
    if not safe_code: safe_code = "default_task"
    return BASE_DIR / f"batch_history_{safe_code}.csv", BASE_DIR / f"process_progress_{safe_code}.json"

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
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o（推荐）": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Google Gemini 1.5 Pro": {"provider": "gemini", "model": "gemini-1.5-pro", "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"},
    "Google Gemini 1.5 Flash": {"provider": "gemini", "model": "gemini-1.5-flash", "api_key": os.getenv("GEMINI_API_KEY"), "env_var": "GEMINI_API_KEY"},
    "Moonshot（Kimi）": {"provider": "moonshot", "model": "kimi-k2.6", "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"},
    "Qwen（通义千问）": {"provider": "qwen", "model": "qwen-max", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS: AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 安全操作辅助函数
# ===============================

def _lock_file(lock_path: Path, timeout=30, poll=0.2):
    """跨进程排他锁：基于 O_EXCL，避免多进程同时检查/写入。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
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


def _normalize_word(word: Any) -> str:
    """统一词语键：处理空白、换行，避免视觉相同但字符串不同造成重复。"""
    if word is None:
        return ""
    return re.sub(r"\s+", "", str(word)).strip()


def _csv_lock_path(file_path: Path) -> Path:
    """所有 CSV 维护、检查、写入统一使用同一把锁，避免不同锁互相失效。"""
    return file_path.with_suffix(file_path.suffix + '.lock')


def clean_csv_duplicates(file_path: Path):
    """启动/展示前统一清理历史重复：优先任务ID，其次序数，最后词语；保留最后一次结果。"""
    if not file_path.exists():
        return
    lock_path = _csv_lock_path(file_path)
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=30, poll=0.2)
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        original_len = len(df)
        if original_len == 0:
            return

        # 规范化词语，但不覆盖原始显示值。
        if '词语' in df.columns:
            word_keys = df['词语'].map(_normalize_word)
        else:
            word_keys = pd.Series([''] * len(df), index=df.index)

        # 严格按一个统一主键去重：任务ID > 序数 > 词语。
        duplicate_mask = pd.Series(False, index=df.index)
        if '任务ID' in df.columns:
            duplicate_mask |= df['任务ID'].astype(str).duplicated(keep='last')
        if '序数' in df.columns:
            serial = pd.to_numeric(df['序数'], errors='coerce')
            valid = serial.notna()
            duplicate_mask |= valid & serial.duplicated(keep='last')
        # 当前项目是“词表逐词分析”，同一个词只保留一条最终结果，防止旧版本产生的重复继续污染。
        nonempty_word = word_keys != ''
        duplicate_mask |= nonempty_word & word_keys.duplicated(keep='last')

        if duplicate_mask.any():
            df = df.loc[~duplicate_mask].copy()
            tmp = file_path.with_suffix(file_path.suffix + '.dedup.tmp')
            df.to_csv(tmp, index=False, encoding='utf-8-sig')
            os.replace(tmp, file_path)
            logger.warning(f"已清理 CSV 重复数据：{original_len} -> {len(df)}，删除 {original_len-len(df)} 条")
    except Exception as e:
        logger.exception(f"清理 CSV 重复项失败: {file_path}: {e}")
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)


def append_unique_csv(df: pd.DataFrame, file_path: Path, task_id: str, max_retries=60):
    """真正的 exactly-once 写入：同一把锁内完成读取、去重、判断、写入，禁止并发重复。"""
    lock_path = _csv_lock_path(file_path)
    last_error = ''
    for attempt in range(1, max_retries + 1):
        fd = None
        try:
            fd = _lock_file(lock_path, timeout=10)

            if file_path.exists() and file_path.stat().st_size > 0:
                existing = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                existing = pd.DataFrame()

            serial = int(df.iloc[0]['序数']) if '序数' in df.columns and pd.notna(df.iloc[0]['序数']) else None
            word = _normalize_word(df.iloc[0].get('词语', ''))

            if not existing.empty:
                # 兼容旧 CSV：先补任务ID列。
                if '任务ID' not in existing.columns:
                    if '序数' in existing.columns:
                        existing['任务ID'] = existing['序数'].apply(
                            lambda x: f'legacy::row::{int(x)}' if pd.notna(x) else ''
                        )
                    else:
                        existing['任务ID'] = [f'legacy::index::{i+1}' for i in range(len(existing))]

                # 先清理锁内已经存在的历史重复，避免继续累积。
                keys = existing['任务ID'].astype(str)
                keep = ~keys.duplicated(keep='last')
                if '序数' in existing.columns:
                    serials = pd.to_numeric(existing['序数'], errors='coerce')
                    valid = serials.notna()
                    keep &= ~(valid & serials.duplicated(keep='last'))
                if '词语' in existing.columns:
                    words = existing['词语'].map(_normalize_word)
                    nonempty = words != ''
                    keep &= ~(nonempty & words.duplicated(keep='last'))
                if not keep.all():
                    existing = existing.loc[keep].copy()

                existing_ids = set(existing['任务ID'].astype(str).tolist())
                if task_id in existing_ids:
                    return True, '该任务ID已存在，跳过重复写入。', True

                if serial is not None and '序数' in existing.columns:
                    serials = set(pd.to_numeric(existing['序数'], errors='coerce').dropna().astype(int).tolist())
                    if serial in serials:
                        return True, '该 Excel 行已存在结果，跳过重复写入。', True

                if word and '词语' in existing.columns:
                    words = set(existing['词语'].map(_normalize_word).tolist())
                    if word in words:
                        return True, '该词语已有结果，跳过重复写入。', True

            # 若上面的锁内清理改变了 existing，则先整体原子回写一次。
            needs_rewrite = file_path.exists() and not existing.empty
            if needs_rewrite:
                # 确保新旧列一致，追加时保持完整结构。
                existing_cols = list(existing.columns)
                for c in df.columns:
                    if c not in existing_cols:
                        existing[c] = ''
                for c in existing_cols:
                    if c not in df.columns:
                        df[c] = ''
                df = df[existing.columns]
                combined = pd.concat([existing, df], ignore_index=True)
                tmp = file_path.with_suffix(file_path.suffix + '.append.tmp')
                combined.to_csv(tmp, index=False, encoding='utf-8-sig')
                os.replace(tmp, file_path)
            else:
                header_needed = not file_path.exists() or file_path.stat().st_size == 0
                with open(file_path, 'a', encoding='utf-8-sig', newline='') as f:
                    df.to_csv(f, header=header_needed, index=False)
            return True, '写入成功。', False

        except Exception as e:
            last_error = f'{type(e).__name__}: {e}'
            logger.exception(f'CSV exactly-once 写入失败（第{attempt}/{max_retries}次）: {last_error}')
            time.sleep(min(3, 0.25 * attempt))
        finally:
            if fd is not None:
                _unlock_file(lock_path, fd)
    return False, f'CSV写入最终失败：{last_error}', False


def is_task_processed(file_path: Path, task_id: str, row_number: int, word: str = '') -> bool:
    """快速预检；最终是否写入仍必须以 append_unique_csv 的锁内检查为准。"""
    if not file_path.exists():
        return False
    lock_path = _csv_lock_path(file_path)
    fd = None
    try:
        fd = _lock_file(lock_path, timeout=10, poll=0.1)
        if file_path.stat().st_size == 0:
            return False
        cols = pd.read_csv(file_path, encoding='utf-8-sig', nrows=0).columns.tolist()
        if '任务ID' in cols:
            existing = pd.read_csv(file_path, encoding='utf-8-sig', usecols=['任务ID'])
            if task_id in set(existing['任务ID'].astype(str).tolist()):
                return True
        if '序数' in cols:
            existing = pd.read_csv(file_path, encoding='utf-8-sig', usecols=['序数'])
            vals = pd.to_numeric(existing['序数'], errors='coerce').dropna().astype(int)
            if row_number in set(vals.tolist()):
                return True
        if word and '词语' in cols:
            existing = pd.read_csv(file_path, encoding='utf-8-sig', usecols=['词语'])
            key = _normalize_word(word)
            return key in set(existing['词语'].map(_normalize_word).tolist())
    except Exception as e:
        logger.warning(f'检查任务是否已处理失败，将交由原子写入最终判定：{type(e).__name__}: {e}')
        return False
    finally:
        if fd is not None:
            _unlock_file(lock_path, fd)
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
    if not isinstance(resp_json, dict): return ""
    try:
        if "output" in resp_json and "text" in resp_json["output"]: return resp_json["output"]["text"]
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]: return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception:
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
            except json.JSONDecodeError: pass
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
        if re.sub(r'[\s_]+', '', r["name"]).upper() == k_clean: return r["name"]
    for r in pos_rules:
        code_match = re.match(r'^(NV\d+|N\d+|V\d+)', re.sub(r'[\s_]+', '', r["name"]).upper())
        if code_match and (k_clean == code_match.group(1) or k_clean.startswith(code_match.group(1)) or code_match.group(1) in k_clean):
            return r["name"]
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

def get_history_count(backup_file):
    if not os.path.exists(backup_file): return 0
    try: return len(pd.read_csv(backup_file, encoding='utf-8-sig'))
    except Exception: return 0

# ===============================
# LLM调用与词类判定主函数
# ===============================
def get_provider_config(provider, api_key, model, messages, max_tokens, temperature):
    base_urls = {
        "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "gemini": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"), 
        "moonshot": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
        "qwen": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    }
    url = f"{base_urls.get(provider, base_urls['openai']).rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": provider != "moonshot"}
    if provider == "moonshot":
        payload.update({"max_tokens": 8192, "thinking": {"type": "disabled"}, "temperature": 0.6})
    else:
        payload["max_tokens"] = max_tokens
    return url, headers, payload

def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0, max_retries=3, show_ui=True):
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    url, headers, payload = get_provider_config(_provider, _api_key, _model, messages, max_tokens, temperature)
    streaming_placeholder = st.empty() if show_ui else None
    error_msg = ""
    is_kimi = _provider == "moonshot"

    for attempt in range(max_retries):
        try:
            with requests.post(url, headers=headers, json=payload, stream=not is_kimi, timeout=120) as response:
                if response.status_code != 200:
                    try: detail = response.json()
                    except Exception: detail = response.text
                    if response.status_code == 404: error_msg = f"路径错误 (404)。请确保请求地址正确：{url}"
                    elif response.status_code == 401: error_msg = "鉴权失败 (401)。请检查 API Key 权限。"
                    elif response.status_code in [400, 403]:
                        error_msg = f"请求错误 ({response.status_code})：{detail}"
                        break
                    else: error_msg = f"API 错误: {response.status_code} - {detail}"
                    response.raise_for_status()

                if is_kimi:
                    try: body = response.json()
                    except Exception: body = None
                    if isinstance(body, dict):
                        text = extract_text_from_response(body)
                        if text:
                            if streaming_placeholder is not None: streaming_placeholder.empty()
                            return True, body, ""
                    error_msg = "非流式接口返回异常"
                    break

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
                            delta_text = chunk["choices"][0].get("delta", {}).get("content", "") or ""
                            if delta_text: full_content += delta_text
                    except json.JSONDecodeError: continue

                if full_content:
                    if streaming_placeholder is not None: streaming_placeholder.empty()
                    return True, {"choices": [{"message": {"content": full_content}}]}, ""
                else:
                    error_msg = "模型未返回有效内容"
        except Exception as e:
            error_msg = f"请求异常（第{attempt+1}次尝试）: {str(e)}"
            time.sleep(2 ** attempt)

    if streaming_placeholder is not None: streaming_placeholder.empty()
    return False, {"error": error_msg}, error_msg

def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str, show_ui=True) -> Tuple[Dict[str, Dict[str, int]], str, str, str, bool]:
    if not word: return {}, "", "未知", "", False
    full_rules = {pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules]) for pos, rules in RULE_SETS.items()}
    system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：
- 需要判断的词类：名词、动词、名动词
- 你只需要判断每一条规则是"符合"还是"不符合"，在 JSON 中的 scores 里给出 true / false，程序自动赋值。
【名词】\n{full_rules["名词"]}\n【动词】\n{full_rules["动词"]}\n【名动词】\n{full_rules["名动词"]}
输出要求：
1. explanation: 逐条规则说明判断依据并举例。
2. scores: 各规则对应 true/false。
3. predicted_pos: 选择最典型词类。
4. is_dual_category: 是否属于兼类（true/false）。
严格返回一段合法JSON，不要带多余字符。格式：
{{"explanation": "...", "predicted_pos": "...", "is_dual_category": true, "scores": {{"名词": {{...}}, "动词": {{...}}, "名动词": {{...}}}}}}"""
    
    user_prompt = f"请严格按照上述要求分析词语「{word}」。先进行详细推理过程，最后输出 JSON。"
    with st.spinner(f"正在调用大模型 ({model}) 进行分析...") if show_ui else __import__("contextlib").nullcontext():
        ok, resp_json, err_msg = call_llm_api_cached(provider, model, api_key, [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}], show_ui=show_ui)
        
    if not ok:
        if show_ui: st.error(f"模型调用失败: {err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", f"失败: {err_msg}", False

    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)
    
    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "无推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        is_dual_category = parsed_json.get("is_dual_category", False)
        raw_scores = parsed_json.get("scores", {})
    else:
        if show_ui: st.error("未能解析有效的 JSON。")
        return {}, raw_text, "未知", "JSON 解析失败", False

    scores_out = {pos: {r["name"]: 0 for r in rules} for pos, rules in RULE_SETS.items()}
    for pos, rules in RULE_SETS.items():
        if pos in raw_scores and isinstance(raw_scores[pos], dict):
            for k, v in raw_scores[pos].items():
                norm_k = normalize_key(k, rules)
                if norm_k:
                    rule_def = next(r for r in rules if r["name"] == norm_k)
                    scores_out[pos][norm_k] = map_to_allowed_score(rule_def, v)
    return scores_out, raw_text, predicted_pos, explanation, is_dual_category

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
    """带锁的原子状态更新；任何失败都写入日志，不再静默吞掉。"""
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
        logger.warning(f"读取任务状态失败: {type(e).__name__}: {e}")
        return {}


def _pid_alive(pid):
    """正确判断 PID 是否存在。原代码的 'or True' 会导致已死亡进程永远被判定为存活。"""
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


def _state_age_seconds(state):
    try:
        ts = state.get('updated_at')
        if not ts:
            return 0
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
        index = start_row
        row_number = index + 1
        word = str(df_input.iloc[index][target_col]).strip()
        task_id = _make_task_id(file_name, index)

        if not word:
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row,
                             current_row=index, current_word='', completed_rows=start_row,
                             retry_count=0, error='')
            continue

        if is_task_processed(backup_file, task_id, row_number, word):
            start_row = index + 1
            _write_job_state(job_state_file, status='running', next_row=start_row,
                             current_row=index, current_word=word, completed_rows=start_row,
                             retry_count=0, error='检测到该 Excel 行已有结果，已跳过重复处理')
            continue

        success = False
        last_error = ''
        retry_count = 0
        scores, raw_text, pred_pos, explanation, is_dual_category = {}, '', '处理失败', '无响应', False

        while not success:
            retry_count += 1
            _write_job_state(job_state_file,
                             status='retrying' if retry_count > 1 else 'running',
                             current_row=index, current_word=word, next_row=index,
                             retry_count=retry_count, error=last_error, completed_rows=index)
            try:
                scores, raw_text, pred_pos, explanation, is_dual_category = ask_model_for_pos_and_scores(
                    word, provider, model, api_key, show_ui=False
                )
                if scores:
                    success = True
                    break
                last_error = explanation or '模型返回为空或 JSON 解析失败'
            except BaseException as e:
                last_error = f'{type(e).__name__}: {e}'
                logger.exception(f'Worker处理第{row_number}行失败：{word}')

            wait_seconds = min(60, max(2, 2 ** min(retry_count - 1, 5)))
            _write_job_state(job_state_file, status='waiting_retry', current_row=index,
                             current_word=word, next_row=index, retry_count=retry_count,
                             error=last_error, retry_in_seconds=wait_seconds, completed_rows=index)
            time.sleep(wait_seconds)

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
            '原始响应': raw_text,
            '时间戳': time.strftime('%Y-%m-%d %H:%M:%S')
        }])

        _write_job_state(job_state_file, status='saving', current_row=index,
                         current_word=word, next_row=index, completed_rows=index,
                         retry_count=0, error='')
        write_ok, write_detail, already_exists = append_unique_csv(new_row, backup_file, task_id)
        if not write_ok:
            # 保存失败不推进行号，进入持续重试，而不是把任务静默结束
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

        # 到这里意味着结果已经存在于磁盘：无论是本次新写入还是发现已经写过，都推进断点
        start_row = index + 1
        _write_job_state(job_state_file, status='running', current_row=index,
                         current_word=word, next_row=start_row, completed_rows=start_row,
                         retry_count=0, error=write_detail if already_exists else '')

    _write_job_state(job_state_file, status='completed', file_name=file_name,
                     total_rows=total_rows, next_row=total_rows, completed_rows=total_rows,
                     current_word='', retry_count=0, error='')


def _worker_entry(job_state_file: Path):
    try:
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
    """防止同一批次同时出现两个 Supervisor。"""
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
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode('utf-8'))
            return fd
        raise RuntimeError(f'同一批次已有 Supervisor 在运行（PID {old_pid}）')


def _release_pid_guard(lock_file: Path, fd):
    try:
        os.close(fd)
    finally:
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


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
            if state.get('status') == 'completed':
                return
            if state.get('status') == 'cancelled':
                return

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
                if rc is not None:
                    break
                time.sleep(1)

            # Worker非正常退出：记录精确错误，并自动重启，不把任务当成完成
            if rc == 0:
                detail = 'Worker正常退出，但任务状态未标记完成。将自动检查并继续。'
            else:
                detail = f'Worker退出，状态码 {rc}。{("错误详情：" + str(_load_job_state(job_state_file).get("error"))) if _load_job_state(job_state_file).get("error") else "未捕获到错误详情。请查看 process_log.log。"}'
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
            except Exception:
                pass
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

    # 立刻记录父进程PID，防止启动竞态导致页面重复拉起
    _write_job_state(job_state_file, status='starting', supervisor_pid=proc.pid,
                     worker_pid=None, error='Supervisor 已启动，等待 Worker')
    return True, _load_job_state(job_state_file)


def _auto_recover_if_needed(job_state_file: Path):
    state = _load_job_state(job_state_file)
    active_statuses = {'running', 'retrying', 'waiting_retry', 'saving', 'waiting_save_retry', 'starting', 'supervisor_restarting'}
    if state.get('status') not in active_statuses:
        return state

    if _pid_alive(state.get('supervisor_pid')):
        return state

    # 再检查 Supervisor PID 文件
    try:
        pid_text = _service_file(job_state_file, '.supervisor.pid').read_text(encoding='utf-8').strip()
        if _pid_alive(int(pid_text)):
            return state
    except Exception:
        pass

    # 启动窗口内不要重复拉起
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
                st.markdown(f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;"><span class="status-badge success">● 已配置</span><span style="color: #64748b; font-size: 0.85rem;">提供商: {selected_model_info["provider"].upper()}</span></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-title"><span class="icon-dot"></span> 实验配置</div>', unsafe_allow_html=True)
            st.session_state.project_code = st.text_input("实验批次码 (Project Code)", value=st.session_state.get("project_code", "default_task"), help="隔离不同量化分析任务")
            BACKUP_FILE, PROGRESS_FILE = get_project_files(st.session_state.project_code)
        with col3:
            st.markdown('<div class="section-title" style="justify-content: center;"><span class="icon-dot"></span> 连接测试</div>', unsafe_allow_html=True)
            st.write("")
            if st.button("测试模型链接", type="secondary", use_container_width=True, disabled=not selected_model_info["api_key"]):
                with st.spinner("正在测试连接..."):
                    ok, _, err_msg = call_llm_api_cached(selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"], [{"role": "user", "content": "请回复'pong'"}], max_tokens=10)
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
        st.markdown(f'<div class="section-title"><span class="icon-dot"></span> 批量任务实时监控 (当前批次: <code>{st.session_state.project_code}</code>)</div>', unsafe_allow_html=True)
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            metric_placeholder = st.empty()
            metric_placeholder.metric("已存数据量", f"{get_history_count(BACKUP_FILE)} 条")
            if os.path.exists(BACKUP_FILE): st.caption(f"存储表: `{BACKUP_FILE.name}`")
        with ctrl_col2:
            if os.path.exists(BACKUP_FILE):
                with open(BACKUP_FILE, "rb") as f:
                    st.download_button("下载结果 (CSV)", data=f, file_name=f"{st.session_state.project_code}_results_{time.strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
            else: st.button("下载结果", disabled=True, use_container_width=True)
        with ctrl_col3:
            if st.button("清空本批次记录", use_container_width=True, type="secondary"):
                if os.path.exists(BACKUP_FILE):
                    try:
                        os.remove(BACKUP_FILE)
                        st.success(f"已清空批次 {st.session_state.project_code} 记录")
                        st.rerun()
                    except Exception as e: st.error(f"清空失败: {e}")

        st.divider()
        progress_bar, status_info = st.progress(0), st.empty()

        st.markdown("#### 实时结果预览")
        table_placeholder = st.empty()

        st.divider()
        st.markdown("#### 上传新任务")
        uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df_input = pd.read_excel(uploaded_file)
                target_col = next((col for col in df_input.columns if "词" in str(col) or "word" in str(col).lower()), None)
                if target_col:
                    st.markdown(f'<div class="info-highlight"><div style="font-weight: 600; color: #1e40af;">识别到目标列: <code>{target_col}</code> | 待分析总数: <strong>{len(df_input)}</strong> 条</div></div>', unsafe_allow_html=True)
                    
                    job_state_file = BASE_DIR / f"batch_job_{re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', st.session_state.project_code)}.json"
                    current_state = _auto_recover_if_needed(job_state_file)
                    
                    can_start = not _pid_alive(current_state.get("supervisor_pid")) and current_state.get("status") not in {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}

                    if st.button("开始处理 / 继续任务", type="primary", use_container_width=True, disabled=not can_start):
                        if not selected_model_info["api_key"]: st.error("请配置有效的 API Key")
                        else:
                            try:
                                started, state = start_or_resume_batch_job(df_input, target_col, uploaded_file, f"{st.session_state.project_code}_{uploaded_file.name}", BACKUP_FILE, selected_model_info["provider"], selected_model_info["model"], selected_model_info["env_var"], job_state_file)
                                if started: st.success("守护任务已启动，将在后台持续处理并自动恢复。"); st.rerun()
                            except Exception as e: st.error(f"启动失败：{e}")

                else: st.error("未识别到包含'词'或'word'的目标列")
            except Exception as e: st.error(f"读取Excel文件失败: {e}")
        if uploaded_file and target_col:
            def render_live_table():
                """实时读取 CSV，避免表格停留在首次渲染时的数据。"""
                try:
                    if BACKUP_FILE.exists():
                        for _ in range(3):
                            try:
                                live_df = pd.read_csv(BACKUP_FILE, encoding='utf-8-sig')
                                table_placeholder.dataframe(live_df, use_container_width=True, height=300)
                                metric_placeholder.metric("已存数据量", f"{len(live_df)} 条")
                                return
                            except (pd.errors.EmptyDataError, PermissionError, OSError):
                                time.sleep(0.15)
                        table_placeholder.info("正在写入最新结果，下一次刷新会自动更新表格……")
                    else:
                        table_placeholder.info("暂无数据。开始后结果将在此逐行实时显示。")
                except Exception as e:
                    table_placeholder.warning(f"实时表格刷新失败（下一轮会自动重试）: {e}")

            def render_batch_status():
                latest_state = _auto_recover_if_needed(job_state_file)
                total = int(latest_state.get("total_rows", len(df_input)) or len(df_input))
                completed = int(latest_state.get("completed_rows", latest_state.get("next_row", 0)) or 0)
                status_str = str(latest_state.get("status", ""))
                error_str = str(latest_state.get("error", ""))

                progress_bar.progress(min(1.0, max(0.0, completed / total)) if total else 0.0)

                if status_str in {"starting", "running", "retrying", "waiting_retry", "saving", "waiting_save_retry", "supervisor_restarting"}:
                    spinner_text = {"starting": "启动任务中…", "retrying": "调用失败自动重试中…", "waiting_retry": "等待重试…", "saving": "安全保存中…", "supervisor_restarting": "断流重连中…"}.get(status_str, "任务正常运行中…")
                    details = f"第 {min(int(latest_state.get('current_row', 0)) + 1, total)}/{total} 行 · 当前词语「{latest_state.get('current_word', '')}」"
                    if latest_state.get("retry_count", 0):
                        details += f" · 重试 {latest_state.get('retry_count')} 次"
                    status_html = f'<div class="running-status"><span class="running-spinner"></span><div style="flex:1"><div style="font-size:1rem;font-weight:700">{spinner_text}</div><div class="batch-detail">{details}</div>'
                    if error_str:
                        status_html += f'<div class="batch-detail">最近状态：{error_str}</div>'
                    status_html += '</div></div>'
                    status_info.markdown(status_html, unsafe_allow_html=True)
                elif status_str == "completed":
                    progress_bar.progress(1.0)
                    status_info.success(f"🎉 任务完毕，共 {total} 条。")
                elif status_str == "failed":
                    status_info.error(f"❌ 任务停止：{error_str or '未记录到具体异常。请查看 process_log.log 获取 Worker/Supervisor 的异常堆栈。'}")
                else:
                    status_info.info("等待开始任务。点击上方“开始处理 / 继续任务”即可启动。")

                render_live_table()

            if hasattr(st, "fragment"):
                @st.fragment(run_every="2s")
                def _live_batch_status():
                    render_batch_status()
                _live_batch_status()
            else:
                render_batch_status()


if __name__ == "__main__":
    if "--worker" in sys.argv: _worker_entry(Path(sys.argv[sys.argv.index("--worker") + 1]))
    elif "--supervisor" in sys.argv: _supervisor_entry(Path(sys.argv[sys.argv.index("--supervisor") + 1]))
    else: main()

if not SERVICE_MODE:
    st.markdown('<div style="text-align: center; color: #999; font-size: 13px; line-height: 1.8; padding: 10px 0 5px 0;">汉语词类隶属度检测划类平台 · © 2025 Ryan<br><a href="mailto:shenrui26@gmail.com" style="color: #999; text-decoration: none;">✉ shenrui26@gmail.com</a></div>', unsafe_allow_html=True)

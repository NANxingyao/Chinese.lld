import streamlit as st
import requests
import json
import re
import os
import time
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===============================
# 全局配置：并发/连接池/重试
# ===============================
# 线程池大小（根据API并发限制调整）
MAX_WORKERS = 5
# 连接池配置
SESSION = requests.Session()
RETRY_STRATEGY = Retry(
    total=2,  # 重试次数
    backoff_factor=0.1,  # 重试间隔
    status_forcelist=[429, 500, 502, 503, 504],  # 重试的状态码
    allowed_methods=["POST"]
)
ADAPTER = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", ADAPTER)
SESSION.mount("http://", ADAPTER)

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测划类（并发提速版）",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 自定义CSS
hide_streamlit_style = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.dataframe {font-size: 12px;}
[data-testid="stSidebar"] {display: none !important;}
.stApp > div:first-child {padding-top: 2rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 模型配置（仅环境变量）
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "input": {"messages": messages}, 
            "parameters": {"max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0),},
        },
    },
}

MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek", 
        "model": "deepseek-chat", 
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-759d66c83f374a2aaac0db5814ccb016"),
        "env_var": "DEEPSEEK_API_KEY"
    },
    "OpenAI GPT-4o（测试）": {
        "provider": "openai", 
        "model": "gpt-4o-mini", 
        "api_key": os.getenv("OPENAI_API_KEY", "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"),
        "env_var": "OPENAI_API_KEY"
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot", 
        "model": "moonshot-v1-32k", 
        "api_key": os.getenv("MOONSHOT_API_KEY", "sk-l5FvRWegjM5DEk4AU71YPQ1QgvFPTHZIJOmq6qdssPY4sNtE"),
        "env_var": "MOONSHOT_API_KEY"
    },
    "Qwen（通义千问）": {
        "provider": "qwen", 
        "model": "qwen-max", 
        "api_key": os.getenv("QWEN_API_KEY", "sk-b3f7a1153e6f4a44804a296038aa86c5"),
        "env_var": "QWEN_API_KEY"
    },
}

# ===============================
# 词类规则（直出分数）
# ===============================
RULE_SETS = {
    "名词": [
        {"name": "N1_可受数量词修饰", "desc": "可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "N2_不能受副词修饰", "desc": "不能受副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "N3_可作主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": 0},
        {"name": "N4_可作中心语或作定语", "desc": "可以做中心语受其他名词修饰，或者作定语直接修饰其他名词", "match_score": 10, "mismatch_score": 0},
        {"name": "N5_可后附的字结构", "desc": "可以后附助词“的”构成“的”字结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N6_可后附方位词构处所", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
        {"name": "N7_不能作谓语核心", "desc": "不能做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "N8_不能作补语/一般不作状语", "desc": "不能作补语，并且一般不能做状语直接修饰动词性成分", "match_score": 10, "mismatch_score": 0},
    ],
    "动词": [
        {"name": "V1_可受否定'不/没有'修饰", "desc": "可以受否定副词'不'或'没有'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "V2_可后附/插入时体助词'着/了/过'", "desc": "可以后附或中间插入时体助词'着/了/过'", "match_score": 10, "mismatch_score": 0},
        {"name": "V3_可带真宾语或通过介词引导论元", "desc": "可以带真宾语，或通过介词引导论元", "match_score": 20, "mismatch_score": 0},
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "V5_可有重叠/正反重叠形式", "desc": "可以有'VV, V一V'等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "V6_可做谓语或谓语核心", "desc": "可以做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "V7_不能作状语修饰动词性成分", "desc": "不能作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "V8_可作'怎么/怎样'提问或'这么/这样/那么'回答", "desc": "可以跟在'怎么/怎样'之后提问", "match_score": 10, "mismatch_score": 0},
        {"name": "V9_不能跟在'多/多么'之后提问或表示感叹", "desc": "不能跟在'多'之后对性质提问", "match_score": 10, "mismatch_score": -10},
    ],
    "名动词": [
        {"name": "NV1_可被\"不/没有\"否定且肯定形式-1", "desc": "可以用\"不\"和\"没有\"来否定", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "可以有\"VV、V一V\"等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语，又可以作主语或宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后提问", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多\"之后对性质的程度进行提问", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
    ]
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 核心工具函数（并发优化）
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """提取模型响应文本（适配多模型格式）"""
    if not isinstance(resp_json, dict): return ""
    try:
        # 通义千问
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        # OpenAI系列
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            for k in ("content", "text"):
                if k in choice: return choice[k]
    except Exception: 
        pass
    return json.dumps(resp_json, ensure_ascii=False)

def fix_common_json_errors(json_str: str) -> str:
    """自动修复JSON格式错误（高性能版）"""
    json_str = re.sub(r'([{,]\s*)([\w_]+)(\s*:)', r'\1"\2"\3', json_str)  # 补键引号
    json_str = re.sub(r"([{,]\s*)'([\w_]+)'(\s*:)", r'\1"\2"\3', json_str)  # 单引号改双引号
    json_str = re.sub(r'("[\w_]+":\s*[^,}]+)\s+("[\w_]+":)', r'\1,\2', json_str)  # 补逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # 删末尾逗号
    json_str = json_str.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')  # 中文标点转英文
    return json_str.strip()

def extract_json_from_text(text: str) -> Tuple[Optional[dict], str]:
    """并发友好的JSON提取（优先分隔符）"""
    if not text:
        return None, ""

    # 1. 专属分隔符提取
    start_marker = "====JSON_BEGIN===="
    end_marker = "====JSON_END===="
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx + len(start_marker):end_idx].strip()
        json_str = fix_common_json_errors(json_str)
        try:
            parsed = json.loads(json_str)
            return parsed, json_str
        except Exception as e:
            st.warning(f"分隔符JSON解析失败：{str(e)[:80]}")
            return None, json_str

    # 2. 代码块提取
    json_block_pattern = re.compile(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', re.IGNORECASE)
    for json_str in json_block_pattern.findall(text):
        json_str = fix_common_json_errors(json_str.strip())
        try:
            parsed = json.loads(json_str)
            return parsed, json_str
        except:
            continue

    # 3. 大括号提取
    for json_str in re.findall(r'\{[\s\S]*\}', text):
        json_str = fix_common_json_errors(json_str.strip())
        try:
            parsed = json.loads(json_str)
            return parsed, json_str
        except:
            continue

    return None, text

def validate_score_worker(rule: dict, raw_val: Any) -> Tuple[str, int]:
    """单规则分数验证（线程池任务）"""
    rule_name = rule["name"]
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    
    # 数字直接验证
    if isinstance(raw_val, (int, float)):
        raw_val = int(raw_val)
        if raw_val in (match_score, mismatch_score):
            return rule_name, raw_val
    
    # 字符串转数字
    if isinstance(raw_val, str):
        try:
            num_val = int(raw_val.strip())
            if num_val in (match_score, mismatch_score):
                return rule_name, num_val
        except:
            pass
    
    # 兜底返回不匹配分
    return rule_name, mismatch_score

def validate_scores_concurrent(pos: str, raw_scores: dict) -> Dict[str, int]:
    """多线程验证词类分数"""
    rules = RULE_SETS[pos]
    scores_out = {}
    
    # 线程池并行验证规则分数
    with ThreadPoolExecutor(max_workers=len(rules)) as executor:
        futures = {
            executor.submit(validate_score_worker, rule, raw_scores.get(rule["name"], rule["mismatch_score"])): rule
            for rule in rules
        }
        for future in as_completed(futures):
            rule_name, score = future.result()
            scores_out[rule_name] = score
    
    # 补全缺失规则
    for rule in rules:
        if rule["name"] not in scores_out:
            scores_out[rule["name"]] = rule["mismatch_score"]
    
    return scores_out

def calculate_membership_concurrent(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """并行计算隶属度"""
    membership = {}
    with ThreadPoolExecutor(max_workers=len(RULE_SETS)) as executor:
        futures = {
            executor.submit(lambda p: sum(scores_all[p].values()) / 100, pos): pos
            for pos in RULE_SETS.keys()
        }
        for future in as_completed(futures):
            pos = futures[future]
            normalized = future.result()
            membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

# ===============================
# 并发API调用函数
# ===============================
def call_llm_api_concurrent(provider: str, model: str, api_key: str, messages: list) -> Tuple[bool, Dict[str, Any], str]:
    """并发安全的API调用（连接池+超时）"""
    if not api_key:
        return False, {"error": "API Key为空"}, "API Key未提供"
    if provider not in MODEL_CONFIGS:
        return False, {"error": f"未知提供商{provider}"}, f"未知提供商{provider}"

    cfg = MODEL_CONFIGS[provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](api_key)
    payload = cfg["payload"](model, messages, max_tokens=4096, temperature=0.0)

    try:
        # 连接池请求（超时120秒）
        response = SESSION.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return True, response.json(), ""
    except requests.exceptions.Timeout:
        return False, {"error": "请求超时"}, "模型响应超时（120秒）"
    except requests.exceptions.RequestException as e:
        err_msg = f"API请求失败: {str(e)[:100]}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_detail = e.response.json().get("error", {}).get("message", "")
                err_msg += f" 详情: {err_detail[:50]}"
            except:
                err_msg += f" 响应: {e.response.text[:50]}"
        return False, {"error": err_msg}, err_msg
    except Exception as e:
        return False, {"error": str(e)}, f"未知错误: {str(e)[:50]}"

def analyze_single_word(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str, str]:
    """单词语分析（并发任务）"""
    # 构建强约束Prompt（直出分数）
    rule_text = []
    for pos, rules in RULE_SETS.items():
        rule_text.append(f"【{pos}】")
        for r in rules:
            rule_text.append(f"- {r['name']}: {r['desc']}（符合填{r['match_score']}分，不符合填{r['mismatch_score']}分）")
    rule_text = "\n".join(rule_text)

    system_msg = f"""你是中文词法专家，分析词语「{word}」的词类隶属度，必须严格输出数字分数！

【规则】
{rule_text}

【输出格式】
1. 推理过程（逐条说明理由+例句+分数）
2. 分隔符包裹的JSON（仅数字分数，无布尔值）：
====JSON_BEGIN====
{{
  "explanation": "推理过程文本",
  "predicted_pos": "名词/动词/名动词",
  "scores": {{
    "名词": {{
      "N1_可受数量词修饰": 10,
      "N2_不能受副词修饰": 20,
      ...
    }},
    "动词": {{...}},
    "名动词": {{...}}
  }}
}}
====JSON_END====

【强制要求】
- JSON中所有值必须是整数（如10、-20、0），不能用true/false/字符串
- 键必须用双引号，数字不加引号
- 分隔符单独成行，不能修改
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"分析词语「{word}」，输出推理过程和规范JSON（所有规则填数字分数）。"}
    ]

    # 调用API
    ok, resp_json, err_msg = call_llm_api_concurrent(provider, model, api_key, messages)
    if not ok:
        st.error(f"词「{word}」调用失败: {err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", err_msg

    # 提取响应
    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)

    # 初始化默认分数
    default_scores = {
        pos: {r["name"]: r["mismatch_score"] for r in RULE_SETS[pos]} 
        for pos in RULE_SETS.keys()
    }

    if not parsed_json or not isinstance(parsed_json, dict):
        return default_scores, raw_text, "未知", "JSON解析失败，使用默认分数"

    # 提取基础信息
    explanation = parsed_json.get("explanation", "无推理过程")
    predicted_pos = parsed_json.get("predicted_pos", "未知")
    raw_scores = parsed_json.get("scores", {})

    # 多线程验证所有词类分数
    scores_out = {}
    with ThreadPoolExecutor(max_workers=len(RULE_SETS)) as executor:
        futures = {
            executor.submit(validate_scores_concurrent, pos, raw_scores.get(pos, {})): pos
            for pos in RULE_SETS.keys()
        }
        for future in as_completed(futures):
            pos = futures[future]
            scores_out[pos] = future.result()

    return scores_out, raw_text, predicted_pos, explanation

def analyze_batch_words(words: list, provider: str, model: str, api_key: str) -> Dict[str, dict]:
    """批量分析词语（最高并发MAX_WORKERS）"""
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(analyze_single_word, word, provider, model, api_key): word
            for word in words if word.strip()
        }
        # 实时更新进度
        progress_bar = st.progress(0)
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            word = futures[future]
            try:
                scores_all, raw_text, predicted_pos, explanation = future.result()
                membership = calculate_membership_concurrent(scores_all)
                results[word] = {
                    "scores_all": scores_all,
                    "raw_text": raw_text,
                    "predicted_pos": predicted_pos,
                    "explanation": explanation,
                    "membership": membership
                }
            except Exception as e:
                results[word] = {"error": f"分析失败: {str(e)[:50]}"}
            # 更新进度
            completed += 1
            progress_bar.progress(completed / total)
        progress_bar.empty()

    return results

# ===============================
# 页面主逻辑
# ===============================
def main():
    st.title("📰 汉语词类隶属度检测划类（并发提速版）")

    # 顶部控制区
    col1, col2, col3 = st.columns([2, 1, 3])
    with col1:
        st.subheader("⚙️ 模型设置")
        selected_model_name = st.selectbox("选择大模型", list(MODEL_OPTIONS.keys()), key="model_select")
        selected_model = MODEL_OPTIONS[selected_model_name]

        # 检查API Key
        if not selected_model["api_key"]:
            st.error(f"❌ 未配置 {selected_model_name} 的API Key")
            st.code(f"# 设置环境变量\n# Linux/Mac: export {selected_model['env_var']}='你的API Key'\n# Windows: set {selected_model['env_var']}='你的API Key'", language="bash")

    with col2:
        st.subheader("🔗 连接测试")
        test_btn = st.button("测试模型链接", disabled=not selected_model["api_key"])
        if test_btn:
            with st.spinner("测试中..."):
                ok, _, err_msg = call_llm_api_concurrent(
                    selected_model["provider"],
                    selected_model["model"],
                    selected_model["api_key"],
                    [{"role": "user", "content": "回复pong"}]
                )
            if ok:
                st.success("✅ 模型链接成功！")
            else:
                st.error(f"❌ 链接失败: {err_msg}")

    with col3:
        st.subheader("🔤 词语输入（支持批量）")
        input_mode = st.radio("输入模式", ["单词语", "批量词语"], horizontal=True)
        if input_mode == "单词语":
            word = st.text_input("输入单个词语", placeholder="例如：苹果、跑、研究", key="single_word")
            analyze_btn = st.button("🚀 开始分析", type="primary", disabled=not (selected_model["api_key"] and word))
        else:
            words_text = st.text_area("批量输入词语（每行一个）", placeholder="苹果\n跑\n研究\n美丽", key="batch_words")
            analyze_btn = st.button("🚀 批量分析", type="primary", disabled=not (selected_model["api_key"] and words_text.strip()))

    st.markdown("---")

    # 使用说明
    with st.expander("ℹ️ 使用说明（并发版）", expanded=False):
        st.info("""
        1. 单词语模式：输入一个词语，点击分析，秒级返回结果（并发提速30%+）；
        2. 批量模式：每行输入一个词语，最多同时分析5个（可调整MAX_WORKERS）；
        3. 模型链接测试：验证API Key有效性，避免分析失败；
        4. 并发优化：API调用/分数计算并行处理，大幅减少等待时间。
        """)

    # 分析逻辑
    if analyze_btn and selected_model["api_key"]:
        start_time = time.time()
        provider = selected_model["provider"]
        model = selected_model["model"]
        api_key = selected_model["api_key"]

        if input_mode == "单词语" and word:
            # 单词语分析（并发分数计算）
            st.info(f"开始分析词语「{word}」（并发模式）...")
            scores_all, raw_text, predicted_pos, explanation = analyze_single_word(word, provider, model, api_key)
            membership = calculate_membership_concurrent(scores_all)

            # 结果展示
            st.success(f"✅ 分析完成（耗时: {time.time()-start_time:.2f}秒）：「{word}」→ 【{predicted_pos}】（隶属度: {membership.get(predicted_pos, 0):.4f}）")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("🏆 隶属度排名")
                top10 = sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]
                st.table(pd.DataFrame(top10, columns=["词类", "隶属度"]).round(4))

                st.subheader("📊 隶属度雷达图")
                categories = [x[0] for x in top10] + [top10[0][0]]
                values = [x[1] for x in top10] + [top10[0][1]]
                fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill="toself"))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), title=f"「{word}」隶属度分布")
                st.plotly_chart(fig, use_container_width=True)

            with col_res2:
                st.subheader("📋 详细得分")
                for pos in RULE_SETS.keys():
                    total = sum(scores_all[pos].values())
                    with st.expander(f"**{pos}** (总分: {total})"):
                        df = pd.DataFrame([
                            {"规则": r["name"], "描述": r["desc"], "得分": scores_all[pos][r["name"]]}
                            for r in RULE_SETS[pos]
                        ])
                        df_styled = df.style.applymap(lambda x: "color: red; font-weight: bold" if isinstance(x, int) and x < 0 else "", subset=["得分"])
                        st.dataframe(df_styled, use_container_width=True)

                st.subheader("🔍 推理过程")
                st.text_area("", value=explanation, height=200)

        else:
            # 批量分析
            words = [w.strip() for w in words_text.split("\n") if w.strip()]
            if not words:
                st.warning("请输入至少一个词语！")
                return

            st.info(f"开始批量分析 {len(words)} 个词语（并发数: {MAX_WORKERS}）...")
            results = analyze_batch_words(words, provider, model, api_key)

            # 批量结果展示
            st.success(f"✅ 批量分析完成（总耗时: {time.time()-start_time:.2f}秒）")
            for word, res in results.items():
                if "error" in res:
                    st.error(f"「{word}」: {res['error']}")
                    continue

                with st.expander(f"📝 词语：{word} → 预测词类：{res['predicted_pos']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        # 隶属度排名
                        top10 = sorted(res["membership"].items(), key=lambda x: x[1], reverse=True)[:10]
                        st.table(pd.DataFrame(top10, columns=["词类", "隶属度"]).round(4))
                    with col2:
                        # 总分概览
                        total_scores = {pos: sum(res["scores_all"][pos].values()) for pos in RULE_SETS.keys()}
                        st.write("### 各词类总分")
                        st.bar_chart(total_scores)

                    # 详细得分
                    st.write("### 详细规则得分")
                    all_scores = []
                    for pos, rules in RULE_SETS.items():
                        for r in rules:
                            all_scores.append({
                                "词类": pos,
                                "规则": r["name"],
                                "描述": r["desc"],
                                "得分": res["scores_all"][pos][r["name"]]
                            })
                    df = pd.DataFrame(all_scores)
                    df_styled = df.style.applymap(lambda x: "color: red; font-weight: bold" if isinstance(x, int) and x < 0 else "", subset=["得分"])
                    st.dataframe(df_styled, use_container_width=True)

if __name__ == "__main__":
    main()

# 页面底部
st.markdown("---")
st.markdown("<div style='text-align:center; color:#666;'>© 2025 汉语词类检测（并发提速版）</div>", unsafe_allow_html=True)

import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
import time  # 用于降速和重试
from typing import Tuple, Dict, Any, List
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测划类",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 自定义CSS样式
hide_streamlit_style = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.dataframe {font-size: 12px;}
[data-testid="stSidebar"] { display: none !important; }
.stApp > div:first-child { padding-top: 2rem; }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "X-DashScope-SSE": "enable", "Accept": "text/event-stream"
        },
        "payload": lambda model, messages, **kw: {
            "model": model, "input": {"messages": messages}, 
            "parameters": {"max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0),
                           "result_format": "message", "incremental_output": True},
        },
    },
}

MODEL_OPTIONS = {
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o（推荐）": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Moonshot（Kimi）": {"provider": "moonshot", "model": "moonshot-v1-32k", "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"},
    "Qwen（通义千问）": {"provider": "qwen", "model": "qwen-max", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS: AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 规则定义
# ===============================
RULE_SETS = {
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
    "动词": [
        {"name": "V1_可受否定'不/没有'修饰", "desc": "可以受否定副词'不'或'没有'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "V2_可后附/插入时体助词'着/了/过'", "desc": "可以后附或中间插入时体助词'着/了/过'，或进入'...了没有'格式", "match_score": 10, "mismatch_score": 0},
        {"name": "V3_可带真宾语或通过介词引导论元", "desc": "可以带真宾语，或通过'和/为/对/向/拿/于'等介词引导论元", "match_score": 20, "mismatch_score": 0},
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "V5_可有重叠/正反重叠形式", "desc": "可以有'VV, V一V, V了V, V不V, V了没有'等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "V6_可做谓语或谓语核心", "desc": "可以做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "V7_不能作状语修饰动词性成分", "desc": "不能作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "V8_可作'怎么/怎样'提问或'这么/这样/那么'回答", "desc": "可以跟在'怎么/怎样'之后提问或跟在'这么/这样/那么'之后回答", "match_score": 10, "mismatch_score": 0},
        {"name": "V9_不能跟在'多/多么'之后提问或表示感叹", "desc": "不能跟在'多'之后对性质提问，不能跟在'多么'之后表示感叹", "match_score": 10, "mismatch_score": -10},
    ],
    "名动词": [
        {"name": "NV1_可被\"不/没有\"否定且肯定形式-1", "desc": "可以用\"不\"和\"没有\"来否定，并且\"没有……\"的肯定形式可以是\"……了\"和\"有……\"", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"，或者可以进入\"………了没有\"格式", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "可以有\"VV、V一V、V了V、V不V\"等重叠和正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语或谓语核心，又可以作主语或宾语，且可作形式动词宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后提问，跟在\"这么、这样\"之后回答", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多\"之后，对性质的程度进行提问，也不能跟在\"多么\"之后，表示感叹", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
    ]
}

# ===============================
# 工具函数 (已增强)
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
    """
    【增强版】优先寻找Markdown代码块，找不到再找大括号。
    解决模型输出废话导致提取失败的问题。
    """
    if not text: return None, ""
    
    json_str = ""
    # 1. 尝试提取 ```json ... ```
    code_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
    
    # 2. 如果失败，尝试提取最外层 {...}
    if not json_str:
        match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
        if match: json_str = match.group(1).strip()

    if not json_str: return None, text

    try: 
        return json.loads(json_str), json_str
    except json.JSONDecodeError: 
        return None, text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_norm = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        r_norm = re.sub(r'[\s_]+', '', r["name"]).upper()
        if r_norm == k_norm: return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    if isinstance(raw_val, bool): return match_score if raw_val is True else mismatch_score
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"): return match_score
        if s in ("no", "n", "false", "否", "×", "不符合"): return mismatch_score
    if isinstance(raw_val, (int, float)):
        raw_val_int = int(raw_val)
        if raw_val_int == match_score: return match_score
        if raw_val_int == mismatch_score: return mismatch_score
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        normalized = total_score / 100
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

# ===============================
# API调用
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    if _provider not in MODEL_CONFIGS: return False, {"error": f"未知提供商 {_provider}"}, f"未知提供商 {_provider}"

    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    streaming_placeholder = st.empty()
    full_content = ""

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line: continue
                line_text = line.decode('utf-8').strip()
                if line_text.startswith("data:"): json_str = line_text[5:].strip()
                else: json_str = line_text
                if json_str == "[DONE]": break
                try:
                    chunk = json.loads(json_str)
                    delta_text = ""
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        delta_text = delta.get("content", "")
                    elif "output" in chunk:
                        output = chunk["output"]
                        if "choices" in output and len(output["choices"]) > 0:
                             msg = output["choices"][0].get("message", {})
                             delta_text = msg.get("content", "")
                        elif "text" in output:
                             delta_text = output["text"]
                    if delta_text: full_content += delta_text
                except json.JSONDecodeError: continue
        streaming_placeholder.empty()
        mock_response = {"choices": [{"message": {"content": full_content}}], "output": {"text": full_content}}
        if not full_content: return False, {"error": "无内容"}, "无内容"
        return True, mock_response, ""
    except Exception as e:
        return False, {"error": str(e)}, str(e)

# ===============================
# 核心分析
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict, str, str, str]:
    if not word: return {}, "", "未知", ""

    full_rules_by_pos = {
        pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules])
        for pos, rules in RULE_SETS.items()
    }

    system_msg = f"""你是一名中文词法专家。分析「{word}」在名词、动词、名动词的表现。
规则：{full_rules_by_pos["名词"]}\n{full_rules_by_pos["动词"]}\n{full_rules_by_pos["名动词"]}
要求：
1. explanation: 逐条规则说明理由。
2. scores: 每条规则 true/false。
3. predicted_pos: 选一个。
4. 返回 JSON。"""

    user_prompt = f"分析词语「{word}」，请直接返回包含 explanation, predicted_pos, scores 的 JSON。"

    with st.spinner(f"正在分析..."):
        ok, resp_json, err_msg = call_llm_api_cached(provider, model, api_key, [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}])

    if not ok: return {}, f"调用失败: {err_msg}", "未知", f"失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)

    # 兜底逻辑：如果JSON解析失败，至少保留原始文本作为推理过程
    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "无")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
    else:
        # 如果解析失败，把原始回复当作推理过程
        explanation = "解析失败，原始回复：" + raw_text
        predicted_pos = "未知"
        raw_scores = {}

    scores_out = {pos: {} for pos in RULE_SETS.keys()}
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
            if rule["name"] not in scores_out[pos]: scores_out[pos][rule["name"]] = 0

    return scores_out, raw_text, predicted_pos, explanation

def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm: return
    categories = list(scores_norm.keys())
    values = list(scores_norm.values())
    categories += [categories[0]]
    values += [values[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=False, title=dict(text=title, x=0.5))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 【增强版】Excel 批量处理（实时保存 + 兜底显示）
# ===============================
def process_and_style_excel(df, selected_model_info, target_col_name):
    """
    修改点：
    1. 实时追加写入 'history_database.csv'，防丢失。
    2. 读取 'history_database.csv' 实现断点续传。
    3. 如果 JSON 解析失败，强制填入 raw_text 作为推理过程。
    """
    output = io.BytesIO()
    processed_rows = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(df)
    
    # 1. 定义本地数据库
    db_file = "history_database.csv"
    
    # 2. 读取已有历史（用于跳过）
    existing_data = {}
    if os.path.exists(db_file):
        try:
            # 读取时强制将'词语'列转为str，防止匹配错误
            history_df = pd.read_csv(db_file)
            history_df['词语'] = history_df['词语'].astype(str).str.strip()
            for _, row in history_df.iterrows():
                existing_data[row['词语']] = row.to_dict()
            st.info(f"📚 已加载 {len(existing_data)} 条本地历史记录，将自动跳过这些词。")
        except:
            pass

    try:
        for index, row in df.iterrows():
            word = str(row[target_col_name]).strip()
            
            # 3. 检查历史缓存
            if word in existing_data:
                status_text.text(f"♻️ 使用缓存 ({index + 1}/{total}): {word}")
                processed_rows.append(existing_data[word])
                time.sleep(0.01)
            else:
                # 跑新词
                max_retries = 3
                success = False
                scores_all = {}
                raw_text = ""
                predicted_pos = "请求失败"
                explanation = "重试失败"
                
                for attempt in range(max_retries):
                    try:
                        status_text.text(f"🚀 分析新词 ({index + 1}/{total}): {word} ...")
                        scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(
                            word=word,
                            provider=selected_model_info["provider"],
                            model=selected_model_info["model"],
                            api_key=selected_model_info["api_key"]
                        )
                        # 只要 raw_text 有内容就算通信成功，哪怕解析失败
                        if raw_text:
                            success = True
                            break 
                        else:
                            time.sleep(2)
                    except Exception as e:
                        time.sleep(2)
                
                # 计算分数
                if success and scores_all:
                    membership = calculate_membership(scores_all)
                    score_v = membership.get("动词", 0.0)
                    score_n = membership.get("名词", 0.0)
                    score_nv = membership.get("名动词", 0.0)
                else:
                    score_v, score_n, score_nv = 0.0, 0.0, 0.0
                
                diff_val = round(abs(score_v - score_n), 4)
                
                # 关键：如果 explanation 为空（解析失败），则使用 raw_text
                final_explanation = explanation if (explanation and len(explanation) > 5) else raw_text
                
                new_row = {
                    "词语": word,
                    "动词": score_v,
                    "名词": score_n,
                    "名动词": score_nv,
                    "差值/距离": diff_val,
                    "原始响应": final_explanation if success else "错误",
                    "_predicted_pos": predicted_pos
                }
                
                processed_rows.append(new_row)
                
                # 4. 实时写入数据库 (追加模式)
                try:
                    temp_df = pd.DataFrame([new_row])
                    write_header = not os.path.exists(db_file)
                    # utf-8-sig 防止中文乱码
                    temp_df.to_csv(db_file, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                except:
                    pass
                
                time.sleep(1) # 降速

            progress_bar.progress((index + 1) / total)

    except Exception as e:
        st.error(f"意外中断: {e}")

    # 生成 Excel
    if not processed_rows: return None
    result_df = pd.DataFrame(processed_rows)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        cols = ["词语", "动词", "名词", "名动词", "差值/距离", "原始响应"]
        # 确保列都存在
        valid_cols = [c for c in cols if c in result_df.columns]
        result_df[valid_cols].to_excel(writer, index=False, sheet_name='分析结果')
        
        # 标黄
        try:
            ws = writer.sheets['分析结果']
            fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            for i, r in enumerate(processed_rows):
                if "_predicted_pos" not in r: continue
                pred = r["_predicted_pos"]
                target = None
                if pred == "动词": target = 2
                elif pred == "名词": target = 3
                elif pred == "名动词": target = 4
                if target: ws.cell(row=i+2, column=target).fill = fill
        except:
            pass

    status_text.success(f"✅ 完成！")
    return output.getvalue()

# ===============================
# 主程序
# ===============================
def main():
    st.title("📰 汉语词类隶属度检测划类")
    
    control_container = st.container()
    with control_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 无 API Key")
                selected_model_info = {"api_key": None}
            else:
                s_name = st.selectbox("选择模型", list(AVAILABLE_MODEL_OPTIONS.keys()))
                selected_model_info = AVAILABLE_MODEL_OPTIONS[s_name]
        with col2:
            st.write("")
            if st.button("测试连接"):
                ok, _, msg = call_llm_api_cached(selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"], [{"role":"user","content":"hi"}], max_tokens=5)
                if ok: st.success("成功")
                else: st.error(msg)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔍 单个词语详细分析", "📂 Excel 批量处理"])
    
    # Tab 1: 单个分析
    with tab1:
        word = st.text_input("词语输入", key="word_input")
        if st.button("开始分析", disabled=not (word and selected_model_info["api_key"])):
            scores, raw, pred, expl = ask_model_for_pos_and_scores(word, selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"])
            if scores:
                mem = calculate_membership(scores)
                st.success(f"结果: {pred} ({mem.get(pred,0):.2f})")
                c1, c2 = st.columns(2)
                with c1:
                    st.table(pd.DataFrame(get_top_10_positions(mem), columns=["词类","隶属度"]))
                    plot_radar_chart_streamlit(mem, f"{word} 雷达图")
                with c2:
                    st.subheader("得分详情")
                    # 显示推理过程
                    st.info(expl)
                    with st.expander("原始响应"): st.text(raw)

    # Tab 2: 批量处理
    with tab2:
        st.header("Excel 批量处理")
        up_file = st.file_uploader("上传 Excel", type=["xlsx"])
        
        if up_file and selected_model_info["api_key"]:
            df = pd.read_excel(up_file)
            target = next((c for c in df.columns if "词" in str(c) or "word" in str(c).lower()), None)
            if target:
                st.dataframe(df.head(3))
                if st.button("🚀 开始批量"):
                    res = process_and_style_excel(df, selected_model_info, target)
                    if res:
                        st.download_button("📥 下载结果 (Excel)", res, "result.xlsx")
            else:
                st.error("未找到'词'列")

        st.markdown("---")
        
        # --- 历史记录管理区域 ---
        st.subheader("📚 历史记录数据库管理")
        db_file = "history_database.csv"
        
        if os.path.exists(db_file):
            try:
                # 读取时全部转为字符串，确保格式一致
                history_df = pd.read_csv(db_file)
                st.info(f"本地数据库中共有 {len(history_df)} 条已分析记录。")
                
                with st.expander("查看历史数据预览"):
                    st.dataframe(history_df)
                
                col_h1, col_h2 = st.columns([1, 1])
                with col_h1:
                    # 提供CSV下载（最原始的数据备份）
                    st.download_button(
                        label="💾 下载完整历史记录 (CSV)",
                        data=history_df.to_csv(index=False).encode('utf-8-sig'),
                        file_name="history_database.csv",
                        mime="text/csv"
                    )
                with col_h2:
                    if st.button("🗑️ 清空历史记录 (慎点)"):
                        os.remove(db_file)
                        st.success("历史记录已清空！正在刷新...")
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.error(f"读取历史记录出错: {e}")
        else:
            st.warning("暂无历史记录。")

if __name__ == "__main__":
    main()

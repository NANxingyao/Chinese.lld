import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
from typing import Tuple, Dict, Any, List
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测划类 (专业版)",
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
# 模型配置 (启用流式 Streaming 以解决超时和Status 0问题)
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), 
            "stream": True, 
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), 
            "stream": True,
        },
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), 
            "stream": True,
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        # Qwen Native API 需要 Accept: text/event-stream 来触发 SSE
        "headers": lambda key: {
            "Authorization": f"Bearer {key}", 
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable",
            "Accept": "text/event-stream"
        },
        "payload": lambda model, messages, **kw: {
            "model": model, 
            "input": {"messages": messages}, 
            "parameters": {
                "max_tokens": kw.get("max_tokens", 4096), 
                "temperature": kw.get("temperature", 0.0),
                "result_format": "message",
                "incremental_output": True 
            },
        },
    },
}

MODEL_OPTIONS = {
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o-mini (推荐)": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Moonshot (Kimi)": {"provider": "moonshot", "model": "moonshot-v1-32k", "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"},
    "Qwen Turbo (速度快)": {"provider": "qwen", "model": "qwen-turbo", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
    "Qwen Max": {"provider": "qwen", "model": "qwen-max", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
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
        {"name": "NV1_可被\"不/没有\"否定且肯定形式-1", "desc": "可以用\"不\"和\"没有\"来否定，肯定形式可以是\"……了\"和\"有……\"", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"，或者可以进入\"………了没有\"格式", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "有重叠和正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语，又可以作主语或宾语，且可作形式动词宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后提问，跟在\"这么\"之后回答", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多/多么\"之后", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
    ]
}

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict): return ""
    try:
        # Qwen / OpenAI 兼容处理
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    if not match: return None, text
    json_text = match.group(1).strip()
    try:
        parsed_json = json.loads(json_text)
        return parsed_json, json_text
    except json.JSONDecodeError:
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
    if isinstance(raw_val, bool): return match_score if raw_val is True else mismatch_score
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"): return match_score
        if s in ("no", "n", "false", "否", "×", "不符合"): return mismatch_score
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
# API 调用函数 (流式处理)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    if _provider not in MODEL_CONFIGS: return False, {"error": f"未知提供商 {_provider}"}, f"未知提供商 {_provider}"

    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    
    full_content = ""
    # 仅在单次分析模式下显示流式过程，防止批量处理时UI混乱（可选）
    # streaming_placeholder = st.empty() 

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
                    elif "output" in chunk: # Qwen
                        output = chunk["output"]
                        if "choices" in output and len(output["choices"]) > 0:
                             msg = output["choices"][0].get("message", {})
                             delta_text = msg.get("content", "")
                        elif "text" in output:
                             delta_text = output["text"]
                    
                    if delta_text:
                        full_content += delta_text
                        # streaming_placeholder.markdown(full_content + "▌") 
                except json.JSONDecodeError: continue
        
        # streaming_placeholder.empty()
        mock_response = {"choices": [{"message": {"content": full_content}}], "output": {"text": full_content}}
        if not full_content: return False, {"error": "无内容"}, "无内容"
        return True, mock_response, ""

    except Exception as e:
        return False, {"error": str(e)}, str(e)

# ===============================
# 核心分析逻辑 (Prompt 已优化提速)
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str, str]:
    if not word: return {}, "", "未知", ""

    # 简化的 Prompt，不再要求造句，大幅提速
    full_rules_by_pos = {
        pos: "\n".join([f"- {r['name']}: {r['desc']}" for r in rules])
        for pos, rules in RULE_SETS.items()
    }

    system_msg = f"""你是一名汉语语言学专家。分析词语「{word}」。
任务：判断该词是否符合名词、动词、名动词的各项规则。
要求：
1. 直接输出 JSON 格式结果。
2. scores 字段中，必须对每一条规则返回 true 或 false。
3. explanation 字段请用一句话简要概括理由，不需要详细造句。
4. predicted_pos 请在'名词','动词','名动词'中选一个。

规则参考：
【名词】
{full_rules_by_pos["名词"]}
【动词】
{full_rules_by_pos["动词"]}
【名动词】
{full_rules_by_pos["名动词"]}
"""
    user_prompt = f"请分析「{word}」并返回 JSON。"

    ok, resp_json, err_msg = call_llm_api_cached(provider, model, api_key, [{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}])

    if not ok: return {}, f"调用失败: {err_msg}", "未知", f"Fail: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, _ = extract_json_from_text(raw_text)

    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "无")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
    else:
        return {}, raw_text, "未知", "JSON解析失败"

    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)
        # 补全缺失规则
        for rule in rules:
            if rule["name"] not in scores_out[pos]: scores_out[pos][rule["name"]] = 0

    return scores_out, raw_text, predicted_pos, explanation

# ===============================
# Excel 处理与标黄逻辑
# ===============================
def process_and_style_excel(df, selected_model_info, target_col_name):
    output = io.BytesIO()
    processed_rows = []
    
    # 进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df)

    for index, row in df.iterrows():
        word = str(row[target_col_name]).strip()
        status_text.text(f"正在处理 ({index + 1}/{total}): {word}")
        
        # 1. 调用模型
        scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(
            word=word,
            provider=selected_model_info["provider"],
            model=selected_model_info["model"],
            api_key=selected_model_info["api_key"]
        )
        
        # 2. 计算隶属度
        membership = calculate_membership(scores_all) if scores_all else {}
        score_v = membership.get("动词", 0.0)
        score_n = membership.get("名词", 0.0)
        score_nv = membership.get("名动词", 0.0)
        
        # 3. 计算差值 (动词 - 名词 的绝对值)
        diff_val = round(abs(score_v - score_n), 4)
        
        # 4. 构造数据行 (顺序：词语, 动词, 名词, 名动词, 差值/距离, 原始响应)
        new_row = {
            "词语": word,
            "动词": score_v,
            "名词": score_n,
            "名动词": score_nv,
            "差值/距离": diff_val,
            "原始响应": raw_text,
            "_predicted_pos": predicted_pos # 隐藏辅助列
        }
        processed_rows.append(new_row)
        progress_bar.progress((index + 1) / total)

    result_df = pd.DataFrame(processed_rows)
    
    # 使用 openpyxl 导出并标黄
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        cols = ["词语", "动词", "名词", "名动词", "差值/距离", "原始响应"]
        result_df[cols].to_excel(writer, index=False, sheet_name='分析结果')
        
        worksheet = writer.sheets['分析结果']
        yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        
        for i, data_row in enumerate(processed_rows):
            row_num = i + 2 # Header is row 1
            pred = data_row["_predicted_pos"]
            
            # A=1(词), B=2(动), C=3(名), D=4(名动)
            target_idx = None
            if pred == "动词": target_idx = 2
            elif pred == "名词": target_idx = 3
            elif pred == "名动词": target_idx = 4
            
            if target_idx:
                worksheet.cell(row=row_num, column=target_idx).fill = yellow_fill

    status_text.success("✅ 处理完成！已自动标黄获胜词类。")
    return output.getvalue()

# ===============================
# UI 模块
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm: return
    categories = list(scores_norm.keys())
    values = list(scores_norm.values())
    categories += [categories[0]]
    values += [values[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=False, title=dict(text=title, x=0.5))
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📰 汉语词类隶属度检测 ")
    
    # 顶部设置栏
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 找不到 API Key，请设置环境变量 (如 QWEN_API_KEY)！")
                selected_model_info = {"api_key": None}
            else:
                s_name = st.selectbox("选择模型", list(AVAILABLE_MODEL_OPTIONS.keys()))
                selected_model_info = AVAILABLE_MODEL_OPTIONS[s_name]
        with col2:
            st.write("") # Spacer
            if st.button("测试连接"):
                ok, _, msg = call_llm_api_cached(selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"], [{"role":"user","content":"hi"}], max_tokens=5)
                if ok: st.success("连接成功")
                else: st.error(f"连接失败: {msg}")

    st.markdown("---")
    
    # 分页签：单词测试 vs 批量处理
    tab1, tab2 = st.tabs(["🔍 单个词语详细分析", "📂 Excel 批量处理"])
    
    # --- Tab 1: 单词分析 ---
    with tab1:
        word = st.text_input("输入词语", placeholder="例如：发展")
        if st.button("开始分析", type="primary", disabled=not (word and selected_model_info["api_key"])):
            with st.spinner("分析中..."):
                scores, raw, pred, expl = ask_model_for_pos_and_scores(word, selected_model_info["provider"], selected_model_info["model"], selected_model_info["api_key"])
                
                if scores:
                    mem = calculate_membership(scores)
                    st.success(f"最可能词类：**{pred}** (隶属度: {mem.get(pred,0):.2f})")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        top10_df = pd.DataFrame(get_top_10_positions(mem), columns=["词类", "隶属度"])
                        st.table(top10_df)
                        plot_radar_chart_streamlit(mem, f"{word} 隶属度雷达图")
                    with c2:
                        st.subheader("推理简述")
                        st.info(expl)
                        with st.expander("查看原始响应"): st.code(raw)

    # --- Tab 2: 批量处理 ---
    with tab2:
        st.info("上传 Excel 文件，必须包含表头为 **'词语'** 的列。系统将生成包含 **[动词|名词|名动词|差值]** 的结果表，并自动标黄。")
        uploaded_file = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
        
        if uploaded_file and selected_model_info["api_key"]:
            try:
                df = pd.read_excel(uploaded_file)
                # 寻找目标列
                target_col = next((c for c in df.columns if "词" in str(c) or "word" in str(c).lower()), None)
                
                if target_col:
                    st.write(f"✅ 识别到目标列：`{target_col}`，共 {len(df)} 个词。")
                    if st.button("🚀 开始批量分析"):
                        excel_data = process_and_style_excel(df, selected_model_info, target_col)
                        st.download_button("📥 下载结果 (已标黄)", excel_data, file_name="分析结果.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("❌ 未找到包含 '词' 的列名，请修改 Excel 表头。")
            except Exception as e:
                st.error(f"文件读取错误: {e}")

if __name__ == "__main__":
    main()


import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
import time
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
[data-testid="stSidebar"] {display: none !important;}
.stApp > div:first-child {padding-top: 2rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 词类规则定义（全局，修复核心：提取到全局避免复杂读取）
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
# 模型配置 (启用流式 Stream)
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

# 模型选项（仅从环境变量获取API Key）
MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek", 
        "model": "deepseek-chat", 
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "env_var": "DEEPSEEK_API_KEY"
    },
    "OpenAI GPT-4o（推荐）": {
        "provider": "openai", 
        "model": "gpt-4o-mini", 
        "api_key": os.getenv("OPENAI_API_KEY"),
        "env_var": "OPENAI_API_KEY"
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot", 
        "model": "moonshot-v1-32k", 
        "api_key": os.getenv("MOONSHOT_API_KEY"),
        "env_var": "MOONSHOT_API_KEY"
    },
    "Qwen（通义千问）": {
        "provider": "qwen", 
        "model": "qwen-max", 
        "api_key": os.getenv("QWEN_API_KEY"),
        "env_var": "QWEN_API_KEY"
    },
}

# 过滤掉没有配置 API Key 的模型
AVAILABLE_MODEL_OPTIONS = {
    name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]
}

if not AVAILABLE_MODEL_OPTIONS:
    AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """从不同格式的LLM响应中安全提取文本内容。"""
    if not isinstance(resp_json, dict):
        return ""
    try:
        # Qwen 格式
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        
        # OpenAI/DeepSeek/Moonshot 格式
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """从混合文本中提取并解析JSON对象。"""
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    if not match:
        return None, text

    json_text = match.group(1).strip()
    try:
        parsed_json = json.loads(json_text)
        return parsed_json, json_text
    except json.JSONDecodeError:
        return None, json_text

def normalize_key(k: str, pos_rules: list) -> str:
    """标准化模型返回的规则名称"""
    if not isinstance(k, str): return None
    k_norm = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        r_norm = re.sub(r'[\s_]+', '', r["name"]).upper()
        if r_norm == k_norm:
            return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    """将模型返回值映射为规则得分"""
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    if isinstance(raw_val, bool):
        return match_score if raw_val else mismatch_score
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"):
            return match_score
        if s in ("no", "n", "false", "否", "×", "不符合"):
            return mismatch_score
    if isinstance(raw_val, (int, float)):
        raw_val_int = int(raw_val)
        if raw_val_int == match_score: return match_score
        if raw_val_int == mismatch_score: return mismatch_score
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """计算隶属度"""
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        normalized = total_score / 100
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    """获取隶属度最高的前 10 个词类"""
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

def get_history_count(backup_file):
    """获取最新的历史记录数量（实时更新用）"""
    if not os.path.exists(backup_file):
        return 0
    try:
        temp_history = pd.read_csv(backup_file, encoding='utf-8-sig')
        return len(temp_history)
    except Exception as e:
        st.warning(f"读取历史记录数量失败: {e}")
        return 0

# ===============================
# 安全的 LLM 调用函数 (流式版)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    """封装LLM调用逻辑"""
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
                json_str = line_text[5:].strip() if line_text.startswith("data:") else line_text
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
                    if delta_text:
                        full_content += delta_text
                except json.JSONDecodeError:
                    continue
        
        streaming_placeholder.empty()
        mock_response = {
            "choices": [{"message": {"content": full_content}}],
            "output": {"text": full_content}
        }
        
        if not full_content:
             return False, {"error": "未接收到有效内容"}, "模型未返回内容"

        return True, mock_response, ""

    except requests.exceptions.RequestException as e:
        error_msg = f"网络请求异常: {str(e)}"
        return False, {"error": error_msg}, error_msg
    except Exception as e:
        error_msg = f"流式处理未知错误: {str(e)}\n已接收内容: {full_content[:100]}..."
        return False, {"error": error_msg}, error_msg

# ===============================
# 词类判定主函数
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str, str]:
    """词类判定核心函数"""
    if not word:
        return {}, "", "未知", ""

    # 构建规则说明文本（使用全局RULE_SETS）
    full_rules_by_pos = {
        pos: "\n".join([f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）" for r in rules])
        for pos, rules in RULE_SETS.items()
    }

    # 系统提示词
    system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：
- 需要判断的词类：名词、动词、名动词
- 评分规则已经由系统定义，你**不要**自己设计分值，也**不要**在 JSON 中给出具体数字分数。程序将根据你的判断（true/false）自动赋值。
- 你只需要判断每一条规则是“符合”还是“不符合”。

【各词类的规则说明（仅供你判断使用）】
【名词】
{full_rules_by_pos["名词"]}

【动词】
{full_rules_by_pos["动词"]}

【名动词】
{full_rules_by_pos["名动词"]}

【输出要求】
1. 在 explanation 字段中，必须**逐条规则**说明判断依据，并举例（可以自己造句）：
   - 格式示例：
     - 「名词-N1_可受数量词修饰：符合。理由：……。例句：……。」
     - 「动词-V2_可后附/插入时体助词'着/了/过'：不符合。理由：……。例句：……。」
   - explanation 里要覆盖 **三个词类的所有规则**，不能只写几条。

2. 在 JSON 中的 scores 字段里：
   - 每一类下的每一条规则，只能给出 **布尔值 true / false**，表示是否符合该规则
   - 严禁在 scores 里使用数值分数（例如 0, 5, 10 等）
   - 如果你不确定，也必须做出判断（true 或 false），不要用 null、0 或其它值
   - JSON 结构必须是：{{"explanation": "...", "predicted_pos": "...", "scores": {{"名词": {{...}}, "动词": {{...}}, "名动词": {{...}}}}}}

3. predicted_pos：
   - 请选择「名词」「动词」「名动词」之一，作为该词语最典型的词类。

4. **最后输出时，先写详细的文字推理，最后单独且完整地给出一段合法的 JSON（不要再加注释）。**
"""

    user_prompt = f"""
请严格按照上述要求分析词语「{word}」。
特别注意：
- 在 JSON 的 scores 部分，只能用 true/false 表示“是否符合规则”，不能使用任何数字。
- explanation 中必须对每一条规则写明“符合/不符合 + 理由 + 例句”。

请先给出详细推理过程，然后在最后单独输出一个 JSON 对象。
"""

    with st.spinner(f"正在调用大模型 ({model}) 进行分析，请稍候..."):
        ok, resp_json, err_msg = call_llm_api_cached(
            _provider=provider,
            _model=model,
            _api_key=api_key,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
        )

    if not ok:
        st.error(f"模型调用失败: {err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", f"模型调用失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)

    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
        if predicted_pos not in RULE_SETS:
             st.warning(f"模型预测的词类 '{predicted_pos}' 不在分析范围内 ('名词', '动词', '名动词')。")
    else:
        st.error("❌ 未能从模型响应中解析出有效的JSON。请检查模型输出是否符合要求。")
        explanation = "无法解析模型输出。原始响应：\n" + raw_text
        predicted_pos = "未知"
        raw_scores = {}
        cleaned_json_text = raw_text

    # 初始化得分字典
    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)

    # 补全缺失的规则得分（默认0分）
    for pos, rules in RULE_SETS.items():
        for rule in rules:
            rule_name = rule["name"]
            if rule_name not in scores_out[pos]:
                scores_out[pos][rule_name] = 0

    return scores_out, raw_text, predicted_pos, explanation

# ===============================
# 雷达图绘制函数
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    """绘制词类隶属度雷达图"""
    if not scores_norm:
        st.warning("无法绘制雷达图：没有有效数据。")
        return
    
    categories = list(scores_norm.keys())
    if not categories:
        st.warning("无法绘制雷达图：没有有效词类。")
        return
        
    values = list(scores_norm.values())
    categories += [categories[0]]
    values += [values[0]]
    
    min_val = min(values)
    max_val = max(values)
    axis_min = min(min_val, -0.1) 
    axis_max = max(max_val, 1.0)
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values, 
            theta=categories, 
            fill="toself", 
            name="隶属度",
            hovertemplate = '<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>'
        )
    ])
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[axis_min, axis_max],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0] if axis_min >= 0 else [-1.0, -0.5, 0, 0.5, 1.0]
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 批量处理函数
# ===============================
def process_and_style_excel(df, selected_model_info, target_col_name, metric_placeholder, BACKUP_FILE):
    """批量处理Excel并实时更新数据量"""
    output = io.BytesIO()
    if 'processed_history' not in st.session_state:
        st.session_state.processed_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    backup_info_placeholder = st.container()
    total = len(df)
    backup_file = BACKUP_FILE

    try:
        for index, row in df.iterrows():
            word = str(row[target_col_name]).strip()
            max_retries = 3
            success = False
            scores_all, raw_text, predicted_pos, explanation = {}, "", "请求失败", ""
            
            for attempt in range(max_retries):
                try:
                    status_text.text(f"正在处理 ({index + 1}/{total}): {word} ... (尝试 {attempt + 1})")
                    scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(
                        word=word,
                        provider=selected_model_info["provider"],
                        model=selected_model_info["model"],
                        api_key=selected_model_info["api_key"]
                    )
                    if scores_all:
                        success = True
                        break
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
            
            # 构造数据行
            membership = calculate_membership(scores_all) if success else {}
            new_row = {
                "序数": index + 1,
                "词语": word,
                "动词": membership.get("动词", 0.0),
                "名词": membership.get("名词", 0.0),
                "名动词": membership.get("名动词", 0.0),
                "差值/距离": round(abs(membership.get("动词", 0.0) - membership.get("名词", 0.0)), 4),
                "预测词类": predicted_pos,
                "原始响应": raw_text if success else f"错误: {explanation}",
                "时间戳": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存到SessionState
            st.session_state.processed_history.append(new_row)
            
            # 写入CSV并实时更新数据量
            try:
                temp_df = pd.DataFrame([new_row])
                header_needed = not os.path.exists(backup_file)
                temp_df.to_csv(backup_file, mode='a', header=header_needed, index=False, encoding='utf-8-sig')
                # 核心：更新已存数据量指标
                latest_count = get_history_count(backup_file)
                metric_placeholder.metric("已存数据量", f"{latest_count} 条")
            except Exception as csv_err:
                st.error(f"保存第 {index+1} 条记录失败: {csv_err}")

            with backup_info_placeholder:
                st.info(f"💾 已自动保存第 {index+1} 条记录。如遇中断，请检查目录下 `{backup_file}`")

            progress_bar.progress((index + 1) / total)
            time.sleep(0.5)

    except Exception as e:
        st.error(f"⚠️ 批量处理意外中断: {e}")
    
    # 导出Excel
    final_data = st.session_state.processed_history
    if not final_data:
        return None

    result_df = pd.DataFrame(final_data)
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            cols = ["词语", "动词", "名词", "名动词", "差值/距离", "预测词类", "原始响应"]
            result_df[cols].to_excel(writer, index=False, sheet_name='分析结果')
            
            workbook = writer.book
            worksheet = writer.sheets['分析结果']
            yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            
            for i, data_row in enumerate(final_data):
                row_num = i + 2
                pred = data_row["预测词类"]
                target_idx = {"动词": 2, "名词": 3, "名动词": 4}.get(pred)
                if target_idx:
                    worksheet.cell(row=row_num, column=target_idx).fill = yellow_fill
                    
        return output.getvalue()
    except Exception as e:
        st.error(f"Excel 生成失败: {e}")
        return None

# ===============================
# 主页面逻辑
# ===============================
def main():
    st.title("📰 汉语词类隶属度检测划类")
    
    # 顶部控制区
    control_container = st.container()
    with control_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("⚙️ 模型设置")
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 找不到可用的 API Key！请设置以下任意一个环境变量来启用模型:")
                for name, info in MODEL_OPTIONS.items():
                      st.code(f"export {info['env_var']}='你的API Key'", language="bash")
                selected_model_display_name = list(MODEL_OPTIONS.keys())[0]
                selected_model_info = MODEL_OPTIONS[selected_model_display_name]
                st.selectbox("选择大模型 (不可用)", list(MODEL_OPTIONS.keys()), disabled=True)
            else:
                selected_model_display_name = st.selectbox(
                    "选择大模型", 
                    list(AVAILABLE_MODEL_OPTIONS.keys()), 
                    key="model_select"
                )
                selected_model_info = AVAILABLE_MODEL_OPTIONS[selected_model_display_name]
                
        with col2:
            st.subheader("🔗 连接测试")
            st.write("")
            if not selected_model_info["api_key"]:
                st.button("测试模型链接 (不可用)", type="secondary", disabled=True)
            else:
                if st.button("测试模型链接", type="secondary"):
                    with st.spinner("正在测试连接..."):
                        ok, _, err_msg = call_llm_api_cached(
                            _provider=selected_model_info["provider"],
                            _model=selected_model_info["model"],
                            _api_key=selected_model_info["api_key"],
                            messages=[{"role": "user", "content": "请回复'pong'"}],
                            max_tokens=10
                        )
                    if ok:
                        st.success("✅ 成功！")
                    else:
                        st.error(f"❌ 失败: {err_msg}")

    st.markdown("---")

    # 分页
    tab1, tab2 = st.tabs(["🔍 单个词语详细分析", "📂 Excel 批量处理"])

    # 单个词语分析
    with tab1:
        st.subheader("🔤 词语输入")
        word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
        analyze_button = st.button(
            "🚀 开始分析", 
            type="primary",
            disabled=not (selected_model_info["api_key"] and word)
        )

        with st.expander("ℹ️ 使用说明", expanded=False):
            st.info("""
            1. **配置 API Key**: 请在运行程序前设置必要的环境变量。
            2. **词语输入**：在上方的“词语输入”框中输入一个汉语词。
            3. **开始分析**：点击“开始分析”按钮。
            4. **结果解析**：系统将显示隶属度、雷达图和详细规则得分。
            """)

        if analyze_button and word and selected_model_info["api_key"]:
            status_placeholder = st.empty()
            status_placeholder.info(f"正在为词语「{word}」启动分析，使用模型：{selected_model_display_name}...")

            scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(
                word=word,
                provider=selected_model_info["provider"],
                model=selected_model_info["model"],
                api_key=selected_model_info["api_key"]
            )
            
            status_placeholder.empty()
            
            if scores_all:
                membership = calculate_membership(scores_all)
                final_membership = membership.get(predicted_pos, 0)
                st.success(f'**分析完成**：词语「{word}」最可能的词类是 **【{predicted_pos}】**，隶属度为 **{final_membership:.4f}**')
                
                col_results_1, col_results_2 = st.columns(2)
                
                with col_results_1:
                    st.subheader("🏆 词类隶属度排名")
                    top10 = get_top_10_positions(membership)
                    top10_df = pd.DataFrame(top10, columns=["词类", "隶属度"])
                    top10_df["隶属度"] = top10_df["隶属度"].apply(lambda x: f"{x:.4f}")
                    st.table(top10_df)
                    
                    st.subheader("📊 词类隶属度雷达图")
                    plot_radar_chart_streamlit(dict(top10), f"「{word}」的词类隶属度分布")

                with col_results_2:
                    st.subheader("📋 各词类详细得分")
                    pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in scores_all.keys()}
                    sorted_pos_names = sorted(pos_total_scores.keys(), key=lambda pos: pos_total_scores[pos], reverse=True)
                    
                    for pos in sorted_pos_names:
                        total_score = pos_total_scores[pos]
                        max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                        with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                            rule_data = []
                            for rule_name, rule_score in scores_all[pos].items():
                                # 修复核心：简化规则描述查找（直接用全局RULE_SETS）
                                rule_desc = ""
                                if pos in RULE_SETS:
                                    for rule in RULE_SETS[pos]:
                                        if rule["name"] == rule_name:
                                            rule_desc = rule["desc"]
                                            break
                                rule_data.append({
                                    "规则代码": rule_name,
                                    "规则描述": rule_desc,
                                    "得分": rule_score
                                })
                            rule_data_sorted = sorted(rule_data, key=lambda x: x["得分"], reverse=True)
                            rule_df = pd.DataFrame(rule_data_sorted)
                            styled_df = rule_df.style.applymap(
                                lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, int) and x < 0 else "",
                                subset=["得分"]
                            )
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=min(len(rule_df) * 30 + 50, 400)
                            )
                    
                    st.subheader("📥 模型原始响应")
                    with st.expander("点击展开查看原始响应", expanded=False):
                        st.code(raw_text, language="text")

    # 批量处理
    with tab2:
        st.header("📂 批量任务实时监控")
        BACKUP_FILE = "batch_history_log.csv"

        # 控制面板
        st.subheader("🛠️ 控制面板")
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            # 可实时更新的metric占位符
            metric_placeholder = st.empty()
            # 初始化显示最新数量
            history_count = get_history_count(BACKUP_FILE)
            metric_placeholder.metric("已存数据量", f"{history_count} 条")
            
            has_history = os.path.exists(BACKUP_FILE)
            if has_history:
                st.caption(f"存储位置: `{BACKUP_FILE}`")
        
        with ctrl_col2:
            if os.path.exists(BACKUP_FILE):
                with open(BACKUP_FILE, "rb") as f:
                    st.download_button(
                        label="📥 下载历史文件(CSV)",
                        data=f,
                        file_name=f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.button("📥 下载历史文件", disabled=True, use_container_width=True)

        with ctrl_col3:
            if st.button("🗑️ 清空本地记录", use_container_width=True, type="secondary"):
                if os.path.exists(BACKUP_FILE):
                    try:
                        os.remove(BACKUP_FILE)
                        st.success("✅ 已清空本地记录")
                        # 清空后更新数据量显示
                        metric_placeholder.metric("已存数据量", "0 条")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空记录失败: {e}")
                else:
                    st.info("📄 暂无本地记录可清空")

        st.divider()

        # 运行状态
        st.subheader("📈 运行状态")
        progress_bar = st.progress(0)
        status_info = st.empty()
        
        # 实时结果预览
        st.subheader("📋 实时结果预览")
        table_placeholder = st.empty()
        if os.path.exists(BACKUP_FILE):
            try:
                table_placeholder.dataframe(
                    pd.read_csv(BACKUP_FILE, encoding='utf-8-sig'), 
                    use_container_width=True, 
                    height=300
                )
            except Exception as e:
                table_placeholder.error(f"显示历史记录失败: {e}")
        else:
            table_placeholder.info("暂无数据。上传文件并点击开始后，结果将在此逐行实时显示。")

        st.divider()

        # 上传任务
        st.subheader("📤 上传新任务")
        uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df_input = pd.read_excel(uploaded_file)
                target_col = next((col for col in df_input.columns if "词" in str(col) or "word" in str(col).lower()), None)
                
                if target_col:
                    st.write(f"✅ 识别到目标列: `{target_col}` | 待分析总数: {len(df_input)}")
                    
                    if st.button("🚀 开始处理 (自动续传)", type="primary", use_container_width=True):
                        if not selected_model_info["api_key"]:
                            st.error("❌ 请先在上方配置有效的 API Key")
                        else:
                            # 获取已处理的词语
                            existing_words = set()
                            if os.path.exists(BACKUP_FILE):
                                try:
                                    existing_df = pd.read_csv(BACKUP_FILE, encoding='utf-8-sig')
                                    if "词语" in existing_df.columns:
                                        existing_words = set(existing_df["词语"].astype(str).tolist())
                                    st.info(f"ℹ️ 已跳过 {len(existing_words)} 条已处理记录")
                                except Exception as e:
                                    st.warning(f"读取已处理记录失败，将重新处理所有数据: {e}")

                            total_rows = len(df_input)
                            
                            for index, row in df_input.iterrows():
                                word = str(row[target_col]).strip()
                                if not word:
                                    status_info.write(f"⏩ **跳过空值**: 第 {index+1}/{total_rows} 行")
                                    progress_bar.progress((index + 1) / total_rows)
                                    continue
                                
                                pct = int((index + 1) / total_rows * 100)
                                progress_bar.progress((index + 1) / total_rows)
                                
                                if word in existing_words:
                                    status_info.write(f"⏩ **跳过已处理**: {word} ({index+1}/{total_rows}) | 进度: {pct}%")
                                    continue
                                
                                status_info.write(f"🔍 **正在分析**: `{word}` | 进度: {index+1}/{total_rows} ({pct}%)")
                                
                                # 调用API处理
                                max_retries = 3
                                success = False
                                scores, raw_text, pred_pos, explanation = {}, "", "处理失败", "无响应"
                                for attempt in range(max_retries):
                                    try:
                                        scores, raw_text, pred_pos, explanation = ask_model_for_pos_and_scores(
                                            word=word,
                                            provider=selected_model_info["provider"],
                                            model=selected_model_info["model"],
                                            api_key=selected_model_info["api_key"]
                                        )
                                        success = bool(scores)
                                        if success:
                                            break
                                        time.sleep(2)
                                    except Exception as e:
                                        explanation = f"调用异常: {str(e)}"
                                        time.sleep(2)
                                
                                # 构造数据行
                                membership = calculate_membership(scores) if success else {}
                                new_row = {
                                    "序数": index + 1,
                                    "词语": word,
                                    "动词": membership.get("动词", 0.0),
                                    "名词": membership.get("名词", 0.0),
                                    "名动词": membership.get("名动词", 0.0),
                                    "差值/距离": round(abs(membership.get("动词", 0.0) - membership.get("名词", 0.0)), 4),
                                    "预测词类": pred_pos,
                                    "原始响应": raw_text if success else f"错误: {explanation}",
                                    "时间戳": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # 保存数据并实时更新数量
                                try:
                                    temp_df = pd.DataFrame([new_row])
                                    header_needed = not os.path.exists(BACKUP_FILE)
                                    temp_df.to_csv(
                                        BACKUP_FILE, 
                                        mode='a', 
                                        header=header_needed, 
                                        index=False, 
                                        encoding='utf-8-sig'
                                    )
                                    existing_words.add(word)
                                    # 实时更新已存数据量
                                    latest_count = get_history_count(BACKUP_FILE)
                                    metric_placeholder.metric("已存数据量", f"{latest_count} 条")
                                except Exception as csv_err:
                                    st.error(f"⚠️ 保存第 {index+1} 条记录失败: {csv_err}")
                                
                                # 刷新表格
                                try:
                                    updated_df = pd.read_csv(BACKUP_FILE, encoding='utf-8-sig')
                                    table_placeholder.dataframe(updated_df, use_container_width=True, height=300)
                                except Exception as read_err:
                                    st.warning(f"刷新表格失败: {read_err}")
                                
                                time.sleep(0.1)
                            
                            progress_bar.progress(100)
                            status_info.success(f"🎉 批量处理完成！总处理量: {total_rows} 条，已保存到 {BACKUP_FILE}")
                            st.rerun()
                else:
                    st.error("❌ 未识别到包含'词'或'word'的列，请检查Excel文件结构")
            except Exception as e:
                st.error(f"读取Excel文件失败: {e}")

# ===============================
# 运行主函数
# ===============================
if __name__ == "__main__":
    main()

# ===============================
# 页面底部说明
# ===============================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "© 2025 汉语词类隶属度检测划类  "
    "</div>",
    unsafe_allow_html=True
)

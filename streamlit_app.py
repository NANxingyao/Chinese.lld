import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
import time  # <--- 必须添加这行，用于降速和重试
from typing import Tuple, Dict, Any, List
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测划类",
    page_icon="📰",
    layout="wide",  # 使用宽布局
    initial_sidebar_state="collapsed",  # 默认折叠侧边栏
    menu_items=None
)

# 自定义CSS样式
hide_streamlit_style = """
<style>
/* 隐藏顶部菜单栏和页脚 */
header {visibility: hidden;}
footer {visibility: hidden;}

/* 调整表格样式 */
.dataframe {font-size: 12px;}

/* 隐藏默认的侧边栏 */
[data-testid="stSidebar"] {
    display: none !important;
}

/* 为顶部控制区添加边框和背景色，使其看起来像一个固定的面板 */
.stApp > div:first-child {
    padding-top: 2rem;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 模型配置 (修改版：启用流式 Stream)
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
            "X-DashScope-SSE": "enable",  # 显式开启 SSE
            "Accept": "text/event-stream" # 关键：告诉服务器我们要流式
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

# 模型选项（仅从环境变量获取API Key，已移除默认值）
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

# 过滤掉没有配置 API Key 的模型，只保留可用的
AVAILABLE_MODEL_OPTIONS = {
    name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]
}

# 如果没有可用模型，则显示所有模型但给出警告
if not AVAILABLE_MODEL_OPTIONS:
    AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 词类规则与最大得分
# ===============================
RULE_SETS = {
    # 1.1 名词
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
    # 1.5 动词
    "动词": [
        {"name": "V1_可受否定'不/没有'修饰", "desc": "可以受否定副词'不'或'没有'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "V2_可后附/插入时体助词'着/了/过'", "desc": "可以后附或中间插入时体助词'着/了/过'，或进入'...了没有'格式", "match_score": 10, "mismatch_score": 0},
        {"name": "V3_可带真宾语或通过介词引导论元", "desc": "可以带真宾语，或通过'和/为/对/向/拿/于'等介词引导论元", "match_score": 20, "mismatch_score": 0},
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语（按条目给予得分）", "match_score": 10, "mismatch_score": -10},
        {"name": "V5_可有重叠/正反重叠形式", "desc": "可以有'VV, V一V, V了V, V不V, V了没有'等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "V6_可做谓语或谓语核心", "desc": "可以做谓语或谓语核心（一般可受状语或补语修饰）", "match_score": 10, "mismatch_score": -10},
        {"name": "V7_不能作状语修饰动词性成分", "desc": "不能作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "V8_可作'怎么/怎样'提问或'这么/这样/那么'回答", "desc": "可以跟在'怎么/怎样'之后提问或跟在'这么/这样/那么'之后回答", "match_score": 10, "mismatch_score": 0},
        {"name": "V9_不能跟在'多/多么'之后提问或表示感叹", "desc": "不能跟在'多'之后对性质提问，不能跟在'多么'之后表示感叹", "match_score": 10, "mismatch_score": -10},
    ],
    # 4.6 名动词
    "名动词": [
        {"name": "NV1_可被\"不/没有\"否定且肯定形式-1", "desc": "可以用\"不\"和\"没有\"来否定，并且\"没有……\"的肯定形式可以是\"……了\"和\"有……\"(前一种情况中的\"没有\"是副词，后一种情况中的\"没有\"是动词)", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"，或者可以进入\"………了没有\"格式", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "可以有\"VV、V一V、V了V、V不V\"等重叠和正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语或谓语核心，又可以作主语或宾语，并且，可以作形式动词\"作、进行、加以、给予、受到\"等的宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后，对动作的方式进行提问，并且可以跟在\"这么、这样、那么、那样\"之后，用以作出相应的回答", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多\"之后，对性质的程度进行提问，也不能跟在\"多么\"之后，表示感叹", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构(然后作“在、到、从”等介词的宾语，这种介词结构又可以作状语或补语修饰动词性成分)", "match_score": 10, "mismatch_score": 0},
    ]
}

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
            
        # 兜底
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        # st.error(f"提取文本失败: {e}")
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """
    【新增】从包含推理过程和JSON的混合文本中提取并解析JSON对象。
    寻找最外层大括号 {} 包裹的JSON结构。
    """
    # 正则表达式匹配以 '{' 开始，以 '}' 结束的最外层结构
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    
    if not match:
        return None, text

    json_text = match.group(1).strip()
    
    # 尝试解析
    try:
        parsed_json = json.loads(json_text)
        return parsed_json, json_text
    except json.JSONDecodeError as e:
        # st.error(f"JSON解析失败: {e}. 尝试解析的文本片段:\n{json_text}")
        return None, json_text

def normalize_key(k: str, pos_rules: list) -> str:
    """标准化模型返回的规则名称，确保匹配到 RULE_SETS 中的键。"""
    if not isinstance(k, str): return None
    # 移除空格和下划线，转为大写进行匹配
    k_norm = re.sub(r'[\s_]+', '', k).upper()
    for r in pos_rules:
        r_norm = re.sub(r'[\s_]+', '', r["name"]).upper()
        if r_norm == k_norm:
            return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    """将模型返回的布尔值/字符串映射为规则定义的 match_score 或 mismatch_score。"""
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    
    if isinstance(raw_val, bool):
        return match_score if raw_val is True else mismatch_score
    
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"):
            return match_score
        if s in ("no", "n", "false", "否", "×", "不符合"):
            return mismatch_score
            
    # 即使模型错误地返回了数值，也尝试匹配规则分，否则默认不匹配
    if isinstance(raw_val, (int, float)):
        raw_val_int = int(raw_val)
        if raw_val_int == match_score: return match_score
        if raw_val_int == mismatch_score: return mismatch_score
    
    # 默认返回不匹配得分
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """计算隶属度：总分除以 100，并限制在 [-1, 1] 区间。"""
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        # 总得分除以100得到隶属度（几十分对应零点几）
        normalized = total_score / 100
        # 限制在 [-1.0, 1.0] 区间
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    """获取隶属度最高的前 10 个词类。"""
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

# ===============================
# 安全的 LLM 调用函数 (流式版)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    """
    封装请求逻辑，使用流式传输 (Streaming) 解决超时问题。
    逐步接收数据并拼接，最后返回完整的响应结构。
    """
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    if _provider not in MODEL_CONFIGS: return False, {"error": f"未知提供商 {_provider}"}, f"未知提供商 {_provider}"

    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)

    # 用于在界面上实时显示进度的占位符（可选，提升体验）
    streaming_placeholder = st.empty()
    full_content = ""

    try:
        # 1. 开启 stream=True
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            
            # 2. 逐行读取流式响应
            for line in response.iter_lines():
                if not line: continue
                
                # 解码并去除前缀
                line_text = line.decode('utf-8').strip()
                
                # 处理 SSE 格式 (通常以 "data: " 开头)
                if line_text.startswith("data:"):
                    json_str = line_text[5:].strip() # 去掉 "data:"
                else:
                    # 部分非标准流可能不带 data: 前缀，直接尝试解析
                    json_str = line_text

                # 遇到结束标记停止
                if json_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(json_str)
                    
                    # --- 提取文本片段 (Delta) ---
                    delta_text = ""
                    
                    # 情况 A: OpenAI / DeepSeek / Moonshot 格式
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        delta_text = delta.get("content", "")
                    
                    # 情况 B: Qwen Native 格式 (incremental_output=True)
                    elif "output" in chunk:
                        # Qwen Native 在 incremental_output=True 时，output.text 是增量
                        output = chunk["output"]
                        if "choices" in output and len(output["choices"]) > 0:
                             # Qwen 兼容 message 格式
                             msg = output["choices"][0].get("message", {})
                             delta_text = msg.get("content", "")
                        elif "text" in output:
                             # Qwen 纯文本格式
                             delta_text = output["text"]

                    if delta_text:
                        full_content += delta_text
                        # (可选) 实时在界面展示部分内容，让用户知道没死机
                        # streaming_placeholder.markdown(full_content + "▌")

                except json.JSONDecodeError:
                    continue
        
        # 清除流式占位符
        streaming_placeholder.empty()

        # 3. 构造一个模拟的完整响应，以便兼容后续的 extract_text_from_response 函数
        # 这样您就不需要修改后面的代码了
        mock_response = {
            "choices": [{"message": {"content": full_content}}], # OpenAI 风格
            "output": {"text": full_content} # Qwen 风格兼容
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
    if not word:
        return {}, "", "未知", ""

    # 规则文字说明（给模型看，让它老老实实按规则来判断）
    full_rules_by_pos = {
        pos: "\n".join([
            f"- {r['name']}: {r['desc']}（符合: {r['match_score']} 分，不符合: {r['mismatch_score']} 分）"
            for r in rules
        ])
        for pos, rules in RULE_SETS.items()
    }

    # ===== 系统提示：只允许输出“符合/不符合”，禁止自己打数字分 =====
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

    # 用户提示：再强调一次
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
        # 返回一个空结果，但保留错误信息
        return {}, f"调用失败: {err_msg}", "未知", f"模型调用失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    
    # 【修复核心问题】调用新增的 JSON 提取函数
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)

    # 解析 JSON
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
        cleaned_json_text = raw_text  # 展示原始文本

    # --- 关键：初始化所有词类的得分字典，并进行分数转换 ---
    scores_out = {pos: {} for pos in RULE_SETS.keys()}

    # 只把“符合/不符合”转成具体分值（正分 / 负分）
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    # 找到当前规则的定义
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    # 使用 map_to_allowed_score：true/false/“是/否”等 → match_score / mismatch_score
                    scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)

    # 保证每条规则都有得分，没有就默认 Mismatch Score（更严格）或 0 分（更保守）
    # 采用更保守的 0 分，因为模型没提及，可能是不适用，而不是明确不符合
    for pos, rules in RULE_SETS.items():
        for rule in rules:
            rule_name = rule["name"]
            if rule_name not in scores_out[pos]:
                scores_out[pos][rule_name] = 0

    return scores_out, raw_text, predicted_pos, explanation

# ===============================
# 雷达图
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm:
        st.warning("无法绘制雷达图：没有有效数据。")
        return
    
    # 过滤掉隶属度小于等于 0 的词类，以美化雷达图（可选，但通常雷达图只显示正向结果）
    # 这里我们保留所有数据，因为隶属度可能为负。但只显示分析的词类。
    
    categories = list(scores_norm.keys())
    if not categories:
        st.warning("无法绘制雷达图：没有有效词类。")
        return
        
    values = list(scores_norm.values())
    
    # 闭合雷达图
    categories += [categories[0]]
    values += [values[0]]
    
    # 确保 radialaxis range 包含负值，以正确显示负隶属度
    min_val = min(values)
    max_val = max(values)
    
    # 确保范围至少从 0 开始或包含 -1 到 1
    axis_min = min(min_val, -0.1) 
    axis_max = max(max_val, 1.0)
    
    # 调整雷达图的配置，使其更适用于负值（如果需要）
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values, 
            theta=categories, 
            fill="toself", 
            name="隶属度",
            hovertemplate = '<b>%{theta}</b><br>隶属度: %{r:.4f}<extra></extra>' # 优化悬停显示
        )
    ])
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[axis_min, axis_max], # 调整范围以包含负分
                tickvals=[0, 0.25, 0.5, 0.75, 1.0] if axis_min >= 0 else [-1.0, -0.5, 0, 0.5, 1.0] # 调整刻度
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 【最终增强版】批量处理（实时存档 + 断点生成）
# ===============================
def process_and_style_excel(df, selected_model_info, target_col_name):
    output = io.BytesIO()
    
    # --- 核心修改：使用 session_state 存储历史，防止界面刷新丢失 ---
    if 'processed_history' not in st.session_state:
        st.session_state.processed_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    backup_info_placeholder = st.container() # 用于显示实时进度和备份提示
    
    total = len(df)
    backup_file = "batch_process_history.csv" # 历史记录存盘文件

    try:
        for index, row in df.iterrows():
            word = str(row[target_col_name]).strip()
            
            # 检查是否已经处理过（可选：避免重复消耗Token）
            # if any(h['词语'] == word for h in st.session_state.processed_history): continue

            # 重试机制
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
            
            # 计算数据
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
            
            # --- 实时存留记录 ---
            # 1. 存入 SessionState (内存)
            st.session_state.processed_history.append(new_row)
            
            # 2. 写入本地磁盘 CSV (持久化)
            try:
                temp_df = pd.DataFrame([new_row])
                # header 只在文件不存在时写入
                header_needed = not os.path.exists(backup_file)
                temp_df.to_csv(backup_file, mode='a', header=header_needed, index=False, encoding='utf-8-sig')
            except:
                pass

            # 3. 实时在界面展示已完成的记录（让用户放心）
            with backup_info_placeholder:
                st.info(f"💾 已自动保存第 {index+1} 条记录。如遇中断，请检查目录下 `{backup_file}`")

            progress_bar.progress((index + 1) / total)
            time.sleep(0.5)

    except Exception as e:
        st.error(f"⚠️ 批量处理意外中断: {e}")
    
    # ===============================
    # 导出逻辑（支持中断后导出已完成部分）
    # ===============================
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
    
    # --- 顶部固定控制区 ---
    control_container = st.container()
    with control_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("⚙️ 模型设置")
            
            # 检查是否有可用模型
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 找不到可用的 API Key！请设置以下任意一个环境变量来启用模型:")
                for name, info in MODEL_OPTIONS.items():
                      st.code(f"export {info['env_var']}='你的API Key'", language="bash")
                
                # 禁用所有功能
                selected_model_display_name = list(MODEL_OPTIONS.keys())[0] # 占位符
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
            st.write("") # Spacer
            if not selected_model_info["api_key"]:
                st.button("测试模型链接 (不可用)", type="secondary", disabled=True)
            else:
                if st.button("测试模型链接", type="secondary"):
                    with st.spinner("正在测试连接..."):
                        # 使用一个简单的ping请求来测试连接
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

    # ===============================
    # 分页功能：单个分析 / 批量处理
    # ===============================
    tab1, tab2 = st.tabs(["🔍 单个词语详细分析", "📂 Excel 批量处理"])

    # --- Tab 1: 原有的单个词语分析逻辑 ---
    with tab1:
        st.subheader("🔤 词语输入")
        word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
        
        # 开始分析按钮（API Key为空时禁用）
        analyze_button = st.button(
            "🚀 开始分析", 
            type="primary",
            disabled=not (selected_model_info["api_key"] and word)
        )

        # --- 使用说明 ---
        with st.expander("ℹ️ 使用说明", expanded=False):
            st.info("""
            1. **配置 API Key**: 请在运行程序前设置必要的环境变量。
            2. **词语输入**：在上方的“词语输入”框中输入一个汉语词。
            3. **开始分析**：点击“开始分析”按钮。
            4. **结果解析**：系统将显示隶属度、雷达图和详细规则得分。
            """)

        # --- 结果显示区 ---
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
            
            # 只有在成功解析出分数时才进行后续显示
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
                    
                    # 1. 计算所有词类的总分
                    pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in RULE_SETS.keys()}
                    
                    # 按总分降序排序
                    sorted_pos_names = sorted(pos_total_scores.keys(), key=lambda pos: pos_total_scores[pos], reverse=True)
                    
                    # 2. 依次显示所有词类（而不是只显示前10，让用户可以看全部）
                    for pos in sorted_pos_names:
                        total_score = pos_total_scores[pos]
                        
                        # 找到该词类下得分最高的规则
                        max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                        
                        # 创建expander，显示词类名称、总分和最高分规则
                        with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                            # 显示该词类下的所有规则得分（按规则得分降序排列）
                            rule_data = []
                            for rule in RULE_SETS[pos]:
                                rule_score = scores_all[pos][rule["name"]]
                                rule_data.append({
                                    "规则代码": rule["name"],
                                    "规则描述": rule["desc"],
                                    "得分": rule_score
                                })
                            
                            # 按得分降序排序规则，让高分规则排在前面
                            rule_data_sorted = sorted(rule_data, key=lambda x: x["得分"], reverse=True)
                            rule_df = pd.DataFrame(rule_data_sorted)
                            
                            # 负分标红
                            styled_df = rule_df.style.applymap(
                                lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, int) and x < 0 else "",
                                subset=["得分"]
                            )
                            
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                # 动态调整高度，避免过高
                                height=min(len(rule_df) * 30 + 50, 400) 
                            )
                    
                    st.subheader("📥 模型原始响应")
                    with st.expander("点击展开查看原始响应", expanded=False):
                        st.code(raw_text, language="text") # 改为 text 以更好地展示混合文本
# --- Tab 2: 批量处理逻辑 (历史记录常驻显示版) ---
    with tab2:
        st.header("📂 批量 Excel 处理")
        
        # 定义本地持久化文件名
        BACKUP_FILE = "batch_history_log.csv"

        # ==========================================
        # 1. 实时历史记录看板 (常驻显示)
        # ==========================================
        st.subheader("📊 历史记录与恢复面板")
        
        if os.path.exists(BACKUP_FILE):
            try:
                # 读取本地已保存的所有数据
                history_df = pd.read_csv(BACKUP_FILE)
                total_saved = len(history_df)
                
                # 使用两栏展示状态和下载
                stat_col, action_col = st.columns([2, 1])
                with stat_col:
                    st.info(f"✅ **检测到本地存留数据**：共计 **{total_saved}** 条记录。")
                    # 展示最后3条实时结果
                    st.dataframe(history_df.tail(3), use_container_width=True, height=150)
                
                with action_col:
                    st.write("操作已存数据：")
                    # 即使程序中断，这个按钮也一直存在，可以直接下载已完成的部分
                    st.download_button(
                        "📥 导出已完成记录 (CSV)",
                        data=history_df.to_csv(index=False, encoding='utf-8-sig'),
                        file_name=f"分析记录备份_{time.strftime('%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    if st.button("🗑️ 清空本地缓存", use_container_width=True, help="处理新文件前请先清空旧数据"):
                        os.remove(BACKUP_FILE)
                        if 'processed_history' in st.session_state:
                            st.session_state.processed_history = []
                        st.rerun()
            except Exception as e:
                st.warning(f"读取备份时遇到点小问题: {e}")
        else:
            st.write("💡 当前暂无历史存留记录。开始处理后，数据将实时显示在这里。")

        st.divider()

        # ==========================================
        # 2. 上传与处理区
        # ==========================================
        st.subheader("🚀 新任务上传")
        uploaded_file = st.file_uploader("上传待分析的 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                # 自动识别列名
                target_col = next((col for col in df.columns if "词" in str(col) or "word" in str(col).lower()), None)
                
                if not target_col:
                    st.error("❌ 无法识别目标列。请确保 Excel 中有一列标题包含 '词' 或 'Word'。")
                else:
                    st.success(f"准备就绪！目标列：`{target_col}` | 总数：{len(df)}")
                    
                    if st.button("开始批量执行 (自动存盘)", type="primary"):
                        if not selected_model_info["api_key"]:
                            st.error("API Key 未配置，请在上方设置。")
                        else:
                            # 调用处理函数
                            excel_data = process_and_style_excel(df, selected_model_info, target_col)
                            
                            if excel_data:
                                st.balloons()
                                st.success("🎉 全部任务处理完成！")
                                st.download_button(
                                    label="💾 下载最终标黄版 Excel",
                                    data=excel_data,
                                    file_name="词类分析结果_最终版.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
            
            except Exception as e:
                st.error(f"文件处理出错: {e}")


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

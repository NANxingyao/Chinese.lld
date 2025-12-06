import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List
# 增加宽松的JSON解析库（需先安装：pip install demjson）
try:
    import demjson
    DEMJSON_AVAILABLE = True
except ImportError:
    DEMJSON_AVAILABLE = False
    st.warning("未安装demjson库，JSON解析容错能力会降低。建议执行：pip install demjson")

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
# 模型配置 (仅从环境变量获取API Key)
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "input": {"messages": messages}, "parameters": {"max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0),},
        },
    },
}

# 模型选项（仅从环境变量获取API Key，不提供手动输入）
MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek", 
        "model": "deepseek-chat", 
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-759d66c83f374a2aaac0db5814ccb016"),
        "env_var": "DEEPSEEK_API_KEY"
    },
    "OpenAI GPT-4o（尚不支持）": {
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

# 预计算每个词类的最大可能得分
MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict): return ""
    try:
        # --- 处理通义千问 (Qwen) 的响应格式 ---
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
            
        # --- 处理 OpenAI 系列的响应格式 ---
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            for k in ("content", "text"):
                if k in choice: return choice[k]
    except Exception: 
        pass
    # 如果以上都失败，返回整个响应的字符串形式，用于调试
    return json.dumps(resp_json, ensure_ascii=False)

def safe_json_parse(json_str: str) -> Tuple[dict, str]:
    """
    安全的JSON解析函数，支持多种容错方式
    返回：(解析后的字典, 最终尝试的JSON字符串)
    """
    # 第一步：基础清理
    json_str = json_str.strip()
    # 移除注释
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    # 替换中文标点
    json_str = json_str.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"').replace("’", "'")
    # 处理单引号
    json_str = re.sub(r"'(\s*[^']+?\s*)'\s*:", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*?)'", r': "\1"', json_str)
    # 移除末尾逗号
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    # 处理布尔值
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)
    
    # 第二步：尝试标准解析
    try:
        return json.loads(json_str), json_str
    except Exception as e1:
        # 第三步：尝试demjson宽松解析（如果可用）
        if DEMJSON_AVAILABLE:
            try:
                parsed = demjson.decode(json_str)
                if isinstance(parsed, dict):
                    return parsed, json_str
            except Exception as e2:
                st.warning(f"demjson解析失败: {e2}")
        
        # 第四步：尝试提取最外层的{}内容
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            try:
                inner_str = match.group(0)
                return json.loads(inner_str), inner_str
            except:
                pass
        
        # 所有方法都失败
        return None, json_str

def extract_json_from_text(text: str) -> Tuple[dict, str]:
    """
    增强版JSON提取函数
    优先提取```json```代码块，再尝试其他方式
    """
    if not text:
        return None, ""
    
    # 第一步：优先提取```json```代码块
    json_block_pattern = re.compile(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', re.IGNORECASE)
    json_block_matches = json_block_pattern.findall(text)
    if json_block_matches:
        for json_str in json_block_matches:
            parsed, final_str = safe_json_parse(json_str)
            if parsed is not None:
                return parsed, final_str
    
    # 第二步：提取所有{}包裹的内容
    all_json_matches = re.findall(r'\{[\s\S]*\}', text)
    if all_json_matches:
        for json_str in all_json_matches:
            parsed, final_str = safe_json_parse(json_str)
            if parsed is not None:
                return parsed, final_str
    
    # 第三步：尝试直接解析整个文本
    parsed, final_str = safe_json_parse(text)
    if parsed is not None:
        return parsed, final_str
    
    # 所有方法都失败
    st.warning("无法从文本中提取有效的JSON结构")
    return None, text
    
def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_upper = re.sub(r'\s+', '', k).upper()
    for r in pos_rules:
        if re.sub(r'\s+', '', r["name"]).upper() == k_upper:
            return r["name"]
    return None

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    # 强制保留原始得分中的负分（如果是有效规则分）
    if isinstance(raw_val, (int, float)):
        # 允许匹配得分或不匹配得分（包括负分）
        if raw_val == match_score or raw_val == mismatch_score:
            return int(raw_val)
    if isinstance(raw_val, bool):
        return match_score if raw_val else mismatch_score
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"):
            return match_score
        if s in ("no", "n", "false", "否", "×", "不符合"):
            return mismatch_score
    # 无效值时返回不匹配得分（保留负分）
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        # 改为：总得分除以100得到隶属度（几十分对应零点几）
        # 同时限制在 [0, 1] 区间内
        # 负分可降低隶属度，保留原始计算逻辑但不强制截断为0（可选调整）
        normalized = total_score / 100
        # 若需允许隶属度为负（更准确反映负分影响），可改为：
        # membership[pos] = normalized
        # 若需限制在[-1, 1]区间：
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

def prepare_detailed_scores_df(scores_all: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for pos, rules in RULE_SETS.items():
        for rule in rules:
            rows.append({
                "词类": pos,
                "规则代码": rule["name"],
                "规则描述": rule["desc"],
                "得分": scores_all[pos].get(rule["name"], 0)
            })
    return pd.DataFrame(rows)

# ===============================
# 安全的 LLM 调用函数 (增加超时)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    if not _api_key: return False, {"error": "API Key 为空"}, "API Key 未提供"
    if _provider not in MODEL_CONFIGS: return False, {"error": f"未知提供商 {_provider}"}, f"未知提供商 {_provider}"

    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)

    try:
        # 增加超时设置到120秒
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return True, response.json(), ""
    except requests.exceptions.Timeout:
        error_msg = "请求超时。模型可能正忙或网络连接较慢。建议尝试其他模型或稍后再试。"
        return False, {"error": error_msg}, error_msg
    except requests.exceptions.RequestException as e:
        # 对于4xx和5xx错误，提取更多信息
        error_msg = f"API请求失败: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                if 'error' in error_details:
                    error_msg += f" 详情: {error_details['error']['message']}"
            except:
                error_msg += f" 响应内容: {e.response.text[:200]}..." # 只显示部分内容
        return False, {"error": error_msg}, error_msg
    except Exception as e:
        error_msg = f"发生未知错误: {str(e)}"
        return False, {"error": error_msg}, error_msg

# ===============================
# 词类判定主函数 (优化Prompt)
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

    # ===== 增强版系统提示：强制规范输出 =====
    system_msg = f"""你是一名中文词法与语法方面的专家。现在要分析词语「{word}」在下列词类中的表现：

- 需要判断的词类：名词、动词、名动词
- 评分规则已经由系统定义，你**不要**自己设计分值，也**不要**在 JSON 中给出具体数字分数
- 你只需要判断每一条规则是“符合”还是“不符合”，程序会自动根据 match_score / mismatch_score 换算成正分或负分

【各词类的规则说明（仅供你判断使用）】

【名词】
{full_rules_by_pos["名词"]}

【动词】
{full_rules_by_pos["动词"]}

【名动词】
{full_rules_by_pos["名动词"]}

【输出要求 - 必须严格遵守】
1. 首先输出详细的推理过程：
   - 逐条规则说明判断依据，并举例（可以自己造句）
   - 格式示例：
     - 「名词-N1_可受数量词修饰：符合。理由：……。例句：……。」
     - 「动词-V2_可后附/插入时体助词'着/了/过'：不符合。理由：……。例句：……。」
   - 必须覆盖三个词类的所有规则，不能遗漏

2. 推理过程结束后，单独输出一个```json```代码块，包含以下结构：
{{
  "explanation": "完整的推理过程文本",
  "predicted_pos": "名词/动词/名动词（选择其一）",
  "scores": {{
    "名词": {{
      "N1_可受数量词修饰": true/false,
      "N2_不能受副词修饰": true/false,
      // 其他名词规则...
    }},
    "动词": {{
      "V1_可受否定'不/没有'修饰": true/false,
      // 其他动词规则...
    }},
    "名动词": {{
      "NV1_可被\"不/没有\"否定且肯定形式-1": true/false,
      // 其他名动词规则...
    }}
  }}
}}

3. 特别注意：
   - JSON中scores字段下的所有值只能是true或false，严禁使用数字、中文或其他值
   - JSON必须是合法的，不能有语法错误
   - JSON必须用```json```代码块包裹，且单独成行
   - 不要在JSON中添加任何注释
"""

    # 用户提示：再强调一次
    user_prompt = f"""
请严格按照上述要求分析词语「{word}」。

特别注意：
- 在 JSON 的 scores 部分，只能用 true/false 表示“是否符合规则”，不能使用任何数字、中文或其他值。
- explanation 中必须对每一条规则写明“符合/不符合 + 理由 + 例句”。
- 最终的JSON必须用```json```代码块包裹，确保可以被程序正确提取。

请先给出详细推理过程，然后在最后单独输出JSON代码块。
"""

    with st.spinner("正在调用大模型进行分析，请稍候..."):
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
        return {}, f"调用失败: {err_msg}", "未知", f"调用失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)

    # 解析 JSON
    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
        st.success("✅ 成功解析模型输出的JSON结构")
    else:
        st.warning("⚠️ 未能从模型响应中解析出有效的JSON，将使用默认得分")
        # 调试信息：显示清理后的JSON文本
        with st.expander("📝 解析失败的JSON文本（调试用）", expanded=False):
            st.code(cleaned_json_text, language="json")
        explanation = "无法解析模型输出。原始响应：\n" + raw_text
        predicted_pos = "未知"
        raw_scores = {}

    # --- 关键：初始化所有词类的得分字典 ---
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

    # 保证每条规则都有得分，没有就默认 0 分（说明模型完全没提到）
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
    categories = list(scores_norm.keys())
    if not categories:
        st.warning("无法绘制雷达图：没有有效词类。")
        return
    values = list(scores_norm.values())
    
    # 闭合雷达图
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")])
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=16))
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 主页面逻辑
# ===============================
def main():
    st.title("📰 汉语词类隶属度检测划类")
    
    # --- 顶部固定控制区 ---
    control_container = st.container()
    with control_container:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.subheader("⚙️ 模型设置")
            selected_model_display_name = st.selectbox("选择大模型", list(MODEL_OPTIONS.keys()), key="model_select")
            selected_model_info = MODEL_OPTIONS[selected_model_display_name]
            
            # 检查API Key是否存在
            if not selected_model_info["api_key"]:
                st.error(f"❌ 未找到 {selected_model_display_name} 的API Key")
                st.info(f"请设置环境变量 **{selected_model_info['env_var']}** 后重试")
                st.code(f"# Linux/Mac\n export {selected_model_info['env_var']}='你的API Key'\n\n# Windows\n set {selected_model_info['env_var']}='你的API Key'", language="bash")
        
        with col2:
            st.subheader("🔗 连接测试")
            if not selected_model_info["api_key"]:
                st.disabled(True)
                st.button("测试模型链接", type="secondary", disabled=True)
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
                        st.success("✅ 模型链接测试成功！")
                    else:
                        st.error(f"❌ 模型链接测试失败: {err_msg}")

        with col3:
            st.subheader("🔤 词语输入")
            word = st.text_input("请输入要分析的汉语词语", placeholder="例如：苹果、跑、美丽...", key="word_input")
            
            # 开始分析按钮（API Key为空时禁用）
            analyze_button = st.button(
                "🚀 开始分析", 
                type="primary",
                disabled=not (selected_model_info["api_key"] and word)
            )

    st.markdown("---")

    
    # --- 使用说明 ---
    info_container = st.container()
    with info_container:
        with st.expander("ℹ️ 使用说明", expanded=False):
            st.info("""
            1. 在上方的“词语输入”框中输入一个汉语词。
            2. （可选）在模型选择区域点击向下箭头展开，可以选择不同的大语言模型。
            3. （可选）点击“测试模型链接”按钮，确认所选模型可以正常访问。
            4. 点击“开始分析”按钮，系统将使用选定的大模型分析该词语的词类隶属度。
            5. 分析结果将显示在下方，包括隶属度排名、详细得分、推理过程和原始响应。
            """)

     # --- 结果显示区 ---
    if analyze_button and word and selected_model_info["api_key"]:
        status_placeholder = st.empty()
        status_placeholder.info(f"正在为词语「{word}」启动分析...")

        scores_all, raw_text, predicted_pos, explanation = ask_model_for_pos_and_scores(
            word=word,
            provider=selected_model_info["provider"],
            model=selected_model_info["model"],
            api_key=selected_model_info["api_key"]
        )
        
        status_placeholder.empty()
        
        membership = calculate_membership(scores_all)
        st.success(f'**分析完成**：词语「{word}」最可能的词类是 【{predicted_pos}】，隶属度为 {membership.get(predicted_pos, 0):.4f}')
        
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
            
            # 1. 计算所有词类的总分并排序，取前10名
            pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in RULE_SETS.keys()}
            # 按总分降序排序，取前10
            top10_pos = sorted(pos_total_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # 2. 只显示排名前10的词类
            for pos, total_score in top10_pos:
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
                        height=min(len(rule_df) * 30 + 50, 800)
                    )
            
            st.subheader("📥 模型原始响应")
            with st.expander("点击展开查看原始响应", expanded=False):
                st.code(raw_text, language="text")
            
            st.subheader("🔍 模型推理过程")
            with st.expander("点击展开查看推理过程", expanded=False):
                st.markdown(explanation)

if __name__ == "__main__":
    main()

# ===============================
# 页面底部说明
# ===============================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "© 2025 汉语词类隶属度检测划类 "
    "</div>",
    unsafe_allow_html=True
)


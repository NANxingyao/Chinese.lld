import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List

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
# 词类规则与得分（核心：明确match/mismatch分数）
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
# 工具函数（核心修改：直接读取分数，抛弃布尔值）
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

def fix_common_json_errors(json_str: str) -> str:
    """自动修复常见的JSON格式错误（无需依赖外部库）"""
    # 1. 给无引号的键加双引号（比如 key: → "key":）
    json_str = re.sub(r'([{,]\s*)([\w_]+)(\s*:)', r'\1"\2"\3', json_str)
    # 2. 把单引号键改成双引号（比如 'key': → "key":）
    json_str = re.sub(r"([{,]\s*)'([\w_]+)'(\s*:)", r'\1"\2"\3', json_str)
    # 3. 补全键值对后的缺失逗号（比如 "a":10 "b":-20 → "a":10, "b":-20）
    json_str = re.sub(r'("[\w_]+":\s*[^,}]+)\s+("[\w_]+":)', r'\1,\2', json_str)
    # 4. 移除末尾多余的逗号（比如 {"a":10,} → {"a":10}）
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # 5. 替换中文标点为英文
    json_str = json_str.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
    # 6. 移除多余的空格/换行
    json_str = re.sub(r'\s+', ' ', json_str).replace('{ ', '{').replace(' }', '}')
    return json_str

def extract_json_from_text(text: str) -> Tuple[dict, str]:
    """
    增强版JSON提取函数：
    1. 优先用专属分隔符提取
    2. 自动修复常见格式错误
    3. 多层级容错
    """
    if not text:
        return None, ""

    # 第一步：用专属分隔符精准提取JSON块（最优先）
    start_marker = "====JSON_BEGIN===="
    end_marker = "====JSON_END===="
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx > start_idx:
        # 提取分隔符之间的内容
        json_str = text[start_idx + len(start_marker):end_idx].strip()
        # 修复常见错误
        json_str = fix_common_json_errors(json_str)
        # 尝试解析
        try:
            parsed = json.loads(json_str)
            st.success("✅ 通过专属分隔符成功解析JSON")
            return parsed, json_str
        except Exception as e:
            st.warning(f"分隔符提取的JSON解析失败，错误：{str(e)[:100]}")
            st.code(f"修复后的JSON：\n{json_str}", language="json")

    # 第二步：fallback到代码块提取
    json_block_pattern = re.compile(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', re.IGNORECASE)
    json_block_matches = json_block_pattern.findall(text)
    for json_str in json_block_matches:
        json_str = fix_common_json_errors(json_str.strip())
        try:
            parsed = json.loads(json_str)
            st.success("✅ 通过代码块成功解析JSON")
            return parsed, json_str
        except Exception as e:
            continue

    # 第三步：最后尝试提取所有大括号内容
    all_json_matches = re.findall(r'\{[\s\S]*\}', text)
    for json_str in all_json_matches:
        json_str = fix_common_json_errors(json_str.strip())
        try:
            parsed = json.loads(json_str)
            st.success("✅ 通过大括号匹配成功解析JSON")
            return parsed, json_str
        except Exception as e:
            continue

    # 所有方法失败
    st.warning("⚠️ 无法提取有效JSON，将使用默认得分")
    # 显示原始文本供调试
    with st.expander("📝 原始响应文本（调试用）", expanded=False):
        st.code(text, language="text")
    return None, text

def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str): return None
    k_upper = re.sub(r'\s+', '', k).upper()
    for r in pos_rules:
        if re.sub(r'\s+', '', r["name"]).upper() == k_upper:
            return r["name"]
    return None

def validate_score(rule: dict, raw_val) -> int:
    """直接验证分数是否合法，不转换布尔值"""
    match_score, mismatch_score = rule["match_score"], rule["mismatch_score"]
    # 如果是数字，直接验证是否在允许的范围内
    if isinstance(raw_val, (int, float)):
        raw_val = int(raw_val)
        # 只允许匹配分或不匹配分
        if raw_val == match_score or raw_val == mismatch_score:
            return raw_val
    # 如果是字符串，尝试转数字
    if isinstance(raw_val, str):
        try:
            num_val = int(raw_val.strip())
            if num_val == match_score or num_val == mismatch_score:
                return num_val
        except:
            pass
    # 无效值返回不匹配分（兜底）
    return mismatch_score

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        # 总得分除以100得到隶属度，限制在[-1, 1]区间
        normalized = total_score / 100
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
# 词类判定主函数（核心修改：Prompt强制输出分数）
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str, str]:
    if not word:
        return {}, "", "未知", ""

    # 生成带具体分数的规则说明（给模型看）
    full_rules_by_pos = {}
    for pos, rules in RULE_SETS.items():
        rule_text = []
        for r in rules:
            rule_text.append(f"- {r['name']}: {r['desc']}（符合填{r['match_score']}分，不符合填{r['mismatch_score']}分）")
        full_rules_by_pos[pos] = "\n".join(rule_text)

    # ===== 核心修改：Prompt强制输出分数，抛弃布尔值 =====
    system_msg = f"""你是中文词法与语法领域的专家，需严格按照以下要求分析词语「{word}」的词类隶属度，格式错误会导致任务完全失败！

【分析范围】
仅分析以下三类词类：名词、动词、名动词，每条规则必须输出**具体的数字得分**（符合=匹配分，不符合=不匹配分）。

【规则与得分说明】
{chr(10).join([f"【{pos}】{full_rules_by_pos[pos]}" for pos in full_rules_by_pos.keys()])}

【输出格式（必须100%遵守，缺一不可）】
1. 首先输出**详细推理过程**：
   - 逐条规则说明判断结果（符合/不符合）+ 理由 + 例句 + 对应分数
   - 格式示例：「名词-N1_可受数量词修饰：符合，得10分。理由：苹果可以说“一个苹果”，受数量词修饰。例句：我买了一个苹果。」
   - 必须覆盖名词（8条）、动词（9条）、名动词（10条）的所有规则，不能遗漏任何一条

2. 推理过程结束后，单独输出**专属分隔符包裹的JSON**（分隔符必须单独成行，不能修改）：
====JSON_BEGIN====
{{
  "explanation": "这里填写完整的推理过程文本（包含所有规则的判断理由、例句和分数）",
  "predicted_pos": "从名词/动词/名动词中选择一个作为最终判定结果",
  "scores": {{
    "名词": {{
      "N1_可受数量词修饰": 10,
      "N2_不能受副词修饰": 20,
      "N3_可作主宾语": 20,
      "N4_可作中心语或作定语": 10,
      "N5_可后附的字结构": 10,
      "N6_可后附方位词构处所": 10,
      "N7_不能作谓语核心": 10,
      "N8_不能作补语/一般不作状语": 10
    }},
    "动词": {{
      "V1_可受否定'不/没有'修饰": 10,
      "V2_可后附/插入时体助词'着/了/过'": 10,
      "V3_可带真宾语或通过介词引导论元": 20,
      "V4_程度副词与带宾语的关系": 10,
      "V5_可有重叠/正反重叠形式": 10,
      "V6_可做谓语或谓语核心": 10,
      "V7_不能作状语修饰动词性成分": 10,
      "V8_可作'怎么/怎样'提问或'这么/这样/那么'回答": 10,
      "V9_不能跟在'多/多么'之后提问或表示感叹": 10
    }},
    "名动词": {{
      "NV1_可被\"不/没有\"否定且肯定形式-1": 10,
      "NV2_可附时体助词或进入\"……了没有\"格式": 10,
      "NV3_可带真宾语且不受\"很\"修饰": 10,
      "NV4_有重叠和正反重叠形式": 10,
      "NV5_可作多种句法成分且可作形式动词宾语": 10,
      "NV6_不能直接作状语": 10,
      "NV7_可修饰名词或受名词/数量词修饰": 10,
      "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后": 10,
      "NV9_不能跟在\"多/多么\"之后": 10,
      "NV10_可后附方位词构成处所结构": 10
    }}
  }}
}}
====JSON_END====

【强制要求】
- JSON必须和上述模板结构完全一致（不能增删字段、不能修改规则名称），仅替换具体的数字得分和文本内容
- JSON中所有值必须是**整数**（如10、-20、0），不能使用布尔值（true/false）、中文、字符串
- JSON中所有键必须用双引号包裹，数字不能加引号，比如"N1_可受数量词修饰": 10（正确），"N1": "10"（错误）
- 分隔符====JSON_BEGIN====和====JSON_END====必须单独成行，且前后不能有其他内容
- 推理过程必须包含所有规则的判断理由、例句和分数，不能省略
- 若违反以上任何一条，本次分析视为无效
"""

    user_prompt = f"""请严格按照上述要求，分析汉语词语「{word}」的词类隶属度，输出推理过程和规范的JSON（所有规则必须填具体分数）。"""

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

    # 解析JSON并补全缺失的规则（直接读分数）
    if parsed_json and isinstance(parsed_json, dict):
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
        
        # 关键：补全所有缺失的词类和规则（兜底）
        for pos in RULE_SETS.keys():
            if pos not in raw_scores:
                raw_scores[pos] = {}
            # 补全当前词类的所有规则（默认填不匹配分）
            for rule in RULE_SETS[pos]:
                rule_name = rule["name"]
                if rule_name not in raw_scores[pos]:
                    raw_scores[pos][rule_name] = rule["mismatch_score"]
    else:
        # 完全解析失败时，初始化所有规则为不匹配分
        explanation = "模型输出格式错误，使用默认得分。"
        predicted_pos = "未知"
        raw_scores = {
            pos: {rule["name"]: rule["mismatch_score"] for rule in RULE_SETS[pos]} 
            for pos in RULE_SETS.keys()
        }

    # 验证分数（直接读数字，不转换布尔值）
    scores_out = {pos: {} for pos in RULE_SETS.keys()}
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    scores_out[pos][normalized_key] = validate_score(rule_def, v)

    # 保证每条规则都有得分（最终兜底）
    for pos, rules in RULE_SETS.items():
        for rule in rules:
            rule_name = rule["name"]
            if rule_name not in scores_out[pos]:
                scores_out[pos][rule_name] = rule["mismatch_score"]

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

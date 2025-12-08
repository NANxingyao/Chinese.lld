import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor # 引入多线程并发库

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
# 模型配置 (仅从环境变量获取API Key，移除硬编码值以提高安全性)
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
            "model": model, "input": {"messages": messages}, "parameters": {"max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0)},
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
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
        
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        st.error(f"提取文本失败: {e}")
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """从包含推理过程和JSON的混合文本中提取并解析JSON对象。"""
    # 使用 re.DOTALL 确保 '.' 匹配换行符
    match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
    
    if not match:
        return None, text

    json_text = match.group(1).strip()
    
    try:
        parsed_json = json.loads(json_text)
        return parsed_json, json_text
    except json.JSONDecodeError as e:
        # st.error(f"JSON解析失败: {e}. 尝试解析的文本片段:\n{json_text}") # 调试信息不在最终应用中显示
        return None, json_text

def normalize_key(k: str, pos_rules: list) -> str:
    """标准化模型返回的规则名称，确保匹配到 RULE_SETS 中的键。"""
    if not isinstance(k, str): return None
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
        normalized = total_score / 100
        membership[pos] = max(-1.0, min(1.0, normalized))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    """获取隶属度最高的前 10 个词类。"""
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

# ===============================
# 安全的 LLM 调用函数 (增加超时)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    """封装请求逻辑，增加超时处理和错误信息提取。"""
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
        error_msg = "请求超时。模型可能正忙或网络连接较慢。"
        return False, {"error": error_msg}, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"API请求失败: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                if 'error' in error_details:
                    detail_msg = error_details['error'].get('message') or json.dumps(error_details['error'])
                    error_msg += f" 详情: {detail_msg}"
                else:
                     error_msg += f" 响应内容: {e.response.text[:200]}..." 
            except:
                error_msg += f" 响应内容: {e.response.text[:200]}..." 
        return False, {"error": error_msg}, error_msg
    except Exception as e:
        error_msg = f"发生未知错误: {str(e)}"
        return False, {"error": error_msg}, error_msg

# ===============================
# 词类判定主函数 (针对单个词语)
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Dict[str, Any]:
    """
    对单个词语进行分析，并返回包含所有结果的字典。
    """
    if not word:
        return {"word": "", "error": "词语为空", "scores_all": {}, "predicted_pos": "未知", "explanation": ""}

    # 规则文字说明（给模型看）
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
- 评分规则已经由系统定义，你**不要**自己设计分值，也**不要**在 JSON 中给出具体数字分数。
- 你只需要判断每一条规则是“符合”还是“不符合”。

【各词类的规则说明（仅供你判断使用）】

【名词】
{full_rules_by_pos["名词"]}

【动词】
{full_rules_by_pos["动词"]}

【名动词】
{full_rules_by_pos["名动词"]}

【输出要求】
1. 在 explanation 字段中，必须**逐条规则**说明判断依据，并举例（可以自己造句）。
2. 在 JSON 中的 scores 字段里，每一类下的每一条规则，只能给出 **布尔值 true / false**。
3. predicted_pos：请选择「名词」「动词」「名动词」之一，作为该词语最典型的词类。
4. **最后输出时，先写详细的文字推理，最后单独且完整地给出一段合法的 JSON（不要再加注释）。**
"""
    user_prompt = f"""请严格按照上述要求分析词语「{word}」。请先给出详细推理过程，然后在最后单独输出一个 JSON 对象。"""

    # 调用模型
    ok, resp_json, err_msg = call_llm_api_cached(
        _provider=provider,
        _model=model,
        _api_key=api_key,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )

    result = {"word": word, "error": "", "scores_all": {}, "predicted_pos": "未知", "explanation": "", "raw_text": ""}

    if not ok:
        result["error"] = f"模型调用失败: {err_msg}"
        return result

    raw_text = extract_text_from_response(resp_json)
    result["raw_text"] = raw_text
    parsed_json, _ = extract_json_from_text(raw_text)

    # 解析 JSON 并转换分数
    if parsed_json and isinstance(parsed_json, dict):
        result["explanation"] = parsed_json.get("explanation", "模型未提供详细推理过程。")
        result["predicted_pos"] = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
        
        scores_out = {pos: {} for pos in RULE_SETS.keys()}
        for pos, rules in RULE_SETS.items():
            raw_pos_scores = raw_scores.get(pos, {})
            if isinstance(raw_pos_scores, dict):
                for k, v in raw_pos_scores.items():
                    normalized_key = normalize_key(k, rules)
                    if normalized_key:
                        rule_def = next(r for r in rules if r["name"] == normalized_key)
                        scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)
            
            # 保证每条规则都有得分，缺失默认 0 分
            for rule in rules:
                if rule["name"] not in scores_out[pos]:
                    scores_out[pos][rule["name"]] = 0
        
        result["scores_all"] = scores_out
        
    else:
        result["error"] = "未能从模型响应中解析出有效的JSON。"
        result["explanation"] = "无法解析模型输出。原始响应：\n" + raw_text

    return result

# ===============================
# 批量处理函数（利用并发）
# ===============================
def process_batch(words: List[str], model_info: Dict[str, Any], max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    使用 ThreadPoolExecutor 并发处理多个词语。
    """
    if not words:
        return []

    results = []
    
    # 使用 ThreadPoolExecutor 来管理并发线程
    # max_workers = 5 是一个合理的默认值，防止API限速
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交每个词语的分析任务
        futures = {
            executor.submit(
                ask_model_for_pos_and_scores, 
                word.strip(), 
                model_info["provider"], 
                model_info["model"], 
                model_info["api_key"]
            ): word 
            for word in words if word.strip()
        }
        
        # 获取结果，保持提交的顺序
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # 捕获线程执行中的意外错误
                results.append({"word": "未知", "error": f"并发执行发生异常: {e}", "scores_all": {}, "predicted_pos": "未知", "explanation": ""})
                
    return results

# ===============================
# 雷达图
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    if not scores_norm:
        st.warning("无法绘制雷达图：没有有效数据。")
        return
        
    # 只取前三个词类（名词、动词、名动词）绘制雷达图
    relevant_pos = {k: scores_norm[k] for k in ["名词", "动词", "名动词"] if k in scores_norm}
    
    categories = list(relevant_pos.keys())
    values = list(relevant_pos.values())
    
    if not categories:
        st.warning("无法绘制雷达图：没有分析所需的词类数据。")
        return
        
    # 闭合雷达图
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
                tickvals=[-1.0, -0.5, 0, 0.5, 1.0] 
            )
        ),
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
            
            # 检查是否有可用模型
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 找不到可用的 API Key！请设置环境变量来启用模型。")
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
            if not selected_model_info.get("api_key"):
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
                        st.success("✅ 模型链接测试成功！")
                    else:
                        st.error(f"❌ 模型链接测试失败: {err_msg}")

        with col3:
            st.subheader("🔤 词语输入 (支持批量)")
            
            # 更改为 text_area 支持批量输入
            words_input = st.text_area(
                "请输入要分析的汉语词语（每行一个）", 
                placeholder="例如：\n苹果\n跑\n美丽", 
                key="words_input", 
                height=150
            )
            
            words_list = [w.strip() for w in words_input.split('\n') if w.strip()]
            
            # 开始分析按钮（API Key为空或词语为空时禁用）
            analyze_button = st.button(
                f"🚀 开始分析 ({len(words_list)}个词)", 
                type="primary",
                disabled=not (selected_model_info.get("api_key") and words_list)
            )

    st.markdown("---")
    
     # --- 结果显示区 ---
    if analyze_button and words_list and selected_model_info.get("api_key"):
        status_placeholder = st.empty()
        status_placeholder.info(f"正在使用并发处理 **{len(words_list)}** 个词语，请稍候...")
        
        # 运行并发批量处理
        results = process_batch(words_list, selected_model_info, max_workers=5)
        
        status_placeholder.empty()
        st.success(f'**批量分析完成**：共处理 **{len(words_list)}** 个词语。')
        
        # 迭代并显示每个词语的结果
        for result in results:
            word = result["word"]
            error = result["error"]
            scores_all = result["scores_all"]
            predicted_pos = result["predicted_pos"]
            explanation = result["explanation"]
            raw_text = result["raw_text"]
            
            st.markdown(f"## 🔎 词语分析结果： 「{word}」")

            if error:
                st.error(f"分析失败: {error}")
                with st.expander("原始响应", expanded=False):
                    st.code(raw_text, language="text")
                st.markdown("---")
                continue

            membership = calculate_membership(scores_all)
            final_membership = membership.get(predicted_pos, 0)
            
            st.info(f'**预测词类**： **【{predicted_pos}】**，隶属度为 **{final_membership:.4f}**')
            
            col_results_1, col_results_2 = st.columns(2)
            
            with col_results_1:
                st.subheader("💡 模型详细推理过程")
                st.markdown(explanation)
                st.markdown("---")
                
                st.subheader("📊 词类隶属度雷达图")
                # 只显示名词、动词、名动词的隶属度
                plot_radar_chart_streamlit(membership, f"「{word}」的词类隶属度分布")
                
            with col_results_2:
                st.subheader("📋 各词类详细得分")
                
                pos_total_scores = {pos: sum(scores_all[pos].values()) for pos in RULE_SETS.keys()}
                sorted_pos_names = sorted(pos_total_scores.keys(), key=lambda pos: pos_total_scores[pos], reverse=True)
                
                for pos in sorted_pos_names:
                    total_score = pos_total_scores[pos]
                    max_rule = max(scores_all[pos].items(), key=lambda x: x[1], default=("无", 0))
                    
                    with st.expander(f"**{pos}** (总分: {total_score}, 最高分规则: {max_rule[0]} - {max_rule[1]}分)"):
                        rule_data = []
                        for rule in RULE_SETS[pos]:
                            rule_score = scores_all[pos][rule["name"]]
                            rule_data.append({
                                "规则代码": rule["name"],
                                "规则描述": rule["desc"],
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
                    st.code(raw_text, language="json")

            st.markdown("---") # 结果分隔线
            
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

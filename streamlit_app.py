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
# 1. 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测 ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.dataframe {font-size: 12px;}
.stApp > div:first-child { padding-top: 2rem; }
/* 优化代码块显示，防止过高 */
.stCode { max-height: 300px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. 模型配置 (OpenAI 兼容协议)
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.0, "stream": True
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.0, "stream": True
        },
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.0, "stream": True
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
            "parameters": {"max_tokens": 4096, "temperature": 0.0, "result_format": "message", "incremental_output": True},
        },
    },
}

MODEL_OPTIONS = {
    "DeepSeek Chat": {"provider": "deepseek", "model": "deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "env_var": "DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o-mini": {"provider": "openai", "model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY"), "env_var": "OPENAI_API_KEY"},
    "Moonshot (Kimi)": {"provider": "moonshot", "model": "moonshot-v1-32k", "api_key": os.getenv("MOONSHOT_API_KEY"), "env_var": "MOONSHOT_API_KEY"},
    "Qwen (通义千问)": {"provider": "qwen", "model": "qwen-max", "api_key": os.getenv("QWEN_API_KEY"), "env_var": "QWEN_API_KEY"},
}

AVAILABLE_MODEL_OPTIONS = {name: info for name, info in MODEL_OPTIONS.items() if info["api_key"]}
if not AVAILABLE_MODEL_OPTIONS: AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS

# ===============================
# 3. 规则定义
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
        {"name": "N8_不能作补语/一般不作状语", "desc": "不能作补语，并且一般不能做状语", "match_score": 10, "mismatch_score": 0},
    ],
    "动词": [
        {"name": "V1_可受否定'不/没有'修饰", "desc": "可以受否定副词'不'或'没有'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "V2_可后附/插入时体助词", "desc": "可以后附或中间插入时体助词'着/了/过'", "match_score": 10, "mismatch_score": 0},
        {"name": "V3_可带真宾语", "desc": "可以带真宾语，或通过介词引导论元", "match_score": 20, "mismatch_score": 0},
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "V5_可有重叠/正反重叠形式", "desc": "可以有'VV, V一V'等形式", "match_score": 10, "mismatch_score": 0},
        {"name": "V6_可做谓语或谓语核心", "desc": "可以做谓语或谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "V7_不能作状语修饰动词性成分", "desc": "不能作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "V8_可作'怎么/怎样'提问", "desc": "可以跟在'怎么/怎样'之后提问", "match_score": 10, "mismatch_score": 0},
        {"name": "V9_不能跟在'多/多么'之后", "desc": "不能跟在'多'之后对性质提问", "match_score": 10, "mismatch_score": -10},
    ],
    "名动词": [
        {"name": "NV1_可被否定且肯定形式-1", "desc": "可以用'不/没有'否定", "match_score": 10, "mismatch_score": -10},
        {"name": "NV2_可附时体助词", "desc": "可以后附时体助词'着/了/过'", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受很修饰", "desc": "可以带真宾语，并且不能受'很'修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "有重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分", "desc": "既可以作谓语，又可以作主语或宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词修饰", "desc": "可以修饰名词或者受名词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在怎么/怎样之后", "desc": "可以跟在'怎么/怎样'之后提问", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在多/多么之后", "desc": "不能跟在'多/多么'之后", "match_score": 10, "mismatch_score": -10},
        {"name": "NV10_可后附方位词", "desc": "可以后附方位词构成处所结构", "match_score": 10, "mismatch_score": 0},
    ]
}

# ===============================
# 4. 核心工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """提取API响应文本，兼容多种格式"""
    if not isinstance(resp_json, dict): return ""
    try:
        # Qwen
        if "output" in resp_json and "text" in resp_json["output"]: return resp_json["output"]["text"]
        # OpenAI/Compatible
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]: return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception: return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """强力JSON提取，优先代码块，其次大括号，提取失败返回None"""
    if not text: return None, ""
    json_str = ""
    # 策略1: Markdown代码块
    code_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_match: json_str = code_match.group(1).strip()
    # 策略2: 最外层大括号
    if not json_str:
        match = re.search(r"(\{.*\})", text.strip(), re.DOTALL)
        if match: json_str = match.group(1).strip()
    
    if not json_str: return None, text
    try:
        return json.loads(json_str), json_str
    except:
        return None, text

def map_to_allowed_score(rule: dict, raw_val) -> int:
    match, mismatch = rule["match_score"], rule["mismatch_score"]
    if isinstance(raw_val, bool): return match if raw_val else mismatch
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"): return match
        return mismatch
    return mismatch

def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    membership = {}
    for pos, scores in scores_all.items():
        total = sum(scores.values())
        membership[pos] = max(-1.0, min(1.0, total / 100))
    return membership

def get_top_10_positions(membership: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(membership.items(), key=lambda x: x[1], reverse=True)[:10]

# ===============================
# 5. API 调用 (流式 + 超时保护)
# ===============================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    if not _api_key: return False, {"error": "API Key缺失"}, "Key未设置"
    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    
    full_content = ""
    try:
        # 设置 stream=True 和 60s 连接超时
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
                        delta_text = chunk["choices"][0].get("delta", {}).get("content", "")
                    elif "output" in chunk: # Qwen
                        if "choices" in chunk["output"]:
                            delta_text = chunk["output"]["choices"][0].get("message", {}).get("content", "")
                        elif "text" in chunk["output"]:
                            delta_text = chunk["output"]["text"]
                    if delta_text: full_content += delta_text
                except: continue
        
        if not full_content: return False, {"error": "空响应"}, "空响应"
        return True, {"choices": [{"message": {"content": full_content}}], "output": {"text": full_content}}, ""
    except Exception as e:
        return False, {"error": str(e)}, str(e)

# ===============================
# 6. 单个词分析逻辑 (Prompt 深度优化版)
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
# 7. 批量处理逻辑 (实时保存 + 增量更新)
# ===============================
def process_batch(df, model_info, col_name):
    """
    核心修改：
    1. 使用 'history_database.csv' 作为持久化存储。
    2. 优先读取 CSV 跳过已处理词汇。
    3. 每处理一个词，追加写入 CSV。
    """
    db_file = "history_database.csv"
    output = io.BytesIO()
    
    # A. 读取历史记录建立缓存
    history_cache = {}
    if os.path.exists(db_file):
        try:
            # 强制按字符串读取，避免数字/文本混淆
            hist_df = pd.read_csv(db_file, dtype=str)
            for _, row in hist_df.iterrows():
                if "词语" in row and pd.notna(row["词语"]):
                    history_cache[str(row["词语"]).strip()] = row.to_dict()
            st.info(f"📚 已加载本地历史记录 {len(history_cache)} 条，将自动跳过这些词。")
        except Exception as e:
            st.warning(f"历史文件读取失败，将重新分析: {e}")

    # B. 准备进度条
    total = len(df)
    bar = st.progress(0)
    status = st.empty()
    
    final_rows = []
    
    # C. 开始循环
    for i, row_data in df.iterrows():
        word = str(row_data[col_name]).strip()
        
        # 1. 检查缓存
        if word in history_cache:
            status.text(f"♻️ [跳过] {word} (已在历史记录)")
            final_rows.append(history_cache[word])
            # 小延时让界面刷新
            time.sleep(0.01) 
            bar.progress((i + 1) / total)
            continue
            
        # 2. 不在缓存，调用 API (带重试)
        status.text(f"🚀 [分析中] {word} ({i + 1}/{total})")
        
        retries = 3
        success = False
        scores, raw, pred, expl = {}, "", "请求失败", "多次重试失败"
        
        for attempt in range(retries):
            try:
                scores, raw, pred, expl = ask_model_for_pos_and_scores(
                    word, model_info["provider"], model_info["model"], model_info["api_key"]
                )
                # 只要 raw 不为空就算有响应
                if raw:
                    success = True
                    break
                time.sleep(2)
            except:
                time.sleep(2)
        
        # 3. 计算结果
        if success and scores:
            mem = calculate_membership(scores)
            v = mem.get("动词", 0.0)
            n = mem.get("名词", 0.0)
            nv = mem.get("名动词", 0.0)
        else:
            v, n, nv = 0.0, 0.0, 0.0
            
        # 4. 构造新行
        new_row = {
            "词语": word,
            "动词": v, "名词": n, "名动词": nv,
            "差值/距离": round(abs(v - n), 4),
            "原始响应": expl if len(expl) > 5 else raw, # 确保推理不丢失
            "_predicted_pos": pred
        }
        final_rows.append(new_row)
        
        # 5. 【核心】实时追加写入 CSV
        try:
            temp_df = pd.DataFrame([new_row])
            # 如果文件不存在则写表头，存在则不写表头直接追加
            write_header = not os.path.exists(db_file)
            temp_df.to_csv(db_file, mode='a', header=write_header, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"写入失败: {e}")
            
        # 6. 防封号延时
        time.sleep(1)
        bar.progress((i + 1) / total)

    # D. 循环结束，生成漂亮的 Excel
    status.success("✅ 全部完成！")
    
    if not final_rows: return None
    
    res_df = pd.DataFrame(final_rows)
    # 导出
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        cols = ["词语", "动词", "名词", "名动词", "差值/距离", "原始响应"]
        # 确保列存在
        valid_cols = [c for c in cols if c in res_df.columns]
        res_df[valid_cols].to_excel(writer, index=False, sheet_name='结果')
        # 标黄
        try:
            ws = writer.sheets['结果']
            fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            for idx, r in enumerate(final_rows):
                # 兼容从CSV读取的数据（可能是字符串）和刚生成的数据
                p = str(r.get("_predicted_pos", ""))
                target = None
                if "动词" in p: target = 2
                elif "名词" in p: target = 3
                elif "名动词" in p: target = 4
                if target: ws.cell(row=idx+2, column=target).fill = fill
        except: pass
        
    return output.getvalue()

# ===============================
# 8. 主程序 UI
# ===============================
def main():
    st.title("📰 汉语词类隶属度检测 (批量旗舰版)")
    
    # 配置区
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            if not AVAILABLE_MODEL_OPTIONS:
                st.error("❌ 未检测到 API Key，请配置环境变量。")
                info = {"api_key": None}
            else:
                name = st.selectbox("选择模型", list(AVAILABLE_MODEL_OPTIONS.keys()))
                info = AVAILABLE_MODEL_OPTIONS[name]
        with c2:
            st.write("")
            if st.button("连接测试"):
                ok, _, msg = call_llm_api_cached(info["provider"], info["model"], info["api_key"], [{"role":"user","content":"hi"}], 5)
                if ok: st.success("通畅")
                else: st.error(msg)

    st.markdown("---")
    
    t1, t2 = st.tabs(["🔍 单个词分析", "📂 批量全自动处理"])
    
    # --- Tab 1: 单个 ---
    with t1:
        w = st.text_input("输入词语", key="single_w")
        if st.button("分析", disabled=not (w and info["api_key"])):
            with st.spinner("思考中..."):
                scores, raw, pred, expl = ask_model_for_pos_and_scores(w, info["provider"], info["model"], info["api_key"])
                if scores:
                    mem = calculate_membership(scores)
                    st.success(f"结果: {pred}")
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.table(pd.DataFrame(get_top_10_positions(mem), columns=["词类","隶属度"]))
                        fig = go.Figure(go.Scatterpolar(r=list(mem.values())+[list(mem.values())[0]], theta=list(mem.keys())+[list(mem.keys())[0]], fill='toself'))
                        st.plotly_chart(fig, use_container_width=True)
                    with c_b:
                        st.info(expl)
                        with st.expander("原始 JSON"): st.code(raw)

    # --- Tab 2: 批量 ---
    with t2:
        st.info("上传 Excel (需含'词语'列)。程序会自动保存进度到 `history_database.csv`，中断后重跑即可自动续传。")
        
        up = st.file_uploader("上传 Excel", type=["xlsx"])
        
        if up and info["api_key"]:
            try:
                df = pd.read_excel(up)
                target = next((c for c in df.columns if "词" in str(c) or "word" in str(c).lower()), None)
                
                if target:
                    if st.button("🚀 开始批量 (支持断点续传)"):
                        res = process_batch(df, info, target)
                        if res:
                            st.download_button("📥 下载最终结果 (Excel)", res, "final_result.xlsx")
                else:
                    st.error("未找到包含'词'的列")
            except Exception as e:
                st.error(f"文件错误: {e}")

        # --- 历史数据管理区 ---
        st.markdown("---")
        st.subheader("💾 数据保险箱")
        db = "history_database.csv"
        if os.path.exists(db):
            try:
                hist = pd.read_csv(db)
                st.write(f"当前已安全保存 **{len(hist)}** 条数据。")
                c_d1, c_d2 = st.columns([1, 4])
                with c_d1:
                    st.download_button("📥 下载历史记录 (CSV)", hist.to_csv(index=False).encode('utf-8-sig'), "history_database.csv", "text/csv")
                with c_d2:
                    if st.button("🗑️ 清空历史 (重新开始)"):
                        os.remove(db)
                        st.rerun()
                with st.expander("预览数据"):
                    st.dataframe(hist)
            except:
                st.error("历史文件读取失败，可能正在写入中，请稍后刷新。")

if __name__ == "__main__":
    main()

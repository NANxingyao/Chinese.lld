import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import io
import time
import traceback
from typing import Tuple, Dict, Any, List
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# ==========================================
# 1. 页面配置与样式
# ==========================================
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
[data-testid="stSidebar"] { display: none !important; }
.stApp > div:first-child { padding-top: 2rem; }
.stCode { max-height: 400px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 模型配置 (OpenAI 兼容协议)
# ==========================================
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

# ==========================================
# 3. 语言学规则定义
# ==========================================
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
        {"name": "V4_程度副词与带宾语的关系", "desc": "不能受'很'修饰，或能同时受'很'修饰并带宾语", "match_score": 10, "mismatch_score": -10},
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

# ==========================================
# 4. 关键工具函数 (增强提取与计算)
# ==========================================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """提取API响应中的文本内容，兼容多种API格式"""
    if not isinstance(resp_json, dict): return ""
    try:
        # Qwen Native
        if "output" in resp_json and "text" in resp_json["output"]: return resp_json["output"]["text"]
        # OpenAI Compatible
        if "choices" in resp_json and len(resp_json["choices"]) > 0:
            choice = resp_json["choices"][0]
            if "message" in choice and "content" in choice["message"]: return choice["message"]["content"]
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception:
        return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """
    强力提取器：
    1. 优先提取 Markdown ```json 代码块
    2. 其次提取最外层 {}
    3. 失败则返回 None
    """
    if not text: return None, ""
    
    json_str = ""
    # 策略 1: 找代码块
    code_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
    
    # 策略 2: 找大括号
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
        # 模糊匹配：只要包含规则代码（如N1）即可
        if r_norm in k_norm or k_norm in r_norm: return r["name"]
    return None

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

# ==========================================
# 5. API 调用 (流式+重试机制)
# ==========================================
def call_llm_api_cached(_provider, _model, _api_key, messages, max_tokens=4096, temperature=0.0):
    if not _api_key: return False, {"error": "API Key缺失"}, "请先配置 Key"
    cfg = MODEL_CONFIGS[_provider]
    url = f"{cfg['base_url'].rstrip('/')}{cfg['endpoint']}"
    headers = cfg["headers"](_api_key)
    payload = cfg["payload"](_model, messages, max_tokens=max_tokens, temperature=temperature)
    
    full_content = ""
    try:
        # 设置 stream=True 和 60秒连接超时，防止200条时网络波动
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
            if response.status_code != 200:
                return False, {"error": f"API Error {response.status_code}"}, response.text
            
            for line in response.iter_lines():
                if not line: continue
                line_text = line.decode('utf-8').strip()
                if line_text.startswith("data:"): json_str = line_text[5:].strip()
                else: json_str = line_text
                
                if json_str == "[DONE]": break
                try:
                    chunk = json.loads(json_str)
                    delta_text = ""
                    # 兼容不同厂商的流式格式
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta_text = chunk["choices"][0].get("delta", {}).get("content", "")
                    elif "output" in chunk: # Qwen Native
                        if "choices" in chunk["output"]:
                            delta_text = chunk["output"]["choices"][0].get("message", {}).get("content", "")
                        elif "text" in chunk["output"]:
                            delta_text = chunk["output"]["text"]
                    if delta_text: full_content += delta_text
                except: continue
        
        if not full_content: return False, {"error": "空响应"}, "模型未返回内容"
        # 构造伪响应以便通用处理
        return True, {"choices": [{"message": {"content": full_content}}], "output": {"text": full_content}}, ""
    except Exception as e:
        return False, {"error": str(e)}, str(e)

# ==========================================
# 6. 单个词分析逻辑 (强Prompt约束)
# ==========================================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str):
    full_rules = {p: "\n".join([f"- {r['name']}: {r['desc']}" for r in rs]) for p, rs in RULE_SETS.items()}
    
    # 强制先输出详细文本，再输出JSON
    system = f"""你是一名汉语语言学专家。请对词语「{word}」进行严格的句法分析。

【输出要求】
1. 第一步：必须输出 Markdown 格式的详细推理过程。
   格式：
   ### 详细推理过程
   #### 名词
   - N1_...: 符合/不符合。理由... 例句...
   (请逐条分析所有规则，不能遗漏)
   ...（动词、名动词同理）

2. 第二步：分析结束后，输出一个 JSON 代码块：
```json
{{
  "explanation": "这里填入上面的详细推理全文，保留Markdown格式",
  "predicted_pos": "名词/动词/名动词",
  "scores": {{
    "名词": {{ "N1_...": true, ... }},
    "动词": {{ "V1_...": false, ... }},
    "名动词": {{ "NV1_...": true, ... }}
  }}
}}
""" ok, resp, err = call_llm_api_cached(provider, model, api_key, [ {"role": "system", "content": system}, {"role": "user", "content": f"分析「{word}」"} ])

if not ok: return {}, "", "失败", err

raw = extract_text_from_response(resp)
data, _ = extract_json_from_text(raw)

# 兜底：确保 explanation 不为空
if data:
    json_expl = data.get("explanation", "")
    # 如果JSON里的解释太短，说明模型可能偷懒了，强制用全文作为解释
    expl = json_expl if len(json_expl) > 50 else raw
    pred = data.get("predicted_pos", "未知")
    raw_scores = data.get("scores", {})
else:
    # JSON解析失败，保留全文，不算作完全失败
    expl = raw 
    pred = "未知"
    raw_scores = {}
    
# 分数转换
scores_out = {p: {} for p in RULE_SETS}
for pos, rules in RULE_SETS.items():
    s_map = raw_scores.get(pos, {})
    for r in rules:
        val = False
        for k, v in s_map.items():
            if normalize_key(k, [r]) == r["name"]:
                val = v
                break
        scores_out[pos][r["name"]] = map_to_allowed_score(r, val)
        
return scores_out, raw, pred, expl


def process_batch(df, model_info, col_name): """ 核心机制： 1. 实时追加写入 'history_database.csv'。 2. Try-Except 包裹整个单次循环，报错也继续。 3. 启动时读取 CSV，跳过已存在的词。 """ db_file = "history_database.csv" output = io.BytesIO()

# A. 读取历史，构建跳过列表
existing_data = {}
if os.path.exists(db_file):
    try:
        # 读成字符串防止类型错误
        hist_df = pd.read_csv(db_file, dtype=str)
        for _, row in hist_df.iterrows():
            if "词语" in row and pd.notna(row["词语"]):
                existing_data[str(row["词语"]).strip()] = row.to_dict()
        st.info(f"📚 已加载历史库：{len(existing_data)} 条。将自动跳过这些词，直接从断点处继续！")
    except:
        st.warning("历史库读取异常，本次将全部重试。")

total = len(df)
bar = st.progress(0)
status = st.empty()

# 内存中的结果，用于最后生成 Excel
final_rows = [] 

# B. 开始不可阻挡的循环
for i, row_data in df.iterrows():
    try:
        word = str(row_data[col_name]).strip()
        
        # --- 1. 检查缓存 (秒传) ---
        if word in existing_data:
            status.text(f"♻️ [跳过] {word} (已在库中)")
            
            # 从历史恢复数据结构
            cached = existing_data[word]
            # 简单类型转换回 float/str
            try:
                v = float(cached.get("动词", 0))
                n = float(cached.get("名词", 0))
                nv = float(cached.get("名动词", 0))
                d = float(cached.get("差值/距离", 0))
            except:
                v, n, nv, d = 0,0,0,0
            
            new_row = {
                "词语": word,
                "动词": v, "名词": n, "名动词": nv,
                "差值/距离": d,
                "原始响应": cached.get("原始响应", ""),
                "_predicted_pos": cached.get("_predicted_pos", "未知")
            }
            final_rows.append(new_row)
            
            time.sleep(0.01)
            bar.progress((i + 1) / total)
            continue

        # --- 2. 真实分析 (带重试) ---
        status.text(f"🚀 [正在分析] ({i+1}/{total}): {word}")
        
        # 无论如何重试 3 次
        retries = 3
        success = False
        scores, raw, pred, expl = {}, "", "请求失败", "多次重试无果"
        
        for attempt in range(retries):
            try:
                scores, raw, pred, expl = ask_model_for_pos_and_scores(
                    word, model_info["provider"], model_info["model"], model_info["api_key"]
                )
                # 只要 raw 不为空，就算拿到东西了
                if raw:
                    success = True
                    break
                time.sleep(2) # 失败休眠
            except Exception as e:
                print(f"Retry Error: {e}")
                time.sleep(2)
        
        # --- 3. 结果计算 ---
        if success and scores:
            mem = calculate_membership(scores)
            v = mem.get("动词", 0.0)
            n = mem.get("名词", 0.0)
            nv = mem.get("名动词", 0.0)
        else:
            v, n, nv = 0.0, 0.0, 0.0
        
        diff = round(abs(v - n), 4)
        
        # 兜底：如果 explanation 依然为空，强行填入 raw
        final_expl = expl if (expl and len(expl) > 5) else raw
        if not final_expl: final_expl = "API Error: No Response"

        new_row = {
            "词语": word,
            "动词": v, "名词": n, "名动词": nv,
            "差值/距离": diff,
            "原始响应": final_expl, # 这里绝对包含了完整推理
            "_predicted_pos": pred
        }
        final_rows.append(new_row)
        
        # --- 4. 实时落盘 (追加模式) ---
        # 这一步保证了即使下一秒断电，当前这个词也保存了
        try:
            temp_df = pd.DataFrame([new_row])
            write_hdr = not os.path.exists(db_file)
            temp_df.to_csv(db_file, mode='a', header=write_hdr, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"Disk Write Error: {e}")

        # --- 5. 强制停顿 (防封) ---
        time.sleep(1) 
        bar.progress((i + 1) / total)

    except Exception as e:
        # 捕捉一切未知异常，确保循环继续！
        print(f"CRITICAL ERROR on {word}: {e}")
        # 即使报错，也尝试往列表里加个空行，保持索引对齐(可选)
        time.sleep(1)
        continue

# C. 导出最终 Excel
status.success("✅ 全部完成！")

if not final_rows: return None

res_df = pd.DataFrame(final_rows)
try:
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        cols = ["词语", "动词", "名词", "名动词", "差值/距离", "原始响应"]
        valid_cols = [c for c in cols if c in res_df.columns]
        res_df[valid_cols].to_excel(writer, index=False, sheet_name='结果')
        
        # 标黄
        ws = writer.sheets['结果']
        fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        for idx, r in enumerate(final_rows):
            p = str(r.get("_predicted_pos", ""))
            t = None
            if "动词" in p: t = 2
            elif "名词" in p: t = 3
            elif "名动词" in p: t = 4
            if t: ws.cell(row=idx+2, column=t).fill = fill
except:
    pass
    
return output.getvalue()


def main(): st.title("📰 汉语词类隶属度检测 (批量旗舰版)")

with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        if not AVAILABLE_MODEL_OPTIONS:
            st.error("❌ 无 API Key")
            info = {"api_key": None}
        else:
            name = st.selectbox("选择模型", list(AVAILABLE_MODEL_OPTIONS.keys()))
            info = AVAILABLE_MODEL_OPTIONS[name]
    with c2:
        st.write("")
        if st.button("测试连接"):
            ok, _, msg = call_llm_api_cached(info["provider"], info["model"], info["api_key"], [{"role":"user","content":"hi"}], 5)
            if ok: st.success("成功")
            else: st.error(msg)

st.markdown("---")

t1, t2 = st.tabs(["🔍 单个分析", "📂 批量全自动处理"])

# Tab 1: 单个
with t1:
    w = st.text_input("输入词语", key="s_input")
    if st.button("分析", disabled=not (w and info["api_key"])):
        with st.spinner("分析中..."):
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
                    st.info("推理简述")
                    st.markdown(expl[:500]+"..." if len(expl)>500 else expl)
                    with st.expander("完整原始响应"): st.code(raw)

# Tab 2: 批量
with t2:
    st.info("💡 核心特性：自动断点续传 + 实时存盘。每跑一个词都会存入历史库，中断后刷新重跑即可接关。")
    
    up = st.file_uploader("上传 Excel", type=["xlsx"])
    
    if up and info["api_key"]:
        try:
            df = pd.read_excel(up)
            target = next((c for c in df.columns if "词" in str(c) or "word" in str(c).lower()), None)
            
            if target:
                if st.button("🚀 开始批量 (200+条稳定模式)"):
                    res = process_batch(df, info, target)
                    if res:
                        st.download_button("📥 下载本次结果 (Excel)", res, "final_result.xlsx")
            else:
                st.error("未找到'词'列")
        except Exception as e:
            st.error(f"文件错误: {e}")

    st.markdown("---")
    st.subheader("📚 历史记录数据库")
    st.caption("这里是总账本。即使程序现在崩溃，数据也全在这里面。")
    
    db = "history_database.csv"
    if os.path.exists(db):
        try:
            hist = pd.read_csv(db)
            st.write(f"当前数据库已安全保存 **{len(hist)}** 条记录。")
            c_d1, c_d2 = st.columns([1, 4])
            with c_d1:
                st.download_button("📥 下载所有历史 (CSV)", hist.to_csv(index=False).encode('utf-8-sig'), "history_database.csv", "text/csv")
            with c_d2:
                if st.button("🗑️ 清空历史 (慎重)"):
                    os.remove(db)
                    st.rerun()
            with st.expander("预览数据"):
                st.dataframe(hist.tail(10))
        except:
            st.warning("正在写入中，请稍后刷新...")
    else:
        st.info("暂无数据。")
if name == "main": main()


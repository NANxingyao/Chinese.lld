import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import time
import logging
import subprocess
import sys
import sqlite3
import hashlib
import traceback
import contextlib
from pathlib import Path
from typing import Tuple, Dict, Any, List

try:
    import fcntl
except ImportError:
    fcntl = None

SERVICE_MODE = "--worker" in sys.argv or "--supervisor" in sys.argv
BASE_DIR = Path(__file__).resolve().parent

# ===============================
# 日志
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "process_log.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

if not SERVICE_MODE:
    st.set_page_config(
        page_title="基于大语言模型的汉语隶属度检测划类平台",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items=None,
    )

CUSTOM_CSS = """
<style>
header {visibility:hidden;} footer {visibility:hidden;}
[data-testid="stSidebar"] {display:none !important;}
.stApp > div:first-child {padding-top:1rem;}
.stApp {background:linear-gradient(135deg,#f5f7fa 0%,#e4e9f2 100%);}
.block-container {padding-top:1.5rem;padding-bottom:2rem;max-width:95% !important;}
.title-header-card {background:linear-gradient(135deg,#0f2942 0%,#1e4d7b 40%,#2d6cb8 75%,#3d8bd6 100%);padding:2rem 2.5rem;border-radius:20px;margin-bottom:1.5rem;box-shadow:0 12px 40px rgba(15,41,66,.35);}
.title-header-card h1 {color:#fff !important;font-size:1.8rem !important;margin:0 !important;}
.title-header-card .subtitle {color:rgba(255,255,255,.8) !important;font-size:.95rem !important;margin-top:.5rem !important;font-style:italic;}
.title-header-card .badges {margin-top:1rem;display:flex;gap:.75rem;flex-wrap:wrap;}
.title-header-card .badge {background:rgba(255,255,255,.15);color:#fff;padding:.35rem .85rem;border-radius:20px;font-size:.8rem;border:1px solid rgba(255,255,255,.2);}
.section-title {display:flex;align-items:center;gap:.5rem;font-size:1.15rem !important;font-weight:600 !important;color:#1e3a5f !important;margin-bottom:1rem !important;padding-bottom:.5rem;border-bottom:2px solid #e2e8f0;}
.icon-dot {width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,#2d6cb8 0%,#3d8bd6 100%);flex-shrink:0;}
.stButton > button[kind="primary"] {background:linear-gradient(135deg,#1e4d7b 0%,#2d6cb8 50%,#3d8bd6 100%) !important;border:none !important;border-radius:12px !important;padding:.75rem 2rem !important;font-weight:700 !important;color:#fff !important;}
.stButton > button[kind="secondary"] {background:#fff !important;border:2px solid #cbd5e1 !important;border-radius:12px !important;font-weight:600 !important;color:#475569 !important;}
.result-success-card {background:linear-gradient(135deg,#ecfdf5 0%,#d1fae5 100%);border-radius:10px;padding:1rem 1.25rem;border-left:4px solid #10b981;margin:1rem 0;}
.info-highlight {background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border-radius:10px;padding:1rem 1.25rem;border-left:4px solid #3b82f6;margin:.75rem 0;}
.rank-card {display:flex;align-items:center;justify-content:space-between;padding:.85rem 1.25rem;border-radius:12px;margin-bottom:.6rem;background:#fff;border:1.5px solid #e2e8f0;}
.rank-num {width:32px;height:32px;border-radius:50%;background:#e2e8f0;color:#64748b;display:flex;align-items:center;justify-content:center;font-weight:700;}
.status-badge {display:inline-block;padding:.25rem .75rem;border-radius:20px;font-size:.8rem;font-weight:500;background:#d1fae5;color:#065f46;}
.running-status {display:flex;align-items:center;gap:.75rem;padding:.9rem 1.1rem;margin:.5rem 0 1rem;border-radius:12px;background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border:1px solid #bfdbfe;color:#1e40af;font-weight:600;}
.running-spinner {display:inline-block;width:18px;height:18px;min-width:18px;border:3px solid rgba(45,108,184,.22);border-top-color:#2d6cb8;border-radius:50%;animation:batch-spin .75s linear infinite;}
@keyframes batch-spin {to{transform:rotate(360deg);}}
.batch-detail {margin-top:.35rem;font-size:.9rem;font-weight:500;color:#475569;}
</style>
"""
if not SERVICE_MODE:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RULE_SETS = {
    "名词": [
        {"name":"N1_可受数量词修饰","desc":"可以受数量词修饰","match_score":10,"mismatch_score":0},
        {"name":"N2_不能受副词修饰","desc":"不能受副词修饰","match_score":20,"mismatch_score":-20},
        {"name":"N3_可作主宾语","desc":"可以做典型的主语或宾语","match_score":20,"mismatch_score":0},
        {"name":"N4_可作中心语或作定语","desc":"可以做中心语受其他名词修饰，或者作定语直接修饰其他名词","match_score":10,"mismatch_score":0},
        {"name":"N5_可后附的字结构","desc":"可以后附助词'的'构成'的'字结构","match_score":10,"mismatch_score":0},
        {"name":"N6_可后附方位词构处所","desc":"可以后附方位词构成处所结构","match_score":10,"mismatch_score":0},
        {"name":"N7_不能作谓语核心","desc":"不能做谓语或谓语核心","match_score":10,"mismatch_score":-10},
        {"name":"N8_不能作补语/一般不作状语","desc":"不能作补语，并且一般不能做状语直接修饰动词性成分","match_score":10,"mismatch_score":0},
    ],
    "动词": [
        {"name":"V1_可受否定'不/没有'修饰","desc":"可以受否定副词'不'或'没有'修饰","match_score":10,"mismatch_score":0},
        {"name":"V2_可后附/插入时体助词'着/了/过'","desc":"可以后附或中间插入时体助词'着/了/过'","match_score":10,"mismatch_score":0},
        {"name":"V3_可带真宾语或通过介词引导论元","desc":"可以带真宾语或通过介词引导论元","match_score":20,"mismatch_score":0},
        {"name":"V4_程度副词与带宾语的关系","desc":"不能受程度副词'很'修饰，或能同时受'很'修饰并带宾语","match_score":10,"mismatch_score":-10},
        {"name":"V5_可有重叠/正反重叠形式","desc":"可以有'VV, V一V, V了V, V不V, V了没有'等形式","match_score":10,"mismatch_score":0},
        {"name":"V6_可做谓语或谓语核心","desc":"可以做谓语或谓语核心","match_score":10,"mismatch_score":-10},
        {"name":"V7_不能作状语修饰动词性成分","desc":"不能作状语修饰动词性成分","match_score":10,"mismatch_score":0},
        {"name":"V8_可作'怎么/怎样'提问或'这么/这样/那么'回答","desc":"可以跟在'怎么/怎样'之后提问或跟在'这么/这样/那么'之后回答","match_score":10,"mismatch_score":0},
        {"name":"V9_不能跟在'多/多么'之后提问或表示感叹","desc":"不能跟在'多'之后对性质提问","match_score":10,"mismatch_score":-10},
    ],
    "名动词": [
        {"name":"NV1_可被\"不/没有\"否定且肯定形式-1","desc":"可以用\"不\"和\"没有\"来否定","match_score":10,"mismatch_score":-10},
        {"name":"NV2_可附时体助词或进入\"……了没有\"格式","desc":"可以后附时体助词\"着、了、过\"","match_score":10,"mismatch_score":-10},
        {"name":"NV3_可带真宾语且不受\"很\"修饰","desc":"可以带真宾语，并且不能受程度副词\"很\"等修饰","match_score":10,"mismatch_score":-10},
        {"name":"NV4_有重叠和正反重叠形式","desc":"可以有\"VV、V一V、V了V、V不V\"等重叠和正反重叠形式","match_score":10,"mismatch_score":0},
        {"name":"NV5_可作多种句法成分且可作形式动词宾语","desc":"既可以作谓语或谓语核心，又可以作主语或宾语","match_score":10,"mismatch_score":-10},
        {"name":"NV6_不能直接作状语","desc":"不能直接作状语修饰动词性成分","match_score":10,"mismatch_score":-10},
        {"name":"NV7_可修饰名词或受名词/数量词修饰","desc":"可以修饰名词或者受名词修饰，或者可以受数量词修饰","match_score":10,"mismatch_score":0},
        {"name":"NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后","desc":"可以跟在\"怎么、怎样\"之后提问","match_score":10,"mismatch_score":0},
        {"name":"NV9_不能跟在\"多/多么\"之后","desc":"不能跟在\"多\"之后对性质的程度进行提问","match_score":10,"mismatch_score":-10},
        {"name":"NV10_可后附方位词构成处所结构","desc":"可以后附方位词构成处所结构","match_score":10,"mismatch_score":0},
    ],
}

MODEL_CONFIGS = {
    "deepseek": {"base_url": os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com/v1"),"endpoint":"/chat/completions","headers":lambda k:{"Authorization":f"Bearer {k}","Content-Type":"application/json"},"payload":lambda m,msg,**kw:{"model":m,"messages":msg,"max_tokens":kw.get("max_tokens",4096),"temperature":kw.get("temperature",0.0),"stream":True}},
    "openai": {"base_url": os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1"),"endpoint":"/chat/completions","headers":lambda k:{"Authorization":f"Bearer {k}","Content-Type":"application/json"},"payload":lambda m,msg,**kw:{"model":m,"messages":msg,"max_tokens":kw.get("max_tokens",4096),"temperature":kw.get("temperature",0.0),"stream":True}},
    "gemini": {"base_url": os.getenv("GEMINI_BASE_URL","https://generativelanguage.googleapis.com/v1beta"),"endpoint":"/chat/completions","headers":lambda k:{"Authorization":f"Bearer {k}","Content-Type":"application/json"},"payload":lambda m,msg,**kw:{"model":m,"messages":msg,"max_tokens":kw.get("max_tokens",4096),"temperature":kw.get("temperature",0.0),"stream":True}},
    "moonshot": {"base_url":"https://api.moonshot.cn/v1","endpoint":"/chat/completions","headers":lambda k:{"Authorization":f"Bearer {k}","Content-Type":"application/json"},"payload":lambda m,msg,**kw:{"model":m,"messages":msg,"thinking":{"type":"disabled"},"temperature":0.6,"max_tokens":kw.get("max_tokens",8192),"stream":False}},
    "qwen": {"base_url":os.getenv("QWEN_BASE_URL","https://dashscope.aliyuncs.com/api/v1"),"endpoint":"/services/aigc/text-generation/generation","headers":lambda k:{"Authorization":f"Bearer {k}","Content-Type":"application/json","X-DashScope-SSE":"enable","Accept":"text/event-stream"},"payload":lambda m,msg,**kw:{"model":m,"input":{"messages":msg},"parameters":{"max_tokens":kw.get("max_tokens",4096),"temperature":kw.get("temperature",0.0),"result_format":"message","incremental_output":True}}},
}
MODEL_OPTIONS = {
    "DeepSeek Chat":{"provider":"deepseek","model":"deepseek-chat","api_key":os.getenv("DEEPSEEK_API_KEY"),"env_var":"DEEPSEEK_API_KEY"},
    "OpenAI GPT-4o（推荐）":{"provider":"openai","model":"gpt-4o-mini","api_key":os.getenv("OPENAI_API_KEY"),"env_var":"OPENAI_API_KEY"},
    "Google Gemini 1.5 Pro":{"provider":"gemini","model":"models/gemini-1.5-pro","api_key":os.getenv("GEMINI_API_KEY"),"env_var":"GEMINI_API_KEY"},
    "Google Gemini 1.5 Flash":{"provider":"gemini","model":"models/gemini-1.5-flash","api_key":os.getenv("GEMINI_API_KEY"),"env_var":"GEMINI_API_KEY"},
    "Moonshot（Kimi）":{"provider":"moonshot","model":"kimi-k2.6","api_key":os.getenv("MOONSHOT_API_KEY"),"env_var":"MOONSHOT_API_KEY"},
    "Qwen（通义千问）":{"provider":"qwen","model":"qwen-max","api_key":os.getenv("QWEN_API_KEY"),"env_var":"QWEN_API_KEY"},
}
AVAILABLE_MODEL_OPTIONS={k:v for k,v in MODEL_OPTIONS.items() if v["api_key"]} or MODEL_OPTIONS

# ===============================
# 基础文件/锁
# ===============================
def safe_name(text:str)->str:
    return re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]','_',str(text)) or "default_task"

def task_hash_from_upload(data:bytes,target_col:str,provider:str,model:str)->str:
    h=hashlib.sha256(); h.update(data); h.update(b"\0"+str(target_col).encode()); h.update(b"\0"+str(provider).encode()); h.update(b"\0"+str(model).encode()); return h.hexdigest()[:20]

def task_paths(project_code:str, task_hash:str):
    p=safe_name(project_code)
    stem=f"{p}_{task_hash}"
    return {
        "db":BASE_DIR/f"batch_{stem}.db",
        "csv":BASE_DIR/f"batch_history_{stem}.csv",
        "state":BASE_DIR/f"batch_job_{p}.json",
        "spec":BASE_DIR/f"batch_job_{p}.spec.json",
        "input":BASE_DIR/f"batch_input_{stem}.xlsx",
        "supervisor_pid":BASE_DIR/f"batch_job_{p}.supervisor.pid",
        "worker_pid":BASE_DIR/f"batch_job_{p}.worker.pid",
        "start_claim":BASE_DIR/f"batch_job_{p}.start.claim",
        "supervisor_lock":BASE_DIR/f"batch_job_{p}.supervisor.lock",
    }

@contextlib.contextmanager
def locked_file(path:Path, nonblocking=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"a+",encoding="utf-8") as fh:
        if fcntl is not None:
            flags=fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
            fcntl.flock(fh.fileno(),flags)
        try: yield fh
        finally:
            if fcntl is not None:
                try: fcntl.flock(fh.fileno(),fcntl.LOCK_UN)
                except Exception: pass

def atomic_json(path:Path,data:dict):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)

def load_json(path:Path):
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception as e: logger.error("读取状态失败：%s",e); return {}

def save_state(path:Path,**updates):
    try:
        with locked_file(Path(str(path)+".lock")):
            cur=load_json(path); cur.update(updates); cur["updated_at"]=time.strftime("%Y-%m-%d %H:%M:%S"); atomic_json(path,cur)
    except Exception as e: logger.error("写状态失败：%s",e)

def pid_alive(pid):
    try:
        pid=int(pid)
        if pid<=0: return False
        os.kill(pid,0); return True
    except Exception: return False

def terminate_pid(pid):
    try:
        pid=int(pid)
        if pid>0 and pid_alive(pid): os.kill(pid,15)
    except Exception: pass

def claim_start(claim:Path)->bool:
    try:
        fd=os.open(str(claim),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(fd,str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        return False
    except Exception: return False

def release_start_claim(claim:Path):
    try: claim.unlink(missing_ok=True)
    except Exception: pass

# ===============================
# SQLite：唯一结果源
# ===============================
RESULT_COLS=["序数","词语","动词","名词","名动词","差值/距离","预测词类","是否兼类","原始响应","时间戳"]

def db_connect(db:Path):
    conn=sqlite3.connect(str(db),timeout=60)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=FULL"); conn.execute("PRAGMA busy_timeout=60000"); return conn

def db_init(db:Path):
    db.parent.mkdir(parents=True,exist_ok=True)
    conn=db_connect(db)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS results(seq INTEGER PRIMARY KEY, word TEXT NOT NULL, verb REAL NOT NULL, noun REAL NOT NULL, nounverb REAL NOT NULL, distance REAL NOT NULL, predicted_pos TEXT NOT NULL, is_dual TEXT NOT NULL, raw_text TEXT NOT NULL, timestamp TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"); conn.commit()
    finally: conn.close()

def db_contiguous(db:Path):
    if not db.exists(): return 0
    conn=db_connect(db)
    try:
        rows=conn.execute("SELECT seq FROM results ORDER BY seq").fetchall(); expected=1
        for (seq,) in rows:
            if int(seq)!=expected: break
            expected+=1
        return expected-1
    finally: conn.close()

def db_dataframe(db:Path):
    if not db.exists(): return pd.DataFrame(columns=RESULT_COLS)
    conn=db_connect(db)
    try:
        df=pd.read_sql_query("SELECT seq AS 序数, word AS 词语, verb AS 动词, noun AS 名词, nounverb AS 名动词, distance AS 差值距离, predicted_pos AS 预测词类, is_dual AS 是否兼类, raw_text AS 原始响应, timestamp AS 时间戳 FROM results ORDER BY seq",conn)
        return df.rename(columns={"差值距离":"差值/距离"})
    finally: conn.close()

def export_csv(db:Path,csv:Path):
    df=db_dataframe(db); tmp=csv.with_suffix(csv.suffix+".tmp"); df.to_csv(tmp,index=False,encoding="utf-8-sig"); os.replace(tmp,csv)

def commit_result(db:Path,csv:Path,row_number:int,row:dict):
    db_init(db)
    lock=Path(str(db)+".write.lock")
    with locked_file(lock):
        conn=db_connect(db)
        try:
            conn.execute("BEGIN IMMEDIATE")
            current=db_contiguous(db)
            if row_number<=current:
                conn.rollback(); export_csv(db,csv); return True,"already_exists"
            if row_number!=current+1:
                conn.rollback(); return False,f"顺序冲突：期望第{current+1}条，收到第{row_number}条"
            conn.execute("INSERT INTO results(seq,word,verb,noun,nounverb,distance,predicted_pos,is_dual,raw_text,timestamp) VALUES(?,?,?,?,?,?,?,?,?,?)",(
                int(row_number),str(row["词语"]),float(row["动词"]),float(row["名词"]),float(row["名动词"]),float(row["差值/距离"]),str(row["预测词类"]),str(row["是否兼类"]),str(row["原始响应"]),str(row["时间戳"])))
            conn.commit(); export_csv(db,csv); return True,"written"
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            logger.error("提交结果失败：%s\n%s",e,traceback.format_exc()); return False,f"{type(e).__name__}: {e}"
        finally: conn.close()

def reset_task(paths:dict):
    state=load_json(paths["state"]); terminate_pid(state.get("worker_pid")); terminate_pid(state.get("supervisor_pid"))
    deadline=time.time()+5
    while time.time()<deadline and (pid_alive(state.get("worker_pid")) or pid_alive(state.get("supervisor_pid"))):
        time.sleep(0.2)
    for key in ("db","csv","state","spec","input","supervisor_pid","worker_pid","start_claim","supervisor_lock"):
        try: paths[key].unlink(missing_ok=True)
        except Exception: pass

# ===============================
# 文本解析与模型调用
# ===============================
def extract_text(resp:dict)->str:
    if not isinstance(resp,dict): return ""
    choices=resp.get("choices")
    if isinstance(choices,list) and choices:
        msg=choices[0].get("message",{}) or {}
        content=msg.get("content")
        if isinstance(content,str): return content
        delta=choices[0].get("delta",{}) or {}
        if isinstance(delta.get("content"),str): return delta["content"]
    output=resp.get("output")
    if isinstance(output,dict):
        if isinstance(output.get("text"),str): return output["text"]
        choices=output.get("choices")
        if isinstance(choices,list) and choices:
            msg=choices[0].get("message",{}) or {}
            if isinstance(msg.get("content"),str): return msg["content"]
    return ""

def parse_json_text(text:str):
    if not text: return None
    for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```",r"(\{.*\})"):
        for m in re.finditer(pattern,text,re.DOTALL):
            try:
                obj=json.loads(m.group(1));
                if isinstance(obj,dict) and ("scores" in obj or "predicted_pos" in obj): return obj
            except Exception: pass
    return None

def normalize_key(k,pos_rules):
    if not isinstance(k,str): return None
    clean=re.sub(r'[\s_]+','',k).upper()
    for r in pos_rules:
        if re.sub(r'[\s_]+','',r["name"]).upper()==clean: return r["name"]
    for r in pos_rules:
        code=re.match(r'^(NV\d+|N\d+|V\d+)',re.sub(r'[\s_]+','',r["name"]).upper())
        if code and (clean==code.group(1) or clean.startswith(code.group(1))): return r["name"]
    return None

def map_score(rule,val):
    if isinstance(val,bool): return rule["match_score"] if val else rule["mismatch_score"]
    if isinstance(val,str):
        s=val.strip().lower()
        if s in ("true","yes","是","符合","√"): return rule["match_score"]
        if s in ("false","no","否","不符合","×"): return rule["mismatch_score"]
    if isinstance(val,(int,float)):
        if int(val)==rule["match_score"]: return rule["match_score"]
        if int(val)==rule["mismatch_score"]: return rule["mismatch_score"]
    return rule["mismatch_score"]

def make_scores(raw):
    scores={pos:{r["name"]:0 for r in rules} for pos,rules in RULE_SETS.items()}
    for pos,rules in RULE_SETS.items():
        section=raw.get(pos,{}) if isinstance(raw,dict) else {}
        if isinstance(section,dict):
            for k,v in section.items():
                nk=normalize_key(k,rules)
                if nk:
                    rule=next(r for r in rules if r["name"]==nk); scores[pos][nk]=map_score(rule,v)
    return scores

def membership(scores): return {p:max(-1,min(1,sum(v.values())/100)) for p,v in scores.items()}

def call_llm(provider,model,key,messages,max_tokens=4096,temperature=0.0,show_ui=False):
    if not key: return False,{"error":"API Key 未提供"},"API Key 未提供"
    cfg=MODEL_CONFIGS[provider]; url=cfg["base_url"].rstrip("/")+"/"+cfg["endpoint"].lstrip("/")
    payload=cfg["payload"](model,messages,max_tokens=max_tokens,temperature=temperature)
    is_kimi=provider=="moonshot"
    last="未知错误"
    for attempt in range(3):
        try:
            with requests.post(url,headers=cfg["headers"](key),json=payload,stream=not is_kimi,timeout=120) as r:
                if r.status_code!=200:
                    try: detail=r.json()
                    except Exception: detail=r.text
                    last=f"HTTP {r.status_code}: {detail}"
                    if r.status_code in (400,401,403,404): return False,{"error":last},last
                    raise RuntimeError(last)
                if is_kimi:
                    body=r.json(); text=extract_text(body)
                    if text: return True,body,""
                    return False,{"error":f"200但content为空：{r.text[:1000]}"},f"200但content为空：{r.text[:1000]}"
                full=""
                for raw_line in r.iter_lines():
                    if not raw_line: continue
                    line=raw_line.decode("utf-8","ignore").strip()
                    if line.startswith("data:"): line=line[5:].strip()
                    if line=="[DONE]": break
                    try:
                        obj=json.loads(line); t=extract_text(obj)
                        if t: full+=t
                    except Exception: continue
                if full: return True,{"choices":[{"message":{"content":full}}]},""
                last="流式响应没有解析到 content"
        except Exception as e:
            last=f"请求异常：{type(e).__name__}: {e}"; time.sleep(2**attempt)
    return False,{"error":last},last

def analyze_word(word,provider,model,key,show_ui=False):
    full_rules={p:"\n".join(f"- {r['name']}: {r['desc']}（符合:{r['match_score']}分，不符合:{r['mismatch_score']}分）" for r in rs) for p,rs in RULE_SETS.items()}
    system=f"""你是一名中文词法与语法专家。分析词语「{word}」在名词、动词、名动词三类中的表现。每条规则只输出true/false。\n\n【名词】\n{full_rules['名词']}\n【动词】\n{full_rules['动词']}\n【名动词】\n{full_rules['名动词']}\n\n输出合法JSON：{{\"explanation\":\"逐条给出判断依据和例句\",\"predicted_pos\":\"名词/动词/名动词\",\"is_dual_category\":true,\"scores\":{{\"名词\":{{...}},\"动词\":{{...}},\"名动词\":{{...}}}}}}"""
    user=f"严格分析「{word}」，scores只能是true/false，最后给出完整JSON。"
    ctx=st.spinner(f"正在调用大模型 ({model})…") if show_ui else contextlib.nullcontext()
    with ctx: ok,resp,err=call_llm(provider,model,key,[{"role":"system","content":system},{"role":"user","content":user}],show_ui=show_ui)
    if not ok: return {},f"调用失败: {err}","未知",f"失败: {err}",False
    raw=extract_text(resp); parsed=parse_json_text(raw)
    if not parsed: return {},raw,"未知","JSON解析失败",False
    scores=make_scores(parsed.get("scores",{})); return scores,raw,parsed.get("predicted_pos","未知"),parsed.get("explanation",""),bool(parsed.get("is_dual_category",False))

# ===============================
# 后台 Worker / Supervisor
# ===============================
def write_pid(path:Path,pid):
    try: path.write_text(str(pid),encoding="utf-8")
    except Exception: pass

def worker_loop(spec:dict,paths:dict):
    db=Path(paths["db"]); csv=Path(paths["csv"]); state=Path(paths["state"]); progress=state.with_suffix(".progress.json")
    db_init(db); total=len(spec["rows"]); start=db_contiguous(db)
    save_state(state,status="running",task_hash=spec["task_hash"],total_rows=total,next_row=start,completed_rows=start,current_row=max(start-1,0),current_word="",retry_count=0,error="")
    while start<total:
        idx=start; row_number=idx+1; word=str(spec["rows"][idx]).strip()
        if not word or word.lower()=="nan":
            # 空行也占据一个序号，保证严格连续；不调用模型。
            result={"词语":"","动词":0,"名词":0,"名动词":0,"差值/距离":0,"预测词类":"空值","是否兼类":"否","原始响应":"空值，跳过","时间戳":time.strftime("%Y-%m-%d %H:%M:%S")}
            while True:
                ok,msg=commit_result(db,csv,row_number,result)
                if ok: break
                time.sleep(2)
            start=row_number; atomic_json(progress,{"task_hash":spec["task_hash"],"next_row":start,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S")})
            save_state(state,status="running",next_row=start,completed_rows=start,current_row=idx,current_word="（空值）",retry_count=0,error="")
            continue
        success=False; retry=0; last=""
        while not success:
            retry+=1; save_state(state,status="retrying" if retry>1 else "running",next_row=idx,completed_rows=idx,current_row=idx,current_word=word,retry_count=retry,error=last)
            scores,raw,pred,exp,dual=analyze_word(word,spec["provider"],spec["model"],os.getenv(spec["env_var"],""),False)
            if scores: success=True; break
            last=exp or "模型没有返回有效结果"; wait=min(60,2**min(retry-1,5)); save_state(state,status="waiting_retry",next_row=idx,completed_rows=idx,current_row=idx,current_word=word,retry_count=retry,error=last,retry_in_seconds=wait); time.sleep(wait)
        mem=membership(scores); result={"词语":word,"动词":mem.get("动词",0),"名词":mem.get("名词",0),"名动词":mem.get("名动词",0),"差值/距离":round(abs(mem.get("动词",0)-mem.get("名词",0)),4),"预测词类":pred,"是否兼类":"是" if dual else "否","原始响应":raw,"时间戳":time.strftime("%Y-%m-%d %H:%M:%S")}
        save_try=0
        while True:
            save_try+=1; save_state(state,status="saving",next_row=idx,completed_rows=idx,current_row=idx,current_word=word,retry_count=save_try,error="")
            ok,msg=commit_result(db,csv,row_number,result)
            if ok: break
            wait=min(30,2**min(save_try-1,4)); save_state(state,status="waiting_save_retry",next_row=idx,completed_rows=idx,current_row=idx,current_word=word,retry_count=save_try,error=msg,retry_in_seconds=wait); time.sleep(wait)
        start=row_number; atomic_json(progress,{"task_hash":spec["task_hash"],"next_row":start,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S")}); save_state(state,status="running",next_row=start,completed_rows=start,current_row=idx,current_word=word,retry_count=0,error="")
    progress.unlink(missing_ok=True); save_state(state,status="completed",next_row=total,completed_rows=total,current_word="",retry_count=0,error="")

def worker_entry(state_file:Path):
    spec_file=Path(str(state_file)+".spec.json"); spec=load_json(spec_file)
    if not spec.get("task_hash"): raise RuntimeError("缺少任务指纹，拒绝运行旧任务")
    paths={"db":spec["db"],"csv":spec["csv"],"state":str(state_file)}
    try:
        write_pid(Path(spec["worker_pid"]),os.getpid()); worker_loop(spec,paths)
    finally:
        try: Path(spec["worker_pid"]).unlink(missing_ok=True)
        except Exception: pass

def supervisor_entry(state_file:Path):
    spec=load_json(Path(str(state_file)+".spec.json")); lock=Path(spec["supervisor_lock"])
    try:
        with locked_file(lock,nonblocking=True):
            write_pid(Path(spec["supervisor_pid"]),os.getpid()); save_state(state_file,status="running",supervisor_pid=os.getpid())
            while True:
                state=load_json(state_file)
                if state.get("status")=="completed": return
                worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",str(state_file)],cwd=str(BASE_DIR),start_new_session=True)
                write_pid(Path(spec["worker_pid"]),worker.pid); save_state(state_file,status="running",supervisor_pid=os.getpid(),worker_pid=worker.pid)
                rc=worker.wait()
                state=load_json(state_file)
                if state.get("status")=="completed": return
                delay=min(30,max(2,2**min(int(state.get("supervisor_restarts",0)),4)))
                n=int(state.get("supervisor_restarts",0))+1
                save_state(state_file,status="supervisor_restarting",supervisor_pid=os.getpid(),worker_pid=None,supervisor_restarts=n,error=f"Worker退出（返回码{rc}），{delay}秒后自动恢复")
                time.sleep(delay)
    finally:
        try: Path(spec.get("supervisor_pid","")).unlink(missing_ok=True)
        except Exception: pass
        try: Path(spec.get("worker_pid","")).unlink(missing_ok=True)
        except Exception: pass

def auto_recover(paths:dict, task_hash:str):
    """页面刷新时自动恢复已经存在但 Supervisor 已退出的任务。"""
    state=load_json(paths["state"])
    if not state or state.get("task_hash")!=task_hash:
        return state
    status=state.get("status","")
    if status in ("completed","failed"):
        return state
    if pid_alive(state.get("supervisor_pid")):
        return state
    # 避免两个页面/两个 fragment 同时启动 Supervisor。
    if not claim_start(paths["start_claim"]):
        return state
    try:
        # 重新确认一次，防止在等待 claim 的过程中别人已经启动。
        state=load_json(paths["state"])
        if pid_alive(state.get("supervisor_pid")):
            return state
        if not Path(paths["spec"]).exists():
            return state
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--supervisor",str(paths["state"])],cwd=str(BASE_DIR),start_new_session=True)
        write_pid(paths["supervisor_pid"],proc.pid)
        n=int(state.get("auto_recovery_count",0) or 0)+1
        save_state(paths["state"],status="starting",task_hash=task_hash,supervisor_pid=proc.pid,worker_pid=None,auto_recovery_count=n,error="检测到 Supervisor 已退出，已自动恢复任务")
        return load_json(paths["state"])
    except Exception as e:
        logger.exception("自动恢复 Supervisor 失败")
        save_state(paths["state"],status="supervisor_restarting",task_hash=task_hash,supervisor_pid=None,worker_pid=None,error=f"自动恢复失败：{type(e).__name__}: {e}")
        return load_json(paths["state"])
    finally:
        release_start_claim(paths["start_claim"])

def start_job(df,target_col,uploaded_file,project_code,provider,model,env_var):
    data=uploaded_file.getvalue(); th=task_hash_from_upload(data,target_col,provider,model); paths=task_paths(project_code,th); state=load_json(paths["state"])
    # 同一批次的新文件自动切换到全新结果空间。相同 task_hash 则续跑。
    old_hash=state.get("task_hash")
    if old_hash and old_hash!=th:
        reset_task(paths); state={}
    elif state and not old_hash:
        reset_task(paths); state={}
    if pid_alive(state.get("supervisor_pid")): return False,state,paths,th
    if not claim_start(paths["start_claim"]):
        return False,load_json(paths["state"]),paths,th
    try:
        input_file=paths["input"]; input_file.write_bytes(data)
        rows=["" if pd.isna(x) else str(x) for x in df[target_col].tolist()]
        spec={"task_hash":th,"input_file":str(input_file),"rows":rows,"provider":provider,"model":model,"env_var":env_var,"db":str(paths["db"]),"csv":str(paths["csv"]),"state":str(paths["state"]),"spec":str(paths["spec"]),"supervisor_pid":str(paths["supervisor_pid"]),"worker_pid":str(paths["worker_pid"]),"supervisor_lock":str(paths["supervisor_lock"])}
        atomic_json(paths["spec"],spec); db_init(paths["db"]); start=db_contiguous(paths["db"])
        save_state(paths["state"],status="starting",task_hash=th,total_rows=len(rows),next_row=start,completed_rows=start,current_row=max(start-1,0),current_word="",retry_count=0,error="")
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--supervisor",str(paths["state"])],cwd=str(BASE_DIR),start_new_session=True)
        write_pid(paths["supervisor_pid"],proc.pid); save_state(paths["state"],status="starting",task_hash=th,supervisor_pid=proc.pid,worker_pid=None,error="守护任务已启动")
        return True,load_json(paths["state"]),paths,th
    finally: release_start_claim(paths["start_claim"])

# ===============================
# UI
# ===============================
def render_radar(m,title):
    if not m: return
    cats=list(m.keys()); vals=list(m.values()); cats.append(cats[0]); vals.append(vals[0])
    fig=go.Figure(data=[go.Scatterpolar(r=vals,theta=cats,fill="toself")]); fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[min(min(vals),-0.1),max(max(vals),1)])),showlegend=False,title=dict(text=title,x=.5)); st.plotly_chart(fig,use_container_width=True)

def main():
    st.markdown('<div class="title-header-card"><h1>基于大语言模型的汉语词类隶属度检测划类平台</h1><div class="subtitle">Chinese Membership Detection and Classification Platform Based on Large Language Models (LLMs)</div><div class="badges"><span class="badge">多模型支持</span><span class="badge">隶属度分析</span><span class="badge">批量处理</span><span class="badge">严格顺序零重复</span><span class="badge">自动恢复</span></div></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns([5,3,2])
    with c1:
        st.markdown('<div class="section-title"><span class="icon-dot"></span>模型设置</div>',unsafe_allow_html=True); name=st.selectbox("选择大模型",list(AVAILABLE_MODEL_OPTIONS.keys()),key="model_select"); info=AVAILABLE_MODEL_OPTIONS[name]; st.markdown('<span class="status-badge">● 已配置</span>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title"><span class="icon-dot"></span>实验配置</div>',unsafe_allow_html=True); project=st.text_input("实验批次码",value=st.session_state.get("project_code","default_task")); st.session_state.project_code=project
    with c3:
        st.markdown('<div class="section-title"><span class="icon-dot"></span>连接测试</div>',unsafe_allow_html=True)
        if st.button("测试模型链接",type="secondary",use_container_width=True):
            ok,_,err=call_llm(info["provider"],info["model"],info["api_key"],[{"role":"user","content":"请回复 pong"}],max_tokens=64)
            if ok:
                st.success("成功！")
            else:
                st.error(f"失败：{err}")
    tab1,tab2=st.tabs(["单个词语详细分析","Excel 批量处理"])
    with tab1:
        word=st.text_input("请输入汉语词语",placeholder="例如：苹果、跑、美丽…");
        if st.button("开始分析",type="primary",disabled=not(word and info["api_key"])):
            scores,raw,pred,exp,dual=analyze_word(word,info["provider"],info["model"],info["api_key"],True)
            if scores:
                m=membership(scores); st.markdown(f'<div class="result-success-card">词语「<strong>{word}</strong>」最可能的词类：<strong>{pred}</strong>，隶属度 {m.get(pred,0):.4f}；兼类：{"是" if dual else "否"}</div>',unsafe_allow_html=True); render_radar(m,f"「{word}」隶属度")
                with st.expander("查看模型原始响应"): st.code(raw,language="text")
            else: st.error(exp)
    with tab2:
        uploaded=st.file_uploader("选择 Excel 文件",type=["xlsx","xls"])
        if uploaded:
            df=pd.read_excel(uploaded); target=next((c for c in df.columns if "词" in str(c) or "word" in str(c).lower()),None)
            if target:
                data=uploaded.getvalue(); th=task_hash_from_upload(data,target,info["provider"],info["model"]); paths=task_paths(project,th); state=load_json(paths["state"])
                # 新任务只显示自己的结果，绝不读取旧项目下的其他任务。
                st.markdown(f'<div class="info-highlight">识别到目标列：<code>{target}</code> · 总数：<strong>{len(df)}</strong> · 任务指纹：<code>{th}</code></div>',unsafe_allow_html=True)
                if state and state.get("task_hash")!=th:
                    state={}
                status_box=st.empty(); prog=st.progress(0); table_box=st.empty(); metric=st.empty()
                a,b=st.columns([1,2])
                with a:
                    if st.button("重新开始当前任务",type="secondary",use_container_width=True,key=f"reset_{th}"):
                        reset_task(paths); st.success("当前任务已清空，将从第1条重新开始。"); st.rerun()
                with b:
                    active=pid_alive(state.get("supervisor_pid")) and state.get("status") not in ("completed","failed")
                    if st.button("开始处理 / 继续任务",type="primary",use_container_width=True,key=f"start_{th}",disabled=active):
                        if not info["api_key"]: st.error("请配置 API Key")
                        else:
                            ok,newstate,_,_=start_job(df,target,uploaded,project,info["provider"],info["model"],info["env_var"])
                            st.success("后台任务已启动") if ok else st.info("任务已经在运行，无需重复启动")
                            time.sleep(.3); st.rerun()
                def live():
                    auto_recover(paths, th)
                    s=load_json(paths["state"]); total=int(s.get("total_rows",len(df)) or len(df)); done=int(s.get("completed_rows",db_contiguous(paths["db"])) or 0); stt=s.get("status",""); cur=int(s.get("current_row",max(done-1,0)) or 0); word2=s.get("current_word",""); retry=int(s.get("retry_count",0) or 0); err=s.get("error","")
                    prog.progress(done/total if total else 0); metric.metric("已完成",f"{done}/{total}")
                    if stt in {"starting","running","retrying","waiting_retry","saving","waiting_save_retry","supervisor_restarting"}:
                        label={"retrying":"当前词语失败，自动重试…","waiting_retry":"等待重试…","saving":"正在保存…","waiting_save_retry":"保存失败，自动重试…","supervisor_restarting":"Worker 已退出，自动恢复…"}.get(stt,"任务正在运行…")
                        safe_err = re.sub(r"[<>]", "", str(err))[:500] if err else ""
                        err_html = f'<div class="batch-detail">最近状态：{safe_err}</div>' if safe_err else ""
                        status_html = (f'<div class="running-status"><span class="running-spinner"></span><div><b>{label}</b>'
                                       f'<div class="batch-detail">当前第 {min(cur+1,total)}/{total} 行 · 已完成 {done}/{total} · 当前词语「{word2}」 · 重试 {retry} 次</div>'
                                       f'{err_html}</div></div>')
                        status_box.markdown(status_html, unsafe_allow_html=True)
                    elif stt=="completed": status_box.success(f"🎉 任务完成，共 {total} 条")
                    elif stt=="failed": status_box.error(f"❌ 任务停止：{err or '未知错误'}")
                    else: status_box.info("等待开始任务")
                    d=db_dataframe(paths["db"]); table_box.dataframe(d,use_container_width=True,height=360) if not d.empty else table_box.info("暂无已保存结果")
                if hasattr(st,"fragment"):
                    @st.fragment(run_every="2s")
                    def _live(): live()
                    _live()
                else: live()
            else: st.error("未识别到词语列")

if __name__=="__main__":
    if "--worker" in sys.argv: worker_entry(Path(sys.argv[sys.argv.index("--worker")+1]))
    elif "--supervisor" in sys.argv: supervisor_entry(Path(sys.argv[sys.argv.index("--supervisor")+1]))
    else: main()

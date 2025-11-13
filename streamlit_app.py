import streamlit as st
import requests
import json
import re
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests, json
from typing import Tuple, Dict, Any

# ===============================
# 模型与 API Key 配置区
# ===============================
MODEL_CONFIGS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_key": "sk-1f346646d29947d0a5e29dbaa37476b8"   
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key": "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"
    },
    "Moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "api_key": "sk-moonshot-xxxx"
    },
    "GLM（智谱）": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
        "api_key": "9720beea6bff408ea6c26cd5d9ecf3b8.cRNTQiKRNaTieIiz"
    },
    "通义千问": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "api_key": "sk-qwen-xxxx"
    },
    "豆包": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "ep-20241106172028-vhdew",
        "api_key": "WmpRMlptTmxNRGM0TjJNMk5HUTBOR0ZtWVRsbU56TTNNakUyT0RVNU1EUQ=="
    },
}

# ===============================
# 通用模型配置（支持多模型自由切换）
# ===============================
import requests
import streamlit as st
from typing import Tuple

# 默认配置
MODEL_PROVIDER = "deepseek"   # 初始默认模型，可选: deepseek, openai, moonshot, glm, qwen, doubao
MODEL_NAME = "deepseek-chat"
API_KEY = "sk-1f346646d29947d0a5e29dbaa37476b8"

# 模型调用配置模板
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "model": "deepseek-chat",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False
        }
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "model": "gpt-4o-mini",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False
        }
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "model": "moonshot-v1-32k",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.0),
            "stream": False
        }
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "endpoint": "/chat/completions",
        "model": "glm-4",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.0),
            "stream": False
        }
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "model": "qwen-plus",
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "input": {"messages": messages},
            "parameters": {"temperature": kw.get("temperature", 0.0)}
        }
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "endpoint": "/chat/completions",
        "model": "doubao-pro-32k",  # 或 doubao-lite、doubao-1.5-pro 等
        "headers": lambda key: {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.0),
            "max_output_tokens": kw.get("max_tokens", 1024),
            "stream": False
        }
    }
}


# ===============================
# 通用 LLM 调用函数
# ===============================
def call_llm_api(messages: list,
                 provider: str,
                 model: str,
                 api_key: str,
                 max_tokens: int = 1024,
                 temperature: float = 0.0,
                 timeout: int = 30,
                 **kwargs) -> Tuple[bool, dict]:

    if provider not in MODEL_CONFIGS:
        return False, {"error": f"未知的模型提供商: {provider}"}

    cfg = MODEL_CONFIGS[provider]
    url = cfg["base_url"].rstrip("/") + cfg["endpoint"]
    headers = cfg["headers"](api_key)
    payload = cfg["payload"](model, messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        text = ""
        try:
            text = r.text
        except:
            pass
        return False, {"error": str(e), "resp_text": text}


# ===============================
# 规则定义（此处示例，可自行扩展）
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
    # 1.2 时间词
    "时间词": [
        {"name": "T1_可作介宾或“的时候/以来”前", "desc": "可以作介词'在/到/从'和动词性结构'等到'的宾语，或在'的时候/以来'前", "match_score": 20, "mismatch_score": -20},
        {"name": "T2_不能受程度副词", "desc": "不能受副词'很'/'不'修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "T3_可作不典型主语", "desc": "可以做不典型的主语（有人称之为状语，此时一般可在前面加'在'）", "match_score": 10, "mismatch_score": -10},
        {"name": "T4_可做不典型谓语", "desc": "可以做不典型的谓语（后附'了'或受时间副词修饰时，主谓之间一般不能插入'是'）", "match_score": 10, "mismatch_score": 0},
        {"name": "T5_不能带宾语和补语", "desc": "不能带宾语和补语（不能作述语）", "match_score": 10, "mismatch_score": -10},
        {"name": "T6_可作时间中心语/作定语", "desc": "一般可以做中心语受其他时间词修饰，或作定语修饰时间词", "match_score": 10, "mismatch_score": 0},
        {"name": "T7_一般不能受名词修饰", "desc": "一般不能作中心语受名词直接修饰，也不能作定语直接修饰名词", "match_score": 10, "mismatch_score": 0},
        {"name": "T8_可后附'的'作定语但通常不作主宾", "desc": "可以后附助词'的'构成定语，但一般不能作主语和宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "T9_可用'什么时候'提问/可用'这个时候'指代", "desc": "可以用'什么时候'提问或'这个时候/那个时候'指代", "match_score": 10, "mismatch_score": 0},
    ],
    # 1.3 方位词
    "方位词": [
        {"name": "P1_可作介词宾语/可以填介词框架", "desc": "可以作'向/从/往'等介词的宾语，或填入'从...到/向/往'框架", "match_score": 20, "mismatch_score": 0},
        {"name": "P2_可后附构处所结构", "desc": "可以后附在名词性成分之后构成处所结构", "match_score": 20, "mismatch_score": 0},
        {"name": "P3_可受区别词'最'修饰", "desc": "一般都可以受区别词'最'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "P4_不受数量词和形容词修饰", "desc": "不受数量词和形容词的修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "P5_不能直接修饰名词", "desc": "不能直接修饰名词", "match_score": 10, "mismatch_score": -10},
        {"name": "P6_不能受否定副词修饰", "desc": "不能受否定副词'不'和'没有'修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "P7_一般不能受程度副词'很'修饰", "desc": "一般不能受程度副词'很'修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "P8_不能跟在'怎么/怎样'与'这么/这样/那么'之后", "desc": "不能跟在'怎么/怎样'或'这么/这样/那么'之后", "match_score": 10, "mismatch_score": -10},
    ],
    # 1.4 处所词
    "处所词": [
        {"name": "L1_可做介词宾语/填介词框架", "desc": "可以做'在/到/从/往/向'等介词的宾语，或填入'从...到/向/往'框架", "match_score": 10, "mismatch_score": -10},
        {"name": "L2_不能作'等到'宾语/不能出现在'的时候/以来'前", "desc": "不能作动词性结构'等到'的宾语，不能出现在'的时候/以来'前", "match_score": 10, "mismatch_score": -10},
        {"name": "L3_不能后附方位词构处所", "desc": "不能后附方位词构成处所结构", "match_score": 0, "mismatch_score": -20},
        {"name": "L4_不能后附在名词性成分之后构处所", "desc": "不能后附在名词性成分之后构处所结构", "match_score": 10, "mismatch_score": -10},
        {"name": "L5_不能受数量词修饰", "desc": "不能受数量词的修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "L6_一般可作典型主宾语", "desc": "一般可以做典型的主语或宾语", "match_score": 10, "mismatch_score": 0},
        {"name": "L7_可做中心语受定语修饰", "desc": "可以做中心语受定语修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "L8_可后附'的'作结构", "desc": "可以后附助词'的'构成结构", "match_score": 10, "mismatch_score": 0},
        {"name": "L9_可用'哪儿'提问或用'这儿/那儿'指代", "desc": "可用'哪儿'提问或'这儿/那儿'指代", "match_score": 10, "mismatch_score": -10},
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
    # 1.6 形容词
    "形容词": [
        {"name": "A1_可受程度副词'很'修饰", "desc": "可以受程度副词'很'修饰", "match_score": 20, "mismatch_score": 0},
        {"name": "A2_不能直接带单宾语", "desc": "不能直接带单宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "A3_可做谓语/谓语核心", "desc": "可以做谓语或谓语核心（一般可受状语或补语修饰）", "match_score": 10, "mismatch_score": -10},
        {"name": "A4_可作定语修饰名词", "desc": "可以做定语直接修饰名词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "A5_可修饰动词（地）或被副词修饰", "desc": "可以独立或通过'地'等形式修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "A6_可做补语/带'得很/极了'", "desc": "可以做补语或带'得很/极了'等补语形式", "match_score": 10, "mismatch_score": 0},
        {"name": "A7_可在'比'句或'越来越'中用", "desc": "可以做'比'字句的谓语核心，或用在'越来越...'格式中", "match_score": 10, "mismatch_score": 0},
        {"name": "A8_可跟在'多/这样/多么'之后提问/感叹", "desc": "可跟在'多/这样/多么'之后用于提问/回答/感叹", "match_score": 10, "mismatch_score": 0},
    ],
    # 1.7 状态词
    "状态词": [
        {"name": "S1_不能受'很'或否定副词修饰", "desc": "不能受'很'等程度副词和否定副词修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "S2_直接或带'的'后可作谓语", "desc": "可以直接或带上后缀'的'作谓语或谓语核心", "match_score": 20, "mismatch_score": 0},
        {"name": "S3_不能带宾语", "desc": "不能带宾语（即使加上'着/了'也不能）", "match_score": 10, "mismatch_score": -10},
        {"name": "S4_带'的'后可做定语", "desc": "带上助词'的'后可以做定语修饰名词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "S5_可作补语/带补语形式", "desc": "带'的'后可作补语，并且不能带补语（条目按原文设分）", "match_score": 20, "mismatch_score": 0},
        {"name": "S6_可受时间副词等状语修饰或做状语", "desc": "可以受时间副词等状语修饰，或带'的'后作状语修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "S7_不能作'比'字句谓语核心/不能用'越来越'", "desc": "不能做'比'句谓语核心，也不能用在'越来越...'中", "match_score": 10, "mismatch_score": -10},
        {"name": "S8_不能跟在'多/这么/这样/多么'之后", "desc": "不能跟在'多'/'这么'等之后提问/回答/感叹", "match_score": 10, "mismatch_score": -10},
    ],
    # 1.8 区别词
    "区别词": [
        {"name": "D1_可作定语修饰名词", "desc": "可以直接作定语修饰名词性成分", "match_score": 20, "mismatch_score": 0},
        {"name": "D2_可加'的'构'的'字结构", "desc": "可以加上助词'的'构成'的'字结构", "match_score": 20, "mismatch_score": 0},
        {"name": "D3_不能受'不/很'等副词修饰", "desc": "不能受'不'和'很'等副词的修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "D4_不能受名词/形容词等定语修饰", "desc": "不能受一切名词或形容词等定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "D5_不能作主/宾（不考虑借代）", "desc": "不能作主语和宾语（不考虑借代）", "match_score": 10, "mismatch_score": -10},
        {"name": "D6_不能作谓语核心", "desc": "不能作谓语和谓语核心（不能受状语/补语/时体助词）", "match_score": 10, "mismatch_score": -10},
        {"name": "D7_不能作状语和补语", "desc": "不能作状语和补语", "match_score": 10, "mismatch_score": -10},
        {"name": "D8_不能单独回答问题（黏着语）", "desc": "不能单独回答问题（黏着语）", "match_score": 10, "mismatch_score": -10},
    ],
    # 1.9 副词
    "副词": [
        {"name": "ADV1_可作状语直接修饰谓词", "desc": "可以做状语直接修饰动词或形容词等谓词性成分", "match_score": 30, "mismatch_score": -30},
        {"name": "ADV2_不能作定语修饰名词", "desc": "不能作定语修饰名词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "ADV3_不能加'的'构'的'字结构", "desc": "不能加上助词'的'构成'的'字结构", "match_score": 10, "mismatch_score": 0},
        {"name": "ADV4_不能作主语和宾语", "desc": "不能作主语和宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "ADV5_不能作谓语核心", "desc": "不能作谓语和谓语核心（不能带宾语/时体助词等）", "match_score": 10, "mismatch_score": -10},
        {"name": "ADV6_不能受状语和补语修饰", "desc": "不能受状语和补语修饰（不能作谓词性短语的中心语）", "match_score": 10, "mismatch_score": -10},
        {"name": "ADV7_不能作补语（少数例外）", "desc": "不能作补语（只有少数如'很'等例外）", "match_score": 10, "mismatch_score": 0},
        {"name": "ADV8_不能有重叠/正反重叠形式", "desc": "不能有'FF,F一F,F了F'等重叠形式", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.1 介词
    "介词": [
        {"name": "PREP1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP2_后面必须跟宾语", "desc": "后面必须跟着宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "PREP3_介宾之间不能插时体助词", "desc": "在介词和宾语之间不能加入时体助词'着/了/过'", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP4_不能有重叠形式", "desc": "不能构成'PrepPrep'等重叠形式", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP5_不能作主/宾（不能受定语）", "desc": "不能作主语和宾语（因而不能受定语修饰）", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP6_不能作谓语核心/不能受状补", "desc": "不能作谓语和谓语核心，且不能受状语/补语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP7_不能作状语补语和定语", "desc": "不能作状语、补语和定语等修饰性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "PREP8_介词结构可作状语或补语/可后加'的'构定语", "desc": "由介词和宾语组成的介词结构可以做状语或补语（有的可后加'的'作定语）", "match_score": 20, "mismatch_score": -20},
    ],
    # 2.2 连词
    "连词": [
        {"name": "CONJ1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ2_配套或成对使用/五种分布情况", "desc": "可以配套或单独用在成对的语言形式之前（具备概括性五种用法之一得60分）", "match_score": 60, "mismatch_score": -60},
        {"name": "CONJ3_不能作主宾/不能受定语修饰", "desc": "不能作主语和宾语，且不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰性成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.3 助词
    "助词": [
        {"name": "PART1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PART2_只能附着在其他成分之前或之后（六种分布之一得60）", "desc": "只能附着在其他成分之前或之后，构成词性结构（若具备下列六种用法之一得60分）", "match_score": 60, "mismatch_score": -60},
        {"name": "PART3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，且不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PART4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "PART5_不能做修饰性成分", "desc": "不能做状语、补语和定语等修饰性成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.4 语气词
    "语气词": [
        {"name": "MOD1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD2_只能附着在其他成分之后（四种用法之一得60）", "desc": "只能附着在其他成分之后（句末/话题性成分后/并列项后/假设分句后）", "match_score": 60, "mismatch_score": -60},
        {"name": "MOD3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰成分", "match_score": 10, "mismatch_score": -10},
    ],
    # 2.5 感叹词
    "感叹词": [
        {"name": "INT1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 30, "mismatch_score": -30},
        {"name": "INT2_可以独立成句（前后长停顿）", "desc": "可以独立成句（前后都可有较长停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "INT3_不能跟其他句法成分组合构句法结构", "desc": "不能与其他句法成分组合构成主谓/述补/并列等结构", "match_score": 50, "mismatch_score": -50},
    ],
    # 2.6 拟声词
    "拟声词": [
        {"name": "ON1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "ON2_可以独立成句", "desc": "可以独立成句", "match_score": 20, "mismatch_score": -20},
        {"name": "ON3_可直接或带'的'作定语", "desc": "可以直接或带'的'后作定语修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON4_可直接或带'地'作状语", "desc": "可以直接或后带'地'作状语修饰动词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON5_不能充当主/宾/谓/补等", "desc": "不能充当主语、宾语、谓语和补语等句法成分", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.1 体代词（代词与数量词部分示例）
    "体代词": [
        {"name": "PR1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "PR2_可做定语或跟'的'构'的'字结构", "desc": "可以做定语或跟助词'的'构成'的'字结构", "match_score": 10, "mismatch_score": -10},
        {"name": "PR3_不能受数量/形容词/'的'修饰", "desc": "不能受数量词、形容词和'的'字结构的修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "PR4_不能受'不/很'等副词修饰", "desc": "不能受'不'和'很'等副词修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PR5_不能作谓语核心", "desc": "不能作谓语和谓语核心（不能带宾语/时体助词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PR6_不能做补语或状语", "desc": "不能做补语，也不能作状语", "match_score": 10, "mismatch_score": -10},
        {"name": "PR7_不能后附单音方位词构处所", "desc": "不能后附单音方位词构处所", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.2 谓代词（示例）
    "谓代词": [
        {"name": "WP1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "WP2_可作状语直接修饰动/形", "desc": "可以作状语直接修饰动词或形容词", "match_score": 20, "mismatch_score": -20},
        {"name": "WP3_不能受'很'等程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP4_可受'不/也'等副词修饰", "desc": "可以受'不'或'也'等副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP5_可做谓语或谓词核心", "desc": "可以做谓语或谓词核心", "match_score": 10, "mismatch_score": -10},
        {"name": "WP6_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 10, "mismatch_score": -10},
    ],
    # 3.3 代词（通用代词条目示例）
    "代词": [
        {"name": "DPR1_可作典型主宾语", "desc": "可做典型主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR2_不能受数量/形容/的修饰", "desc": "不能受数量词、形容词和'的'字结构修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR3_不能受程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR4_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 20, "mismatch_score": 0},
        {"name": "DPR5_可受'不/也'等副词修饰（针对谓代）或不能后附方位（针对体代）", "desc": "混合规则，按具体代词类型判定", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.4 系数词、位数词、合成数词等：示例性规则（你可以继续补全）
    "系数词": [
        {"name": "NUM_CO1_黏着词不能单独回答", "desc": "系数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO2_可在量词前构数量词组", "desc": "可以用在量词前，一起构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO3_可构系谓构造", "desc": "可以用在位数词/构成序数组合等", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO4_可构序数组合（第...）", "desc": "可以用在'第'的后面构成序数组合", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO5_不能直接修饰名词（除非省略'第'）", "desc": "不能直接修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
    ],
    "位数词": [
        {"name": "NUM_POS1_黏着词不能单独回答", "desc": "位数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS2_不能单独用在量词前", "desc": "不能单独用在量词前", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_POS3_可在系数词后构成系位构造", "desc": "可以用在系数词后构成系位构造", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS4_不能用于前缀'第'后面构序数组合", "desc": "不能用于前缀'第'后面构序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS5_不能作定语直接修饰名词", "desc": "不能作定语直接修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_POS6_可用在'来/把'之前构数次组合", "desc": "可以用在助词'来'或'把'之前构成数次组合", "match_score": 10, "mismatch_score": 0},
    ],
    "合成数词": [
        {"name": "NUM_COM1_可以单独回答问题（部分自由）", "desc": "合成数词可以用来单独回答问题", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_COM2_可与量词构数量词组", "desc": "可以用在量词前构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM3_可在'第'后构序数组合", "desc": "可以用在'第'后造成序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM4_不能直接作定语修饰名词（除非省第）", "desc": "不能直接作定语修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_COM5_可出现在'来/多/余'之前等特殊分布", "desc": "可以出现在特定助词之前（见原文条目）", "match_score": 30, "mismatch_score": 0},
    ],
    # 其他规则占位（便于以后补全）
    # "未列出词类": [ ... ],
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}


# ===============================
# 工具函数（解析模型响应等）
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        return ""
    try:
        choices = resp_json.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            msg = first.get("message")
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            for k in ("content", "text", "message"):
                if k in first and isinstance(first[k], str):
                    return first[k]
    except:
        pass
    return json.dumps(resp_json, ensure_ascii=False)


def extract_json_from_text(text: str) -> Tuple[dict, str]:
    if not text:
        return None, ""
    s = text.strip()
    try:
        return json.loads(s), s
    except:
        m = re.search(r"(\{[\s\S]*\})", s)
        if not m:
            return None, s
        cand = m.group(1)
        c = cand.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
        c = re.sub(r"'(\s*[^']+?\s*)'\s*:", r'"\1":', c)
        c = re.sub(r":\s*'([^']*?)'", r': "\1"', c)
        c = re.sub(r",\s*([}\]])", r"\1", c)
        c = re.sub(r"\bTrue\b", "true", c)
        c = re.sub(r"\bFalse\b", "false", c)
        c = re.sub(r"\bNone\b", "null", c)
        try:
            return json.loads(c), c
        except:
            return None, s


def normalize_key(k: str, pos_rules: list) -> str:
    if not isinstance(k, str):
        return None
    kk = re.sub(r'\s+', '', k).upper()
    for r in pos_rules:
        if r["name"].upper() == kk or re.sub(r'\s+', '', r["name"]).upper() == kk:
            return r["name"]
    return None


def map_to_allowed_score(rule: dict, raw_val) -> int:
    match = rule["match_score"]
    mismatch = rule["mismatch_score"]
    if isinstance(raw_val, (int, float)):
        cand = [match, mismatch]
        return min(cand, key=lambda x: abs(x - float(raw_val)))
    if isinstance(raw_val, bool):
        return match if raw_val else mismatch
    if isinstance(raw_val, str):
        s = raw_val.strip().lower()
        if s in ("yes", "y", "true", "是", "√", "符合"):
            return match
        if s in ("no", "n", "false", "否", "×", "不符合"):
            return mismatch
    return mismatch


# ===============================
# 主逻辑：请求模型进行词类判定
# ===============================
def ask_model_for_pos_and_scores(word: str):
    """
    调用所选模型 API，返回：
    - scores_all: 各词类的规则得分
    - raw_out: 模型原始输出
    - predicted_pos: 模型预测的主词类
    """

    prompt = f"""请判断汉语词语“{word}”的词类，并基于规则给出各主要词类（名词、动词、形容词、副词、助词、连词、介词、量词、代词、数词等）的隶属度评分（0-1），格式如下：
{{"名词": 0.2, "动词": 0.8, "形容词": 0.3, "副词": 0.6, ...}}。同时说明理由。
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个汉语语法学家，精通词类划分与语义角色分析。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=60)
        result = response.json()

        # 兼容不同模型的返回字段
        if "choices" in result:
            raw_text = result["choices"][0]["message"]["content"]
        elif "output" in result:
            raw_text = result["output"]
        else:
            raw_text = str(result)

        # 尝试解析出得分
        try:
            score_json = json.loads(raw_text)
        except:
            # 简单容错：从文本中提取伪JSON
            import re
            m = re.search(r"\{.*\}", raw_text, re.S)
            score_json = json.loads(m.group(0)) if m else {}

        # 提取最高分词类
        if score_json:
            predicted_pos = max(score_json, key=score_json.get)
        else:
            predicted_pos = "未知"

        return score_json, raw_text, predicted_pos

    except Exception as e:
        st.error(f"模型调用出错：{e}")
        return {}, "", "错误"

def ask_model_for_pos_and_scores(word: str,
                                 provider: str,
                                 model: str,
                                 api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str]:

    rules_summary_lines = []
    for pos, rules in RULE_SETS.items():
        rules_summary_lines.append(f"{pos}:")
        for r in rules:
            rules_summary_lines.append(f"  - {r['name']}: {r['desc']} (match={r['match_score']}, mismatch={r['mismatch_score']})")
    rules_text = "\n".join(rules_summary_lines)

    system_msg = (
        "你是语言学研究助手。输入一个中文词语，请你判断该词最可能的词类，并返回 JSON："
        '{"predicted_pos":"<词类名>", "scores": {"<词类名>": {"<规则名>": <值>, ...}, ...}, "explanation":"说明"}。'
    )
    user_prompt = f"词语：『{word}』\n请基于下列规则判定并评分：\n\n{rules_text}\n\n仅返回严格 JSON。"

    ok, resp = call_llm_api(
        [{"role": "system", "content": system_msg},
         {"role": "user", "content": user_prompt}],
        provider=provider, model=model, api_key=api_key
    )

    raw_text = extract_text_from_response(resp) if ok else str(resp)
    parsed_json, _ = extract_json_from_text(raw_text)
    if not parsed_json:
        return {}, raw_text, "未知"

    scores_out = {}
    predicted_pos = parsed_json.get("predicted_pos", "未知")
    raw_scores = parsed_json.get("scores", {})

    for pos, rules in RULE_SETS.items():
        scores_out[pos] = {r["name"]: 0 for r in rules}
        raw_for_pos = raw_scores.get(pos, {})
        if isinstance(raw_for_pos, dict):
            for k, v in raw_for_pos.items():
                nk = normalize_key(k, rules)
                if nk:
                    rule_def = next(r for r in rules if r["name"] == nk)
                    scores_out[pos][nk] = map_to_allowed_score(rule_def, v)
    return scores_out, raw_text, predicted_pos


# ===============================
# 可视化函数
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str):
    categories = list(scores_norm.keys())
    values = [float(scores_norm[c]) for c in categories]
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title=dict(text=title, x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Streamlit UI（简洁居中输入 + 模型选择 + 结果）
# ===============================

# ======== 模型选择部分（侧边栏） ========
MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "api_url": "https://api.deepseek.com/v1/chat/completions"
    },
    "OpenAI GPT-4o": {
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "Moonshot（Kimi）": {
        "api_url": "https://api.moonshot.cn/v1/chat/completions"
    },
    "Doubao（豆包）": {
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    }
}

MODEL_API_KEYS = {
    "DeepSeek Chat": "sk-1f346646d29947d0a5e29dbaa37476b8",
    "OpenAI GPT-4o": "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",
    "Moonshot（Kimi）": "sk-your-moonshot-key",
    "Doubao（豆包）": "WmpRMlptTmxNRGM0TjJNMk5HUTBOR0ZtWVRsbU56TTNNakUyT0RVNU1EUQ=="
}

# 由侧边栏选择模型
model_choice = st.sidebar.selectbox("选择模型", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_choice]

st.sidebar.markdown(f"**当前模型：** {model_choice}")
st.sidebar.markdown(f"**API 地址：** `{selected_model['api_url']}`")

# 直接取 Key（不再用环境变量）
API_URL = selected_model["api_url"]
API_KEY = MODEL_API_KEYS.get(model_choice, "")

if not API_KEY:
    st.sidebar.error(f"⚠️ 尚未为模型 {model_choice} 配置 API Key，请在代码中填写。")

# ======== 主体部分 ========
st.markdown("<h1 style='text-align: center;'>汉语词类隶属度检测判类</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>输入单个词 → 模型自动判类并返回各词类规则得分与隶属度（标准化 0~1）</p>", unsafe_allow_html=True)
st.write("")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    word_input = st.text_input("", placeholder="在此输入要分析的词（例如：很 / 跑 / 美丽）")
    confirm = st.button("确认")
# 当按下确认时，进行模型调用与展示
if confirm:
    word = (word_input or "").strip()
    if not word:
        st.warning("请输入一个词语后确认。")
    else:
        with st.spinner("模型打分判类中……"):
            scores_all, raw_out, predicted_pos = ask_model_for_pos_and_scores(word)
        # 计算每个词类总分与归一化隶属度（0~1）
        pos_totals = {}
        pos_normed = {}
        for pos, score_map in scores_all.items():
            total = sum(score_map.values())
            pos_totals[pos] = total
            max_possible = MAX_SCORES.get(pos, sum(abs(x) for x in score_map.values()) or 1)
            # 归一化（0~1），用 max(0,total)/max_possible 避免负值
            norm = round(max(0, total) / max_possible, 3) if max_possible != 0 else 0.0
            pos_normed[pos] = norm

        # 输出顶部摘要
        st.markdown("---")
        st.subheader("判定摘要")
        st.markdown(f"- **输入词**： `{word}`")
        st.markdown(f"- **模型预测词类**： **{predicted_pos}**")
        st.markdown(f"- **解析策略 / 原始响应摘要**： `{raw_out}`")

        # 排名与表格（只显示前 10）
        ranked = sorted(pos_normed.items(), key=lambda x: x[1], reverse=True)
        st.subheader("隶属度排行（前10）")
        for i, (p, s) in enumerate(ranked[:10]):
            st.write(f"{i+1}. **{p}** — 隶属度：{s}")

        # 雷达图（全量显示当前 RULE_SETS 中的词类）
        st.subheader("词类隶属度雷达图（标准化 0~1）")
        # 为了图示美观，将 pos_normed 的顺序固定为字典顺序或按得分排序——这里按得分排序
        radar_scores = {p: pos_normed[p] for p, _ in ranked}
        plot_radar_chart_streamlit(radar_scores, title=f"“{word}” 的词类隶属度分布")

        # 显示归一化表格
        st.subheader("各词类隶属度（标准化 0~1）")
        df_norm = pd.DataFrame([{"词类": p, "隶属度": pos_normed[p]} for p in pos_normed]).set_index("词类")
        st.dataframe(df_norm, use_container_width=True)

        # 折叠详细规则判断（默认收起）
        with st.expander("展开：查看各词类的规则明细与得分（详细）"):
            for pos, rules in RULE_SETS.items():
                st.markdown(f"**{pos}**（隶属度：{pos_normed.get(pos, 0)}）")
                rows = []
                scores_for_pos = scores_all.get(pos, {r["name"]: 0 for r in rules})
                for r in rules:
                    nm = r["name"]
                    sc = scores_for_pos.get(nm, 0)
                    decision = "是" if sc == r["match_score"] else ("否" if sc == r["mismatch_score"] else "")
                    rows.append({"规则": nm, "描述": r["desc"], "得分": sc, "判定": decision})
                if rows:
                    st.table(pd.DataFrame(rows))
                else:
                    st.write("（该词类当前无规则条目）")
                st.markdown("---")

        # 可选：显示原始模型输出
        with st.expander("查看原始模型文本 / 响应"):
            st.code(raw_out if raw_out else "(无)")




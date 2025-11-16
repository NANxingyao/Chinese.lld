# 完整的汉语词类隶属度检测代码
import os
import re
import json
import requests
import traceback
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, List
import streamlit as st

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 隐藏Streamlit默认元素
hide_streamlit_style = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 模型配置
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_parser": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_parser": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_parser": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "messages": messages,
            "max_tokens": kw.get("max_tokens", 1024),
            "temperature": kw.get("temperature", 0.0),
            "stream": False,
        },
        "response_parser": lambda resp: resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model,
            "input": {"messages": messages},
            "parameters": {
                "max_tokens": kw.get("max_tokens", 1024),
                "temperature": kw.get("temperature", 0.0),
            },
        },
        "response_parser": lambda resp: resp.get("output", {}).get("text", "")
    },
}

# 模型选项（从环境变量获取API密钥）
MODEL_OPTIONS = {
    "DeepSeek Chat": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-1f346646d29947d0a5e29dbaa37476b8"),
    },
    "OpenAI GPT-4o": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY", "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"),
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot",
        "model": "moonshot-v1-32k",
        "api_key": os.getenv("MOONSHOT_API_KEY", "sk-l5FvRWegjM5DEk4AU71YPQ1QgvFPTHZIJOmq6qdssPY4sNtE"),
    },
    "Doubao（豆包）": {
        "provider": "doubao",
        "model": "doubao-pro-32k",
        "api_key": os.getenv("DOUBAO_API_KEY", "222afa3f-5f27-403e-bf46-ced2a356ceee"),
    },
    "Qwen（通义千问）": {
        "provider": "qwen",
        "model": "qwen-max",
        "api_key": os.getenv("QWEN_API_KEY", "sk-b3f7a1153e6f4a44804a296038aa86c5"),
    },
}

# ===============================
# 词类规则集
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
    "时间词": [
        {"name": "T1_可作介宾或“的时候/以来”前", "desc": "可以作介词'在/到/从'和动词性结构'等到'的宾语，或在'的时候/以来'前", "match_score": 20, "mismatch_score": -20},
        {"name": "T2_不能受程度副词", "desc": "不能受副词'很'/'不'修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "T3_可作不典型主语", "desc": "可以做不典型的主语（有人称之为状语，此时一般可在前面加'在'）", "match_score": 10, "mismatch_score": -10},
        {"name": "T4_可做不典型谓语", "desc": "可以做不典型的谓语（后附'了'或受时间副词修饰时，主谓之间一般不能插入'是'）", "match_score": 10, "mismatch_score": 0},
        {"name": "T5_不能带宾语和补语", "desc": "不能带宾语和补语（不能作述语）", "match_score": 10, "mismatch_score": -10},
        {"name": "T6_可作时间中心语/作定语", "desc": "一般可以做中心语受其他时间词修饰，或作定语修饰时间词", "match_score": 10, "mismatch_score": 0},
        {"name": "T7_一般不能受名词修饰", "desc": "一般不能作中心语受名词直接修饰，也不能作定语直接修饰名词", "match_score": 10, "mismatch_score": 0},
        {"name": "T8_可后附'的'作定语但通常不作主宾", "desc": "可以后附助词'的'作定语，但一般不能作主语和宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "T9_可用'什么时候'提问/可用'这个时候'指代", "desc": "可以用'什么时候'提问或'这个时候/那个时候'指代", "match_score": 10, "mismatch_score": 0},
    ],
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
    "连词": [
        {"name": "CONJ1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ2_配套或成对使用/五种分布情况", "desc": "可以配套或单独用在成对的语言形式之前（具备概括性五种用法之一得60分）", "match_score": 60, "mismatch_score": -60},
        {"name": "CONJ3_不能作主宾/不能受定语修饰", "desc": "不能作主语和宾语，且不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "CONJ5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰性成分", "match_score": 10, "mismatch_score": -10},
    ],
    "助词": [
        {"name": "PART1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PART2_只能附着在其他成分之前或之后（六种分布之一得60）", "desc": "只能附着在其他成分之前或之后，构成词性结构（若具备下列六种用法之一得60分）", "match_score": 60, "mismatch_score": -60},
        {"name": "PART3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，且不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PART4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "PART5_不能做修饰性成分", "desc": "不能做状语、补语和定语等修饰性成分", "match_score": 10, "mismatch_score": -10},
    ],
    "语气词": [
        {"name": "MOD1_不能单独回答（黏着词）", "desc": "不能单独回答问题（黏着词）", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD2_只能附着在其他成分之后（四种用法之一得60）", "desc": "只能附着在其他成分之后（句末/话题性成分后/并列项后/假设分句后）", "match_score": 60, "mismatch_score": -60},
        {"name": "MOD3_不能作主宾/不能受定语", "desc": "不能作主语和宾语，不能受定语修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD4_不能作谓语核心", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "MOD5_不能作修饰性成分", "desc": "不能作状语、定语和补语等修饰成分", "match_score": 10, "mismatch_score": -10},
    ],
    "感叹词": [
        {"name": "INT1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 30, "mismatch_score": -30},
        {"name": "INT2_可以独立成句（前后长停顿）", "desc": "可以独立成句（前后都可有较长停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "INT3_不能跟其他句法成分组合构句法结构", "desc": "不能与其他句法成分组合构成主谓/述补/并列等结构", "match_score": 50, "mismatch_score": -50},
    ],
    "拟声词": [
        {"name": "ON1_可充当独立成分（停顿）", "desc": "可以充当独立成分（前后可有停顿）", "match_score": 20, "mismatch_score": -20},
        {"name": "ON2_可以独立成句", "desc": "可以独立成句", "match_score": 20, "mismatch_score": -20},
        {"name": "ON3_可直接或带'的'作定语", "desc": "可以直接或带'的'后作定语修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON4_可直接或带'地'作状语", "desc": "可以直接或后带'地'作状语修饰动词", "match_score": 20, "mismatch_score": 0},
        {"name": "ON5_不能充当主/宾/谓/补等", "desc": "不能充当主语、宾语、谓语和补语等句法成分", "match_score": 20, "mismatch_score": -20},
    ],
    "体代词": [
        {"name": "PR1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "PR2_可做定语或跟'的'构'的'字结构", "desc": "可以做定语或跟助词'的'构成'的'字结构", "match_score": 10, "mismatch_score": -10},
        {"name": "PR3_不能受数量/形容词/'的'修饰", "desc": "不能受数量词、形容词和'的'字结构的修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "PR4_不能受'不/很'等副词修饰", "desc": "不能受'不'和'很'等副词修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "PR5_不能作谓语核心", "desc": "不能作谓语和谓语核心（不能带宾语/时体助词）", "match_score": 10, "mismatch_score": -10},
        {"name": "PR6_不能做补语或状语", "desc": "不能做补语，也不能作状语", "match_score": 10, "mismatch_score": -10},
        {"name": "PR7_不能后附单音方位词构处所", "desc": "不能后附单音方位词构处所", "match_score": 20, "mismatch_score": -20},
    ],
    "谓代词": [
        {"name": "WP1_可作典型主宾语", "desc": "可以做典型的主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "WP2_可作状语直接修饰动/形", "desc": "可以作状语直接修饰动词或形容词", "match_score": 20, "mismatch_score": -20},
        {"name": "WP3_不能受'很'等程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP4_可受'不/也'等副词修饰", "desc": "可以受'不'或'也'等副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "WP5_可做谓语或谓词核心", "desc": "可以做谓语或谓词核心", "match_score": 10, "mismatch_score": -10},
        {"name": "WP6_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 10, "mismatch_score": -10},
    ],
    "代词": [
        {"name": "DPR1_可作典型主宾语", "desc": "可做典型主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR2_不能受数量/形容/的修饰", "desc": "不能受数量词、形容词和'的'字结构修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR3_不能受程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR4_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 20, "mismatch_score": 0},
        {"name": "DPR5_可受'不/也'等副词修饰（针对谓代）或不能后附方位（针对体代）", "desc": "混合规则，按具体代词类型判定", "match_score": 20, "mismatch_score": -20},
    ],
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
}

MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any], provider: str) -> str:
    """根据不同模型提供商提取响应文本"""
    if not isinstance(resp_json, dict):
        return ""
    
    try:
        # 使用模型配置中的响应解析器
        if provider in MODEL_CONFIGS:
            return MODEL_CONFIGS[provider]["response_parser"](resp_json)
            
        # 通用解析作为后备
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
# 安全的 LLM 调用函数
# ===============================
def call_llm_api(messages: list, provider: str, model: str, api_key: str,
                 max_tokens: int = 1024, temperature: float = 0.0, timeout: int = 30) -> Tuple[bool, dict, str]:
    """
    调用指定 LLM API 获取响应。
    返回: (成功标志, 响应 dict, 错误信息)
    """
    if not api_key:
        return False, {"error": "API Key 为空"}, "API Key 未提供"

    if provider not in MODEL_CONFIGS:
        return False, {"error": f"未知提供商 {provider}"}, f"未知提供商 {provider}"

    cfg = MODEL_CONFIGS[provider]
    url = cfg["base_url"].rstrip("/") + cfg.get("endpoint", "/chat/completions")
    headers = cfg["headers"](api_key)
    payload = cfg["payload"](model, messages, max_tokens=max_tokens, temperature=temperature)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        if r.status_code != 200:
            return False, {"error": f"HTTP错误 {r.status_code}", "content": r.text}, f"HTTP错误 {r.status_code}: {r.text[:200]}"
            
        r.raise_for_status()
        resp_json = r.json()
        return True, resp_json, ""
    except Exception as e:
        error_msg = str(e)
        return False, {"error": error_msg}, error_msg

# 新增：测试模型连接函数
def test_model_connection(provider: str, model: str, api_key: str) -> Tuple[bool, str]:
    """测试模型连接是否成功"""
    if not api_key:
        return False, "API Key 未提供"
    
    # 使用简单消息测试连接
    test_messages = [
        {"role": "system", "content": "请返回'连接测试成功'"},
        {"role": "user", "content": "测试连接"}
    ]
    
    ok, _, err_msg = call_llm_api(
        messages=test_messages,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=10,
        temperature=0.0
    )
    
    if ok:
        return True, "连接成功"
    else:
        return False, err_msg

# ===============================
# 安全的词类判定函数
# ===============================
def ask_model_for_pos_and_scores(word: str, provider: str, model: str, api_key: str) -> Tuple[Dict[str, Dict[str, int]], str, str]:
    """
    根据输入词调用 LLM 获取词类隶属度评分，返回:
        - scores_all: 每个词类的规则得分字典
        - raw_text: 模型原始输出
        - predicted_pos: 模型预测的最可能词类
    """
    if not word:
        return {}, "", "未知"

    rules_summary_lines = []
    for pos, rules in RULE_SETS.items():
        rules_summary_lines.append(f"{pos}:")
        for r in rules:
            rules_summary_lines.append(f"  - {r['name']}: {r['desc']} (match={r['match_score']}, mismatch={r['mismatch_score']})")
    rules_text = "\n".join(rules_summary_lines)

    system_msg = (
        "你是语言学研究专家，拥有中外语言学界的所有知识。在输入一个中文词语后，请检索全网的相关知识，严格按照定义的规则，请判断最可能的词类并返回 JSON："
        '{"predicted_pos":"<词类名>", "scores": {"<词类名>": {"<规则名>": <值>, ...}, ...}, "explanation":"说明"}。'
    )
    user_prompt = f"词语：『{word}』\n请基于下列规则判定并评分：\n\n{rules_text}\n\n仅返回严格 JSON。"

    ok, resp_json, err_msg = call_llm_api(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_prompt}],
        provider=provider,
        model=model,
        api_key=api_key
    )

    if not ok or not resp_json:
        return {}, f"调用失败或返回异常: {err_msg}", "未知"

    raw_text = extract_text_from_response(resp_json, provider)
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
# 主页面逻辑
# ===============================
def main():
    # 添加侧边栏控制按钮
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("≡"):
            st.session_state.sidebar_state = "expanded" if st.session_state.get("sidebar_state") == "collapsed" else "collapsed"
            st.rerun()
    
    # 设置侧边栏状态
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = "collapsed"
    
    # 侧边栏内容
    with st.sidebar:
        st.markdown("## 模型设置")
        # 模型选择
        selected_model_name = st.selectbox(
            "选择模型",
            list(MODEL_OPTIONS.keys()),
            index=0
        )
        
        # 获取选中模型的配置
        selected_model = MODEL_OPTIONS[selected_model_name]
        provider = selected_model["provider"]
        model_name = selected_model["model"]
        api_key = selected_model["api_key"]
        
        st.markdown(f"**当前模型：** {selected_model_name}")
        st.markdown(f"**模型ID：** `{model_name}`")
        
        # 测试连接按钮
        if st.button("测试模型连接"):
            if not api_key:
                st.error("API Key 未配置")
            else:
                with st.spinner("测试连接中..."):
                    ok, msg = test_model_connection(
                        provider,
                        model_name,
                        api_key
                    )
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ 连接失败：{msg}")
    
    # 主页面内容
    st.title("汉语词类隶属度检测")
    word = st.text_input("请输入要检测的汉语词语", "")
    
    if st.button("开始检测"):
        if not word:
            st.warning("请输入词语")
            return
            
        with st.spinner("正在分析..."):
            scores, raw_text, predicted_pos = ask_model_for_pos_and_scores(
                word=word,
                provider=provider,
                model=model_name,
                api_key=api_key
            )
            
            st.success(f"预测词类: {predicted_pos}")
            
            # 计算归一化得分
            scores_norm = {}
            for pos in scores:
                total = sum(scores[pos].values())
                max_score = MAX_SCORES.get(pos, 1)  # 避免除以零
                if max_score == 0:
                    norm = 0.0
                else:
                    norm = (total / max_score + 1) / 2  # 归一化到0-1范围
                scores_norm[pos] = max(0.0, min(1.0, norm))  # 确保在0-1之间
            
            # 显示雷达图（只显示前8个词类，避免图表过于复杂）
            top_pos = sorted(scores_norm.items(), key=lambda x: x[1], reverse=True)[:8]
            plot_radar_chart_streamlit(dict(top_pos), f"词语『{word}』的词类隶属度雷达图（前8）")
            
            # 显示详细得分
            with st.expander("查看详细得分"):
                for pos in scores:
                    st.subheader(pos)
                    df = pd.DataFrame.from_dict(scores[pos], orient="index", columns=["得分"])
                    st.dataframe(df)
            
            # 显示原始响应
            with st.expander("查看模型原始响应"):
                st.text(raw_text)

if __name__ == "__main__":
    main()

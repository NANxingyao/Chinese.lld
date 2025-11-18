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
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-1f346646d29947d0a5e29dbaa37476b8"),
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
    # 1.2 时间词
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
    # 3.3 代词
    "代词": [
        {"name": "DPR1_可作典型主宾语", "desc": "可做典型主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR2_不能受数量/形容/的修饰", "desc": "不能受数量词、形容词和'的'字结构修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR3_不能受程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR4_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 20, "mismatch_score": 0},
        {"name": "DPR5_可受'不/也'等副词修饰（针对谓代）或不能后附方位（针对体代）", "desc": "混合规则，按具体代词类型判定", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.4 系数词
    "系数词": [
        {"name": "NUM_CO1_黏着词不能单独回答", "desc": "系数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO2_可在量词前构数量词组", "desc": "可以用在量词前，一起构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_CO3_可构系谓构造", "desc": "可以用在位数词/构成序数组合等", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO4_可构序数组合（第...）", "desc": "可以用在'第'的后面构成序数组合", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_CO5_不能直接修饰名词（除非省略'第'）", "desc": "不能直接修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
    ],
     # 3.5 位数词
    "位数词": [
        {"name": "NUM_POS1_黏着词不能单独回答", "desc": "位数词是黏着词不能单独回答", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS2_不能单独用在量词前", "desc": "不能单独用在量词前", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_POS3_可在系数词后构成系位构造", "desc": "可以用在系数词后构成系位构造", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS4_不能用于前缀'第'后面构序数组合", "desc": "不能用于前缀'第'后面构序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_POS5_不能作定语直接修饰名词", "desc": "不能作定语直接修饰名词", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_POS6_可用在'来/把'之前构数次组合", "desc": "可以用在助词'来'或'把'之前构成数次组合", "match_score": 10, "mismatch_score": 0},
    ],
     # 3.6 合成数词
    "合成数词": [
        {"name": "NUM_COM1_可以单独回答问题（部分自由）", "desc": "合成数词可以用来单独回答问题", "match_score": 10, "mismatch_score": 0},
        {"name": "NUM_COM2_可与量词构数量词组", "desc": "可以用在量词前构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM3_可在'第'后构序数组合", "desc": "可以用在'第'后造成序数组合", "match_score": 20, "mismatch_score": -20},
        {"name": "NUM_COM4_不能直接作定语修饰名词（除非省第）", "desc": "不能直接作定语修饰名词（除非省略'第'）", "match_score": 20, "mismatch_score": 0},
        {"name": "NUM_COM5_可出现在'来/多/余'之前等特殊分布", "desc": "可以出现在特定助词之前（见原文条目）", "match_score": 30, "mismatch_score": 0},
    ],
     # 3.7 数词
    "数词": [
        {"name": "N1_可与量词组合", "desc": "可以单独用在量词前面或跟其他数词组成合成数词用在量词前面(一起构成数量词组，然后作定语、主语、宾语或谓语等句法成分)", "match_score": 30, "mismatch_score": -30},
        {"name": "N2_不能直接作定语", "desc": "不能作定语直接修饰名词(除非是作为省略了“第”的序数组合。换句话说，直接修饰名词的“数词+名词”组合前一定可以加上前缀“第”)", "match_score": 30, "mismatch_score": 0},
        {"name": "N3_书面语序数用法", "desc": "(在书面语中)可以抛开量词直接用在名词后面表示序数意义", "match_score": 20, "mismatch_score": 0},
        {"name": "N4_不能单独回答问题", "desc": "一般不能用来单独回答问题", "match_score": 10, "mismatch_score": 0},
        {"name": "N5_有限句法功能", "desc": "一般不能作主语、宾语或谓语(除非是用在算术表达式或流水账式的列举等特殊的格式中)", "match_score": 10, "mismatch_score": 0}
    ],
     # 3.8 简单量词
    "简单量词": [
        {"name": "MQ1_不能单独回答", "desc": "(简单量词是黏着词,)不能用来单独回答问题", "match_score": 20, "mismatch_score": -20},
        {"name": "MQ2_可与数词组合", "desc": "可以直接用在数词(包括简单数词和合成数词)之后构成数量词组(然后作定语修饰名词性成分或作动词性成分的准宾语)", "match_score": 20, "mismatch_score": -20},
        {"name": "MQ3_可与数词短语组合", "desc": "可以直接用在由位数词和数量助词(包括前附助词“上”和后附助词“来、把”)构成的数词短语之后，一起构成数量词组", "match_score": 20, "mismatch_score": -20},
        {"name": "MQ4_可加“每”表全称", "desc": "可以前加“每”来表示全称(universal，或称“周遍性”)(再去修饰名词性成分或独立指称事物)，并且，在“每和量词之间可以自由地插入或删除数词“一”", "match_score": 10, "mismatch_score": 0},
        {"name": "MQ5_可重叠表全称", "desc": "可以通过重叠来表示全称(再去修饰名词性成分，或独立指称事物。有时，还可以在这种重叠式之前加上数词“一”)", "match_score": 10, "mismatch_score": 0},
        {"name": "MQ6_可用于“这/那+量词+名词”结构", "desc": "可以用在代词“这、那”和名词性成分之间，并且在“这、那”和量词之间可以自由地插入或删除数词“一”", "match_score": 10, "mismatch_score": 0},
        {"name": "MQ7_可后附“数”或作“论”的宾语", "desc": "可以后附名词性语素“数”(构成临时性的复合名词)或者用在动词“论”的后面作宾语(构成黏着性的述宾短语)", "match_score": 10, "mismatch_score": 0}
    ],
     # 3.9 复合量词
    "复合量词": [
        {"name": "CMQ1_不能单独回答", "desc": "(复合量词也是黏着词，)不能用来单独回答问题", "match_score": 20, "mismatch_score": -20},
        {"name": "CMQ2_可与数词组合", "desc": "可以直接用在数词(包括简单数词和合成数词)之后构成数量词组(然后主要作动词性成分(包括述宾短语)的准宾语，也可以作定语修饰名词性成分)", "match_score": 20, "mismatch_score": -20},
        {"name": "CMQ3_可与数词短语组合", "desc": "可以直接用在由位数词跟数量助词(包括前附助词“上”和后附助词“来/把”)构成的数词短语之后，一起构成数量词组", "match_score": 20, "mismatch_score": 0},
        {"name": "CMQ4_不能加“每(一)”表全称", "desc": "一般不能前加“每(一)”来表示全称", "match_score": 10, "mismatch_score": 0},
        {"name": "CMQ5_不能重叠表全称", "desc": "不能通过重叠来表示全称(再去修饰名词性成分或独立指称事物)", "match_score": 10, "mismatch_score": -10},
        {"name": "CMQ6_不能用于“这(一)/那(一)+量词+名词”结构", "desc": "不能用在代词“这(一)、那(一)”和名词性成分之间", "match_score": 10, "mismatch_score": 0},
        {"name": "CMQ7_不能后附“数”或作“论”的宾语", "desc": "不能后附名词性语素“数”(构成临时性的复合名词),并且，一般不能用在动词“论”的后面作宾语(构成黏着性的述宾短语)", "match_score": 10, "mismatch_score": 0}
    ],
     # 3.10 量词
    "量词": [
        {"name": "Q1_不能单独回答", "desc": "(量词是黏着词)不能用来单独回答问题", "match_score": 20, "mismatch_score": -20},
        {"name": "Q2_可与数词组合", "desc": "可以直接用在数词(包括简单数词和合成数词)之后构成数量词组(然后作定语修饰名词性成分或作动词性成分的准宾语)", "match_score": 20, "mismatch_score": -20},
        {"name": "Q3_可与数词短语组合", "desc": "可以直接用在由位数词和数量助词(包括前附助词“上”和后附助词“来、把”)构成的数词短语之后，一起构成数量词组", "match_score": 20, "mismatch_score": 0},
        {"name": "Q4_可与含“多/余”的数词短语组合", "desc": "可以直接用在由合成数词和数量助词“多、余”构成的数词短语之后，一起构成数量词组", "match_score": 20, "mismatch_score": -20},
    ],
     # 3.11 真数量词
    "真数量词": [
        {"name": "RQ1_可作定语且不受“很”修饰", "desc": "可以作定语修饰名词性成分，并且不能受形容词和副词“很”修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "RQ2_可转指且作准宾语", "desc": "可以通过转指(即转喻)来代替整个偏正结构(从而作主语或宾语；或者直接作动词(包括及物动词和不及物动词)或形容词的准宾语(表时量))", "match_score": 20, "mismatch_score": -20},
        {"name": "RQ3_定语位置", "desc": "作定语时可以出现在领属定语之后、带“的”的描写性定语之前", "match_score": 10, "mismatch_score": 0},
        {"name": "RQ4_可作谓语且不受补语修饰", "desc": "可以作谓语或谓语核心 (因而可以受状语修饰，并且不能受补语修饰)", "match_score": 10, "mismatch_score": -10},
        {"name": "RQ5_可表概数", "desc": "可以用在“大约”之后或“左右”之前，表示大概的数量", "match_score": 10, "mismatch_score": -10},
        {"name": "RQ6_不能与准数量词相互修饰", "desc": "不能受准数量词修饰，也不能修饰准数量词", "match_score": 10, "mismatch_score": -10},
        {"name": "RQ7_不能构成合成数词", "desc": "不能用在系数词之后或位数词之前，一起构成合成数词", "match_score": 10, "mismatch_score": -10},
        {"name": "RQ8_不能构成概数形式", "desc": "不能用在“上、几、数”之后或“来、把、多、余”之前，一起构成表示概数的复合形式", "match_score": 10, "mismatch_score": -10}
    ],
     # 3.12 准数量词
    "准数量词": [
        {"name": "PQ1_可作定语且不受“很”修饰", "desc": "可以作定语修饰名词性成分，并且不受形容词和副词“很”修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "PQ2_可转指且作准宾语", "desc": "可以通过转指(即转喻)来代替整个偏正结构(从而作主语或宾语，或者直接作动词(包括及物动词和不及物动词)或形容词的准宾语(表时量))", "match_score": 20, "mismatch_score": 0},
        {"name": "PQ3_定语位置", "desc": "作定语时可以出现在领属定语之后、带“的”的描写性定语之前", "match_score": 10, "mismatch_score": 0},
        {"name": "PQ4_不能作谓语", "desc": "不能作谓语和谓语核心(因而不能受状语和补语修饰)", "match_score": 10, "mismatch_score": -10},
        {"name": "PQ5_不能与真数量词相互修饰", "desc": "不能受真数量词修饰，也不能修饰真数量词", "match_score": 10, "mismatch_score": -10},
        {"name": "PQ6_不能构成合成数词", "desc": "不能用在系数词之后或位数词之前，一起构成合成数词", "match_score": 10, "mismatch_score": -10},
        {"name": "PQ7_不能构成序数组合", "desc": "不能用在前缀“第”之后，一起构成序数组合", "match_score": 10, "mismatch_score": -10},
        {"name": "PQ8_不能构成概数形式", "desc": "不能用在“上、几、数”之后或“来、把、多、余”之前，一起构成表示概数的复合形式", "match_score": 10, "mismatch_score": -10}
    ],
     # 4.1 表人名
    "表人名": [
        {"name": "PN1_不受数量词和副词修饰", "desc": "一般不能受数量词修饰，并且不能受副词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "PN2_可作主语宾语且可构成“的”字结构", "desc": "可以作典型的主语和宾语，并且可以后附助词“的”构成“的”字结构(然后作主语、宾语、定语)", "match_score": 10, "mismatch_score": -10},
        {"name": "PN3_可作同位性定语或中心语", "desc": "可以作中心语受同位性定语修饰和作同位性定语修饰其他名词", "match_score": 20, "mismatch_score": -20},
        {"name": "PN4_不能作谓语", "desc": "不能作谓语和谓语核心(因而不能带宾语，也不能受状语和补语的修饰，也不能后附时体助词“着、了、过”)", "match_score": 10, "mismatch_score": -10},
        {"name": "PN5_不能作补语状语且不能构成处所结构", "desc": "不能作补语，也不能作状语直接修饰动词性成分，并且不能后附方位词构成处所结构(然后作“在、到、从”等介词的宾语)", "match_score": 10, "mismatch_score": -10},
        {"name": "PN6_可被“谁/哪个人”提问且可被“他/这个人/那个人”指代", "desc": "可以用“谁”或 “哪个人”提问，并且可以用“他”或 “这个人、那个人”指代(即可以用“谁、哪个人”来替换表人名词，从而构成一个疑问句；并且针对有“谁、哪个人”的句子，可以用表人名词“他、这个人、那个人”来回答)", "match_score": 10, "mismatch_score": -10},
        {"name": "PN7_可与“他”复指且可后附“他们”表复数", "desc": "可以后附或前附“他”来复指，并且可以后附“他们”来表示复数", "match_score": 30, "mismatch_score": -30}
    ],
     # 4.2 单纯方位词
    "单纯方位词": [
        {"name": "SP1_不能单独回答", "desc": "(常用的单纯方位词有“上、下、前、后、里、外、左、右、东、西、南、北、内、中”等。单纯方位词是黏着词，)不能用来单独回答问题", "match_score": 20, "mismatch_score": -20},
        {"name": "SP2_有限句法功能", "desc": "(作句法成分的能力很弱)不能作主语、宾语、谓语、状语和补语等句法成分(只有在习用的对举格式中，可以作主语、宾语和状语)", "match_score": 10, "mismatch_score": 0},
        {"name": "SP3_可作介词宾语", "desc": "可以作“向、从、往”等介词的宾语，或者可以填入介词框架“从……到/向/往……”中", "match_score": 20, "mismatch_score": -20},
        {"name": "SP4_可后附构成处所结构", "desc": "可以后附在名词性成分之后构成处所结构(然后作 “在、到、从”等介词的宾语，这种介词结构又可以作状语或补语修饰动词性成分)", "match_score": 10, "mismatch_score": -10},
        {"name": "SP5_可后附时段数量词构成时间结构", "desc": "可以后附在表示时段 (duration) 的数量词之后，构成一个时间结构。（(i) 直接作话题(后面再带上一个动词性的说明成分)。；(ii) 作“在 、 到 、从”等介词的宾语(然后这种介词结构又可以作状语或补语修饰动词性成分)。；(iii) 后附助词“的”构成“的”字结构(然后作定语 修饰名词)。）", "match_score": 10, "mismatch_score": 0},
        {"name": "SP6_可修饰范围数量词构成部分结构", "desc": "可以作定语修饰表示范围 (range) 的数量词，构成一个部分结构(然后直接作主语或话题，后面再带上一个动词性谓语或说明成分，或者作定语修饰名词)", "match_score": 10, "mismatch_score": 0},
        {"name": "SP7_不能构成“的”字结构且可受“最”修饰", "desc": "不能后附助词“的”构成“的”字结构，也不能受“的”字结构的修饰。一般可以受区别词“最”修饰，构成“最 … … 到……”格式", "match_score": 10, "mismatch_score": -10}
    ],
     # 4.3 合成方位词
    "合成方位词": [
        {"name": "CP1_可以单独回答", "desc": "(合成方位词主要是由单纯方位词加上后缀“边儿、面儿、头儿”构成的。例如：“上边儿、下面儿、前头儿、后边儿、里面儿、外头儿、左面儿、右边儿、东面儿、南头儿、西面儿、北边儿”。合成方位词是自由词)可以用来单独回答问题", "match_score": 20, "mismatch_score": -20},
        {"name": "CP2_可作多种句法成分", "desc": "可以作主语、宾语、定语或状语等句法成分", "match_score": 20, "mismatch_score": -20},
        {"name": "CP3_可作介词宾语", "desc": "可以作“向、在、到、从、往”等介词的宾语，或者可以填入介词框架“从……到/向/往……”中", "match_score": 10, "mismatch_score": -10},
        {"name": "CP4_可后附构成处所结构", "desc": "可以后附在名词性成分之后构成处所结构(然后作“在、到、从”等介词的宾语，这种介词结构又可以作状语或补语修饰动词性成分)", "match_score": 10, "mismatch_score": -10},
        {"name": "CP5_不能后附时段数量词构成时间结构", "desc": "不能后附在表示时段 (duration)的数量词之后，构成一个时间结构", "match_score": 10, "mismatch_score": 0},
        {"name": "CP6_不能修饰范围数量词", "desc": "不能作定语修饰表示范围 (range) 的数量词", "match_score": 10, "mismatch_score": 0},
        {"name": "CP7_可构成“的”字结构或受“的”字结构修饰", "desc": "可以后附助词“的”构成“的”字结构(然后作主语、宾语和定语)。或者，可以受“的”字结构的修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "CP8_可受“最”修饰", "desc": "一般可以受区别词“最”修饰，构成“最……”短语 (这种短语可以作主语、宾语，还可以作“向、在、到、从、往”等介词的宾语，或者可以填入介词框架“从……到/向/往…… ” 中)", "match_score": 10, "mismatch_score": 0}
    ],
     # 4.4 助动词
    "助动词": [
        {"name": "M1_受“不”修饰且不受“没有”修饰", "desc": "可以受否定副词“不”修饰，并且一般不能受“没有”修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "M2_不能附时体助词且不能进入“……了没有”格式", "desc": "不能后附或中间插入时体助词“着、了、过”,也不能进入“……了没有”格式", "match_score": 10, "mismatch_score": -10},
        {"name": "M3_只能带谓词宾语", "desc": "只能带谓词宾语，并且不能带体词宾语和数量宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "M4_可受“很”修饰或有“不V不”双重否定", "desc": "或者可以受程度副词“很”等修饰，或者可以有“不V不”式双重否定", "match_score": 10, "mismatch_score": 0},
        {"name": "M5_无重叠形式且有正反重叠形式", "desc": "没有 “VV 、V 一 V 、V 了 V” 等重叠形式，并且有“V 不V” 这种正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "M6_可作谓语", "desc": "可以作谓语或谓语核心(因而可以受状语和补语修饰)", "match_score": 10, "mismatch_score": -10},
        {"name": "M7_不能作状语", "desc": "不能作状语直接修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "M8_不能跟在“怎么/怎样/这么/这样/那么/那样”之后", "desc": "不能跟在“怎么、怎样”之后，对动作的方式进行提问；也不能跟在“这么、这样、那么、那样”之后，用以作出相应的回答", "match_score": 10, "mismatch_score": -10},
        {"name": "M9_不能跟在“多/多么”之后", "desc": "不能跟在“多”之后，对性质的程度进行提问；也不能跟在“多么”之后，表示感叹", "match_score": 10, "mismatch_score": -10},
        {"name": "M10_不能用在否定祈使句中", "desc": "不能用在带“别、甭”的否定式祈使句中", "match_score": 10, "mismatch_score": -10}
    ],
     # 4.5 形式动词
    "形式动词": [
        {"name": "FV1_带宾语后可受“不/没(有)”修饰", "desc": "(一般只有)带上宾语以后(才)能受否定副词“不” 或“没(有)”修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "FV2_可附时体助词且带宾语后可进入“……了没有”格式", "desc": "可以后附时体助词“着、了、过”,并且 (一般只有)带上宾语以后(才)能进入“ … … 了没有”格式", "match_score": 10, "mismatch_score": 0},
        {"name": "FV3_可带名动词或名形词作宾语", "desc": "可以带名动词或名形词作宾语(这种作宾语的名动词或名形词一般只能受名词性成分修饰，不能受形容词和副词等词类修饰)", "match_score": 10, "mismatch_score": 0},
        {"name": "FV4_不受“很”修饰", "desc": "不能受程度副词“很”等修饰。带上宾语以后仍然不能受“很”等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "FV5_无重叠形式", "desc": "不能有 “VV 、V 一 V 、V 了 V” 等重叠形式", "match_score": 10, "mismatch_score": -10},
        {"name": "FV6_可作谓语且带宾语后可受状语修饰", "desc": "可以作谓语或谓语核心(带上宾语以后可以受状语修饰，但是不能带补语)", "match_score": 10, "mismatch_score": -10},
        {"name": "FV7_带宾语后可跟在“怎么/怎样/这么/这样/那么/那样”之后", "desc": "带上宾语以后可以跟在“怎么、怎样”之后，对动作的方式进行提问，并且可以跟在“这么、这样、那么、那样”之后，用以作出相应的回答", "match_score": 10, "mismatch_score": 0},
        {"name": "FV8_不能作状语", "desc": "不能作状语直接修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "FV9_带宾语后不能跟在“多/多么”之后", "desc": "带上宾语以后仍然不能跟在“多”之后，对性质的程度进行提问；也不能跟在“多么”之后，表示感叹", "match_score": 10, "mismatch_score": -10},
        {"name": "FV10_带宾语后不能构成祈使句", "desc": "带上宾语后仍然不能构成肯定式祈使句，并且一般也不能跟在“别、甭”之后构成否定式祈使句", "match_score": 10, "mismatch_score": 0}
    ],
    # 4.6 名动词
    "名动词": [
        {"name": "NV1_可被\"不/没有\"否定且肯定形式", "desc": "可以用\"不\"和\"没有\"来否定，并且\"没有……\"的肯定形式可以是\"……了\"和\"有……\"(前一种情况中的\"没有\"是副词，后一种情况中的\"没有\"是动词)", "match_score": 10, "mismatch_score": -10},            {"name": "NV2_可附时体助词或进入\"……了没有\"格式", "desc": "可以后附时体助词\"着、了、过\"，或者可以进入\"………了没有\"格式", "match_score": 10, "mismatch_score": -10},
        {"name": "NV3_可带真宾语且不受\"很\"修饰", "desc": "可以带真宾语，并且不能受程度副词\"很\"等修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "NV4_有重叠和正反重叠形式", "desc": "可以有\"VV、V一V、V了V、V不V\"等重叠和正反重叠形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NV5_可作多种句法成分且可作形式动词宾语", "desc": "既可以作谓语或谓语核心，又可以作主语或宾语，并且，可以作形式动词\"作、进行、加以、给予、受到\"等的宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NV6_不能直接作状语", "desc": "不能直接作状语修饰动词性成分", "match_score": 10, "mismatch_score": -10},
        {"name": "NV7_可修饰名词或受名词/数量词修饰", "desc": "可以修饰名词或者受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NV8_可跟在\"怎么/怎样/这么/这样/那么/那样\"之后", "desc": "可以跟在\"怎么、怎样\"之后，对动作的方式进行提问，并且可以跟在\"这么、这样、那么、那样\"之后，用以作出相应的回答", "match_score": 10, "mismatch_score": 0},
        {"name": "NV9_不能跟在\"多/多么\"之后", "desc": "不能跟在\"多\"之后，对性质的程度进行提问，也不能跟在\"多么\"之后，表示感叹", "match_score": 10, "mismatch_score": -10}
    ],
    # 4.7 名形词
    "名形词": [
        {"name": "NA1_可被\"不/没有\"否定且肯定形式", "desc": "可以用\"不\"和\"没有\"来否定，并且\"没有……\"的肯定形式可以是\"……了\"和\"有……\"(前一种情况中的\"没有\"是副词，后一种情况中的\"没有\"是动词)", "match_score": 10, "mismatch_score": -10},
        {"name": "NA2_可受\"很\"修饰且不能直接带单宾语", "desc": "可以受程度副词\"很\"等修饰，并且不能直接带单宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "NA3_可作多种句法成分且可作\"有/没有\"宾语", "desc": "既可以作谓语或谓语核心，又可以作主语或宾语，并且，又可以作动词\"有\"和\"没有\"的宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "NA4_可修饰名词或受名词/数量词修饰", "desc": "可以直接修饰名词性成分，或者可以受名词修饰，或者可以受数量词修饰", "match_score": 10, "mismatch_score": 0},
        {"name": "NA5_可修饰动词性成分", "desc": "可以独立或者构成复杂形式(如：前加\"很、更\"等副词、后加\"地\")修饰动词性成分", "match_score": 10, "mismatch_score": 0},
        {"name": "NA6_可作补语或带补语形式", "desc": "可以作补语，或者可以带\"得很、极了\"等补语形式", "match_score": 10, "mismatch_score": 0},
        {"name": "NA7_可用于\"比\"字句或\"越来越……\"格式", "desc": "可以作\"比\"字句的谓语核心，或者可以用在\"越来越……\"格式中", "match_score": 10, "mismatch_score": 0},
        {"name": "NA8_可跟在\"多/这么/这样/那么/那样/多么\"之后", "desc": "可以跟在\"多\"之后，对性质的程度进行提问；并且可以跟在\"这么、这样、那么、那样\"之后，用以作出相应的回答；或者可以跟在\"多么\"之后，表示感叹", "match_score": 10, "mismatch_score": 0},
        {"name": "NA9_可后附方位词构成处所结构", "desc": "可以后附方位词构成处所结构(然后作\"在、到、从\"等介词的宾语，这种介词结构又可以作状语或补语修饰动词性成分)", "match_score": 10, "mismatch_score": 0}
    ],
    # 4.8 后补词
    "后补词": [
        {"name": "SC1_不受\"不/没有\"修饰", "desc": "不能受否定副词\"不\"和\"没有\"修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "SC2_不能附时体助词且不能进入\"……了没有\"格式", "desc": "不能后附或中间插入时体助词\"着、了、过\",并且不能进入\"……了没有\"格式", "match_score": 10, "mismatch_score": -10},
        {"name": "SC3_不能作多种句法成分", "desc": "不能作谓语和谓语核心，也不能作主语和宾语，并且不能作定语和状语及其中心语", "match_score": 10, "mismatch_score": -10},
        {"name": "SC4_可作补语", "desc": "可以作补语(包括结果补语、可能补语或程度补语)", "match_score": 50, "mismatch_score": -50},
        {"name": "SC5_不受\"很\"修饰且无重叠和正反重叠形式", "desc": "不能受程度副词\"很\"等修饰，并且不能有 \"VV 、V 一 V 、V了 V 、V 不 V\" 等重叠和正反重叠形式", "match_score": 10, "mismatch_score": -10},
        {"name": "SC6_不能跟在\"怎么/怎样/这么/这样/那么/那样/多/多么\"之后", "desc": "不能跟在\"怎么、怎样\"之后，对动作的方式进行提问；也不能跟在\"这么、这样、那么、那样\"之后，用以作出相应的回答；并且不能跟在\"多\"之后，对性质的程度进行提问；也不能 跟在\"多么\"之后，表示感叹", "match_score": 10, "mismatch_score": -10}
    ],
     # 4.9 修饰词
    "修饰词": [
        {"name": "Mod1_可直接作定语", "desc": "可以直接作定语修饰名词性成分", "match_score": 20, "mismatch_score": -20},
        {"name": "Mod2_可构成\"的\"字结构", "desc": "可以后附助词\"的\"构成\"的\"字结构(然后作主语、宾语或定语)", "match_score": 10, "mismatch_score": 0},
        {"name": "Mod3_可独立作状语", "desc": "可以独立作状语修饰动词性成分", "match_score": 20, "mismatch_score": -20},
        {"name": "Mod4_不能作主语宾语", "desc": "不能作主语和宾语", "match_score": 10, "mismatch_score": -10},
        {"name": "Mod5_不能作谓语", "desc": "不能作谓语和谓语核心", "match_score": 10, "mismatch_score": -10},
        {"name": "Mod6_不能作补语且不能构成处所结构", "desc": "不能作补语，也不能后附方位词构成处所结构(然后作 \"在、到、从\"等介词的宾语)", "match_score": 10, "mismatch_score": -10},
        {"name": "Mod7_不受\"很/不\"修饰", "desc": "不能受\"很\"和 \"不\"等一切副词修饰", "match_score": 10, "mismatch_score": -10},
        {"name": "Mod8_不能单独回答", "desc": "不能单独回答问题(即是黏着词)", "match_score": 10, "mismatch_score": -10}
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
        # --- 新增：处理通义千问 (Qwen) 的响应格式 ---
        if "output" in resp_json and "text" in resp_json["output"]:
            return resp_json["output"]["text"]
            
        # --- 原有的：处理 OpenAI 系列的响应格式 ---
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
    
def extract_json_from_text(text: str) -> Tuple[dict, str]:
    if not text: return None, ""
    text = text.strip()
    # 尝试直接解析
    try: return json.loads(text), text
    except: pass
    
    # 尝试提取文本中的JSON块
    match = re.search(r"(\{[\s\S]*\})", text)
    if not match: return None, text
    
    json_str = match.group(1)
    # 清理常见的格式问题
    json_str = json_str.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
    json_str = re.sub(r"'(\s*[^']+?\s*)'\s*:", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*?)'", r': "\1"', json_str)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str) # 去除 trailing commas
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)
    
    try: return json.loads(json_str), json_str
    except Exception as e:
        st.warning(f"解析JSON失败: {e}")
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

    # 优化1：筛选每个词类的核心规则（match_score≥20），减少传输量
    core_rules_text = "\n".join([
        f'"{pos}": {{' + ', '.join([f'"{r["name"]}": {r["match_score"]}' for r in rules if r["match_score"] >= 20]) + '}' 
        for pos, rules in RULE_SETS.items()
    ])
    core_rules_text = "{\n" + core_rules_text + "\n}"

    # 优化2：完整规则仅保留候选词类的，通过思维链分阶段处理
    full_rules_by_pos = {
        pos: "\n".join([f'"{r["name"]}": {r["match_score"]}' for r in rules])
        for pos, rules in RULE_SETS.items()
    }

    # 优化3：分阶段提示词，引入思维链，先筛选再评分
    system_msg = f"""你是一位中文语言学专家。你的任务是根据提供的规则，为给定的词语「{word}」进行词类隶属度评分。请严格按以下步骤操作：

#### 步骤1：初步筛选候选词类（必须包含此思考过程）
1. 快速分析词语「{word}」的语法特征（如能否受数量词修饰、能否带宾语等）
2. 参考以下核心规则（仅展示match_score≥20的关键规则），筛选出最可能的5个候选词类：
{core_rules_text}
3. 说明筛选理由（如："排除动词，因为「{word}」不能带宾语，不符合V3规则"）

#### 步骤2：对候选词类进行完整评分
1. 针对步骤1筛选出的候选词类，使用该词类的全部规则进行评分（符合规则填对应match_score，否则填0）
2. 每个词类包含多个规则，每个规则有明确的match_score（符合得分）和mismatch_score（不符合得分）。
3. 必须严格使用规则中定义的分数，** 不符合规则时必须使用负分（如-20、-10），绝对不能用0分代替 **。
4. 候选词类的完整规则如下（仅使用你筛选出的词类对应的规则）：
"""
    # 拼接所有词类的完整规则（供模型在步骤2使用）
    for pos, rules_str in full_rules_by_pos.items():
        system_msg += f'\n{pos}的完整规则：\n{{{rules_str}}}'
    
    system_msg += f"""

#### 步骤3：返回最终结果（仅输出JSON，无其他文字）
请严格按照以下格式返回，确保JSON完整且格式正确：
{{
  "predicted_pos": "最可能的词类名称（从候选词类中选择）",
  "scores": {{
    "候选词类1": {{ "规则1": 得分, "规则2": 得分, ... }},
    "候选词类2": {{ "规则1": 得分, "规则2": 得分, ... }},
    ...
  }},
  "explanation": "简要说明判定为最可能词类的主要依据（1-2句话）"
}}

关键说明：
1. 步骤1的筛选过程必须在思考中体现，帮助你聚焦核心词类
2. 步骤2仅评分候选词类，无需处理所有词类，减少计算量
3. 确保"scores"中的规则名称与提供的完全一致（如"N1_可受数量词修饰"）
4. 若候选词类不足5个，按实际数量评分；若所有词类均不符合，保留得分最高的3个
5. 严格使用规则定义的mismatch_score（包括负分），不符合规则时禁止用0分替代
"""

    # 用户提示仅需触发模型开始分析
    user_prompt = f"请根据上述步骤，为词语「{word}」进行词类隶属度评分并返回JSON结果。"

    # 显示加载状态
    with st.spinner("正在调用大模型进行分析，请稍候..."):
        # 使用缓存调用API
        ok, resp_json, err_msg = call_llm_api_cached(
            _provider=provider,
            _model=model,
            _api_key=api_key,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_prompt}]
        )

    if not ok:
        st.error(f"模型调用失败: {err_msg}")
        return {}, f"调用失败: {err_msg}", "未知", f"调用失败: {err_msg}"

    raw_text = extract_text_from_response(resp_json)
    parsed_json, cleaned_json_text = extract_json_from_text(raw_text)
    
    # 处理解析结果
    if parsed_json:
        explanation = parsed_json.get("explanation", "模型未提供详细推理过程。")
        predicted_pos = parsed_json.get("predicted_pos", "未知")
        raw_scores = parsed_json.get("scores", {})
    else:
        st.warning("未能从模型响应中解析出有效的JSON。")
        explanation = "无法解析模型输出。"
        predicted_pos = "未知"
        raw_scores = {}
        cleaned_json_text = raw_text  # 展示原始文本

    # --- 关键修复：在循环开始前，初始化 scores_out ---
    # 为了避免 KeyError，先为每个词类（pos）在 scores_out 中创建一个空字典
    scores_out = {pos: {} for pos in RULE_SETS.keys()}

    # 格式化得分（确保所有词类的规则都有对应条目，未评分的规则填0）
    # 改为：认可匹配得分或不匹配得分（包括负分）
    for pos, rules in RULE_SETS.items():
        raw_pos_scores = raw_scores.get(pos, {})
        if isinstance(raw_pos_scores, dict):
            for k, v in raw_pos_scores.items():
                normalized_key = normalize_key(k, rules)
                if normalized_key:
                    # 查找当前规则的定义
                    rule_def = next(r for r in rules if r["name"] == normalized_key)
                    # 关键修改：使用 map_to_allowed_score 函数处理得分，保留负分
                    scores_out[pos][normalized_key] = map_to_allowed_score(rule_def, v)
    
    # 循环结束后，确保所有规则都有一个得分（未被模型评分的规则，其得分为0）
    for pos, rules in RULE_SETS.items():
        for rule in rules:
            rule_name = rule["name"]
            # 如果规则在 scores_out 中没有得分，则默认为0
            if rule_name not in scores_out[pos]:
                scores_out[pos][rule_name] = 0
    
    return scores_out, cleaned_json_text, predicted_pos, explanation
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
        
      # 假设 col_results_1 和 col_results_2 是通过 st.columns() 创建的
# col_results_1, col_results_2 = st.columns(2)

with col_results_1:
    st.subheader("🏆 词类隶属度排名（前十）")
    top10 = get_top_10_positions(membership)
    top10_df = pd.DataFrame(top10, columns=["词类", "隶属度"])
    top10_df["隶属度"] = top10_df["隶属度"].apply(lambda x: f"{x:.4f}")
    st.table(top10_df)
    
    st.subheader("📊 词类隶属度雷达图（前十）")
    plot_radar_chart_streamlit(dict(top10), f"「{word}」的词类隶属度分布")

with col_results_2:
    # 关键修复：这一行以及其下的所有内容都必须缩进，属于 col_results_2 代码块
    st.subheader("📋 各词类详细得分（按总分排名前10）")
    
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
            
            # 负分标红，动态调整高度（确保所有规则都能显示）
            styled_df = rule_df.style.applymap(
                lambda x: "color: #ff4b4b; font-weight: bold" if isinstance(x, int) and x < 0 else "",
                subset=["得分"]
            )
            
            # 调整表格高度，确保所有规则都能显示
            min_height = len(rule_df) * 30 + 50  # 50px为表头高度
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(min_height, 800)  # 最大高度限制为800px
            )
    
    # 这些元素也属于 col_results_2，保持正确缩进
    st.subheader("🔍 模型推理过程")
    st.text_area("推理详情", explanation, height=200, disabled=True)
    
    st.subheader("📥 模型原始响应")
    with st.expander("点击展开查看原始响应", expanded=False):
        st.code(raw_text, language="json")



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

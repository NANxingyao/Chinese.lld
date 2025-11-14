import streamlit as st
import requests
import json
import re
import os
import pandas as pd
import plotly.graph_objects as go
import random
from typing import Tuple, Dict, Any

# ===============================
# 页面配置
# ===============================
st.set_page_config(
    page_title="汉语词类隶属度检测",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
hide_streamlit_style = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.stButton>button {width: 100%;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 模型配置
# ===============================
MODEL_OPTIONS = {
    "本地模拟（推荐测试）": {
        "provider": "mock",
        "model": "mock-model",
        "api_url": "local-mock",
        "api_key": "mock-key",
    },
    "DeepSeek Chat": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key": os.getenv("DEEPSEEK_API_KEY", "sk-1f346646d29947d0a5e29dbaa37476b8"),
    },
    "OpenAI GPT-4o": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "api_key": os.getenv("OPENAI_API_KEY", "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"),
    },
    "Moonshot（Kimi）": {
        "provider": "moonshot",
        "model": "moonshot-v1-32k",
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "api_key": os.getenv("MOONSHOT_API_KEY", "sk-l5FvRWegjM5DEk4AU71YPQ1QgvFPTHZIJOmq6qdssPY4sNtE"),
    },
    "Doubao（豆包）": {
        "provider": "doubao",
        "model": "doubao-pro-32k",
        "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "api_key": os.getenv("DOUBAO_API_KEY", "222afa3f-5f27-403e-bf46-ced2a356ceee"),
    },
    "Qwen（通义千问）": {
        "provider": "qwen",
        "model": "qwen-plus",
        "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "api_key": os.getenv("QWEN_API_KEY", "sk-b3f7a1153e6f4a44804a296038aa86c5"),
    }
}

# ===============================
# 词类规则集
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
    # 3.3 代词（通用代词条目示例）
    "代词": [
        {"name": "DPR1_可作典型主宾语", "desc": "可做典型主语或宾语", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR2_不能受数量/形容/的修饰", "desc": "不能受数量词、形容词和'的'字结构修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR3_不能受程度副词修饰", "desc": "不能受'很'等程度副词修饰", "match_score": 20, "mismatch_score": -20},
        {"name": "DPR4_不能带宾语和补语", "desc": "不能带宾语和补语", "match_score": 20, "mismatch_score": 0},
        {"name": "DPR5_可受'不/也'等副词修饰（针对谓代）或不能后附方位（针对体代）", "desc": "混合规则，按具体代词类型判定", "match_score": 20, "mismatch_score": -20},
    ],
    # 3.4 系数词、位数词、合成数词等：
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

# 计算最大可能分数
MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 本地模拟数据
# ===============================
def get_mock_response(word: str) -> dict:
    """获取模拟响应"""
    if word in MOCK_DATA:
        return MOCK_DATA[word]
    
    # 默认预测规则
    predicted_pos = "名词"  # 默认
    
    if word.endswith(("很", "不", "都", "也", "就", "只")):
        predicted_pos = "副词"
    elif word.endswith(("子", "儿", "头", "们")):
        predicted_pos = "名词"
    elif word.endswith(("着", "了", "过")):
        predicted_pos = "动词"
    elif word in ("美丽", "漂亮", "高兴", "难过", "伟大"):
        predicted_pos = "形容词"
    else:
        # 随机选择，但更倾向于名词和动词
        predicted_pos = random.choices(["名词", "动词", "形容词", "副词"], weights=[0.4, 0.3, 0.2, 0.1])[0]
    
    # 生成相应的规则得分
    scores = {}
    for pos in RULE_SETS.keys():
        scores[pos] = {}
        for rule in RULE_SETS[pos]:
            # 如果是预测的词类，匹配更多规则
            if pos == predicted_pos:
                scores[pos][rule["name"]] = random.random() > 0.3  # 70%概率匹配
            else:
                scores[pos][rule["name"]] = random.random() > 0.7  # 30%概率匹配
    
    return {
        "predicted_pos": predicted_pos,
        "scores": scores,
        "explanation": f"'{word}' 被预测为{predicted_pos}"
    }

# ===============================
# 核心工具函数
# ===============================
def extract_text_from_response(resp_json: Dict[str, Any], provider: str = "") -> str:
    """从不同模型的响应中提取文本内容"""
    try:
        if not isinstance(resp_json, dict):
            return f"响应格式错误: {type(resp_json)}"
        
        # 本地模拟
        if provider == "mock":
            return json.dumps(resp_json, ensure_ascii=False)
        
        # 通义千问
        if provider == "qwen":
            output = resp_json.get("output", {})
            text = output.get("text", "")
            if text:
                return text
            return f"通义千问响应格式不匹配: {str(resp_json)[:500]}"
        
        # 其他模型
        choices = resp_json.get("choices", [])
        if not choices:
            return f"未找到choices字段: {str(resp_json)[:500]}"
        
        first_choice = choices[0]
        if "message" in first_choice and isinstance(first_choice["message"], dict):
            return first_choice["message"].get("content", "")
        elif "content" in first_choice:
            return first_choice["content"]
        elif "text" in first_choice:
            return first_choice["text"]
        else:
            return f"未找到内容字段: {str(first_choice)[:500]}"
    
    except Exception as e:
        return f"解析响应时出错: {str(e)}"

def extract_json_from_text(text: str) -> Tuple[dict, str]:
    """从文本中提取JSON数据"""
    if not text:
        return None, ""
    
    try:
        return json.loads(text), text
    except:
        # 尝试提取JSON对象
        match = re.search(r"(\{[\s\S]*\})", text)
        if not match:
            return None, text
        
        candidate = match.group(1)
        # 清理常见的格式问题
        cleaned = candidate.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
        cleaned = re.sub(r"'(\w+?)'\s*:", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        
        try:
            return json.loads(cleaned), cleaned
        except Exception as e:
            return None, text

def normalize_key(key: str, rules: list) -> str:
    """标准化规则名称"""
    if not isinstance(key, str):
        return None
    
    key_clean = re.sub(r'\s+', '', key).upper()
    for rule in rules:
        rule_name_clean = re.sub(r'\s+', '', rule["name"]).upper()
        if key_clean == rule_name_clean:
            return rule["name"]
    return None

def map_to_score(rule: dict, value) -> int:
    """将原始值映射为规则分数"""
    if isinstance(value, bool):
        return rule["match_score"] if value else rule["mismatch_score"]
    
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("yes", "y", "true", "是", "符合"):
            return rule["match_score"]
        elif value_lower in ("no", "n", "false", "否", "不符合"):
            return rule["mismatch_score"]
    
    return rule["mismatch_score"]

# ===============================
# API调用函数
# ===============================
def call_llm_api(messages: list, provider: str, model: str, api_key: str, api_url: str) -> Tuple[bool, dict, str]:
    """调用LLM API"""
    # 本地模拟
    if provider == "mock":
        try:
            user_msg = messages[-1]["content"]
            word_match = re.search(r'分析词语：([^"\n]+)', user_msg)
            word = word_match.group(1).strip() if word_match else "测试"
            mock_response = get_mock_response(word)
            return True, {
                "choices": [{"message": {"content": json.dumps(mock_response, ensure_ascii=False)}}]
            }, "本地模拟成功"
        except Exception as e:
            return False, {"error": str(e)}, f"本地模拟错误: {str(e)}"
    
    # 检查API密钥
    if not api_key:
        return False, {"error": "API密钥为空"}, "请配置API密钥"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # 构建请求体
        if provider == "qwen":
            payload = {
                "model": model,
                "input": {"messages": messages},
                "parameters": {"max_tokens": 512, "temperature": 0.0}
            }
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.0,
                "stream": False
            }
        
        # 发送请求
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"HTTP错误 {response.status_code}: {response.text[:500]}"
            return False, {"error": error_msg}, error_msg
        
        return True, response.json(), ""
        
    except requests.exceptions.Timeout:
        return False, {"error": "请求超时"}, "API调用超时"
    except requests.exceptions.ConnectionError:
        return False, {"error": "连接错误"}, "网络连接失败"
    except Exception as e:
        return False, {"error": str(e)}, f"API调用错误: {str(e)}"

# ===============================
# 词类分析函数
# ===============================
def analyze_word(word: str, provider: str, model: str, api_key: str, api_url: str) -> Tuple[Dict, str, str]:
    """分析词语的词类隶属度"""
    if not word.strip():
        return {}, "", "请输入有效词语"
    
    # 构建提示词
    rules_text = ""
    for pos, rules in RULE_SETS.items():
        rules_text += f"{pos}:\n"
        for rule in rules:
            rules_text += f"  - {rule['name']}: {rule['desc']}\n"
    
    system_prompt = (
        "你是语言学研究专家。请分析中文词语的词类归属。"
        "仅返回JSON格式结果，不要包含其他解释文字。"
        'JSON格式：{"predicted_pos":"<最可能的词类>","scores":{"<词类名>":{"<规则名>":true/false,...},...},"explanation":"简要说明"}'
    )
    
    user_prompt = f"""
分析词语：{word}

请基于以下词类规则，判断该词属于什么词类，并为每个规则评分（true/false）：

{rules_text}

仅返回JSON格式结果，不要包含其他内容。
    """.strip()
    
    # 调用API
    success, response, error = call_llm_api(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        provider=provider,
        model=model,
        api_key=api_key,
        api_url=api_url
    )
    
    if not success:
        return {}, f"API调用失败: {error}", "未知"
    
    # 解析响应
    raw_text = extract_text_from_response(response, provider)
    parsed_json, cleaned_text = extract_json_from_text(raw_text)
    
    if not parsed_json:
        return {}, f"无法解析响应: {raw_text}", "未知"
    
    # 处理结果
    scores = {}
    predicted_pos = parsed_json.get("predicted_pos", "未知")
    raw_scores = parsed_json.get("scores", {})
    
    for pos, rules in RULE_SETS.items():
        scores[pos] = {}
        pos_scores = raw_scores.get(pos, {})
        
        for rule in rules:
            rule_name = rule["name"]
            # 尝试多种方式匹配规则名称
            value = pos_scores.get(rule_name)
            if value is None:
                # 尝试标准化匹配
                for key in pos_scores.keys():
                    normalized_key = normalize_key(key, rules)
                    if normalized_key == rule_name:
                        value = pos_scores[key]
                        break
            
            scores[pos][rule_name] = map_to_score(rule, value) if value is not None else 0
    
    return scores, raw_text, predicted_pos

# ===============================
# 可视化函数
# ===============================
def plot_radar_chart(scores_norm: Dict[str, float], title: str):
    """绘制雷达图"""
    if not scores_norm or len(scores_norm) == 0:
        st.warning("没有数据可绘制")
        return
    
    categories = list(scores_norm.keys())
    values = [scores_norm[cat] for cat in categories]
    
    # 闭合雷达图
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=categories, fill="toself", name="隶属度")]
    )
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=dict(text=title, x=0.5, font=dict(size=16))
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 主应用函数
# ===============================
def main():
    """主应用入口"""
    # 页面标题
    st.markdown("<h1 style='text-align: center;'>📊 汉语词类隶属度检测</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>输入词语 → 分析词类归属 → 获取隶属度评分</p>", unsafe_allow_html=True)
    st.divider()
    
    # 侧边栏模型选择
    with st.sidebar:
        st.title("模型选择")
        model_choice = st.selectbox("选择分析模型", list(MODEL_OPTIONS.keys()))
        selected_model = MODEL_OPTIONS[model_choice]
        
        st.divider()
        st.info(f"当前模型：{model_choice}")
        st.info(f"模型ID：{selected_model['model']}")
        
        # API密钥配置提示
        if selected_model["provider"] != "mock" and not selected_model["api_key"]:
            st.error("⚠️ API密钥未配置")
            st.markdown("""
            **请设置环境变量：**
            - DeepSeek: `DEEPSEEK_API_KEY`
            - OpenAI: `OPENAI_API_KEY`
            - Moonshot: `MOONSHOT_API_KEY`
            - 豆包: `DOUBAO_API_KEY`
            - 通义千问: `QWEN_API_KEY`
            """)
    
    # 主内容区
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 快速示例
        st.subheader("快速示例")
        example_words = ["很", "跑", "美丽", "苹果", "学习"]
        example_cols = st.columns(len(example_words))
        selected_example = None
        
        for i, word in enumerate(example_words):
            if example_cols[i].button(word):
                selected_example = word
        
        # 输入框
        word_input = st.text_input(
            "输入词语",
            value=selected_example if selected_example else "",
            placeholder="请输入要分析的词语"
        )
        
        # 分析按钮
        if st.button("开始分析", use_container_width=True):
            word = word_input.strip()
            if not word:
                st.warning("请输入词语")
            else:
                with st.spinner("分析中..."):
                    # 获取模型配置
                    provider = selected_model["provider"]
                    model = selected_model["model"]
                    api_key = selected_model["api_key"]
                    api_url = selected_model["api_url"]
                    
                    # 分析词语
                    scores, raw_output, predicted_pos = analyze_word(
                        word, provider, model, api_key, api_url
                    )
                    
                    # 显示结果
                    st.divider()
                    st.subheader("分析结果")
                    st.markdown(f"**输入词：** `{word}`")
                    st.markdown(f"**预测词类：** **{predicted_pos}**")
                    
                    # 显示详细分数
                    if scores:
                        st.subheader("规则评分详情")
                        for pos, rule_scores in scores.items():
                            st.markdown(f"**{pos}**")
                            total_score = sum(rule_scores.values())
                            max_score = MAX_SCORES.get(pos, 1)
                            normalized_score = round(total_score / max_score, 3) if max_score != 0 else 0
                            
                            # 显示总分和归一化分数
                            st.markdown(f"总分: {total_score}/{max_score} ({normalized_score})")
                            
                            # 显示规则详情
                            rule_df = pd.DataFrame({
                                "规则": list(rule_scores.keys()),
                                "得分": list(rule_scores.values())
                            })
                            st.dataframe(rule_df, use_container_width=True)
                            st.divider()
                        
                        # 计算隶属度排名
                        st.subheader("词类隶属度排名")
                        pos_totals = {pos: sum(scores[pos].values()) for pos in scores}
                        pos_normalized = {}
                        
                        for pos, total in pos_totals.items():
                            max_score = MAX_SCORES.get(pos, 1)
                            pos_normalized[pos] = round(max(0, total) / max_score, 3) if max_score != 0 else 0
                        
                        # 排序并显示
                        sorted_pos = sorted(pos_normalized.items(), key=lambda x: x[1], reverse=True)
                        for i, (pos, score) in enumerate(sorted_pos):
                            st.markdown(f"{i+1}. **{pos}**: {score}")
                        
                        # 雷达图
                        st.subheader("词类隶属度雷达图")
                        plot_radar_chart(pos_normalized, f"'{word}' 的词类隶属度分布")
                    
                    # 显示原始输出
                    st.subheader("原始输出")
                    st.text_area("模型原始响应", raw_output, height=200)

if __name__ == "__main__":
    main()

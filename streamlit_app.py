import streamlit as st
import requests
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re, json, requests, pandas as pd, numpy as np
from typing import Tuple, Dict, Any


# ===============================
# 配置
# ===============================
API_KEY = "sk-1f346646d29947d0a5e29dbaa37476b8"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

RULE_SETS = {
    
}


if not RULE_SETS:
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

# 计算每个词类的最大可能分（用于隶属度归一化）
MAX_SCORES = {pos: sum(abs(r["match_score"]) for r in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# DeepSeek 调用函数（保留原实现）
# ===============================
def call_deepseek_chat(messages: list, model: str = MODEL_NAME, max_tokens: int = 1024,
                       temperature: float = 0.0, timeout: int = 30, **kwargs) -> Tuple[bool, dict]:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
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
            for k in ("content","text","message"):
                if k in first and isinstance(first[k], str):
                    return first[k]
    except:
        pass
    return json.dumps(resp_json, ensure_ascii=False)

def extract_json_from_text(text: str) -> Tuple[dict,str]:
    if not text:
        return None,""
    s = text.strip()
    try:
        return json.loads(s), s
    except:
        # 尝试用正则提取最外层花括号
        m = re.search(r"(\{[\s\S]*\})", s)
        if not m:
            return None, s
        cand = m.group(1)
        c = cand.replace("：", ":").replace("，", ",").replace("“", '"').replace("”", '"')
        c = re.sub(r"'(\s*[^']+?\s*)'\s*:", r'"\1":', c)
        c = re.sub(r":\s*'([^']*?)'", r': "\1"', c)
        c = re.sub(r",\s*([}\]])", r"\1", c)
        c = re.sub(r"\bTrue\b","true",c)
        c = re.sub(r"\bFalse\b","false",c)
        c = re.sub(r"\bNone\b","null",c)
        try:
            return json.loads(c), c
        except:
            return None,s

def normalize_key(k: str, pos_rules: list) -> str:
    """把模型返回的键名标准化为规则中定义的 name（简单匹配）"""
    if not isinstance(k, str):
        return None
    kk = re.sub(r'\s+', '', k).upper()
    for r in pos_rules:
        if r["name"].upper() == kk or re.sub(r'\s+','',r["name"]).upper()==kk:
            return r["name"]
    # 允许模型返回短名（如 N1, V3 等）
    m = re.match(r'([A-Za-z\u4e00-\u9fff]+)(\d+)', k)
    if m:
        base = m.group(1)
        idx = m.group(2)
        for r in pos_rules:
            if r["name"].upper().startswith(base.upper()) and idx in r["name"]:
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
        s = raw_val.strip()
        if re.search(r"(-?\d+)", s):
            return map_to_allowed_score(rule, int(re.search(r"(-?\d+)", s).group(1)))
        low = s.lower()
        if low in ("yes","y","true","是","√","符合","yes"): return match
        if low in ("no","n","false","否","×","不符合","no"): return mismatch
        if any(x in low for x in ["能","可以","可","是","可作","可受"]): return match
        if any(x in low for x in ["不能","不","否","没有","不可以","不可"]): return mismatch
        return mismatch if mismatch != 0 else 0
    return mismatch if mismatch != 0 else 0

def query_deepseek(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, data=json.dumps(payload))
    try:
        return resp.json()['choices'][0]['message']['content']
    except:
        return "{}"

# ===============================
# 向心度计算函数（学术版）
# ===============================
def compute_centripetal_degree_rigorous(df: pd.DataFrame, return_all_centers: bool = False):
    """
    学术化的向心度计算（严格版）
    """
    df_work = df.copy()
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame 中无数值列。")
    df_num = df_work[numeric_cols].astype(float)

    max_role = df_num.idxmax(axis=1)
    max_value = df_num.max(axis=1)
    df_work['max_role'] = max_role
    df_work['max_value'] = max_value

    global_max = df_work['max_value'].max()
    center_chars = df_work.index[df_work['max_value'] == global_max].tolist()

    if not center_chars:
        raise RuntimeError("未能确定中心字。")
    if return_all_centers:
        chosen_centers = center_chars
    else:
        chosen_centers = [center_chars[0]]

    center_char = chosen_centers[0]
    center_value = df_work.at[center_char, 'max_value']
    center_role = df_work.at[center_char, 'max_role']

    series_role = df_num[center_role]
    n_total = len(series_role)
    series_others = series_role.drop(index=center_char)
    n_others = len(series_others)

    notes = ""
    if n_others >= 1:
        mean_others = series_others.mean()
        sd_others = series_others.std(ddof=1) if n_others > 1 else 0.0
    else:
        mean_others = float('nan')
        sd_others = float('nan')

    if (mean_others == 0) or np.isnan(mean_others):
        ratio = None
        notes += "其他字均值为0或不可用，Ratio 无法计算。"
    else:
        ratio = float(center_value / mean_others)

    if sd_others == 0 or np.isnan(sd_others):
        cohen_d = None
        notes += " 其他字标准差为0，Cohen's d 无定义。"
    else:
        cohen_d = float((center_value - mean_others) / sd_others)

    t_stat = None
    p_value = None
    if n_others >= 2 and not np.isnan(sd_others) and sd_others > 0:
        denom = sd_others / np.sqrt(n_others)
        t_stat = float((center_value - mean_others) / denom)
        dfree = n_others - 1
        try:
            from scipy import stats
            p_value = float(stats.t.sf(np.abs(t_stat), df=dfree) * 2)
        except Exception:
            p_value = None
            notes += " 无法计算 p 值（scipy.stats 不可用）。"
    else:
        notes += " n_others < 2，t 检验无法进行。"

    structure_judgement = "离心结构"
    if ratio is not None and ratio > 1:
        if cohen_d is None:
            structure_judgement = "向心结构（但效应量不可估）"
        else:
            if cohen_d >= 0.5:
                structure_judgement = "向心结构（中等或以上效应）"
            elif cohen_d >= 0.2:
                structure_judgement = "向心结构（小效应）"
            else:
                structure_judgement = "向心结构（效应极小）"

    result = {
        "结构判读": structure_judgement,
        "中心字": chosen_centers if return_all_centers else chosen_centers[0],
        "中心词类": center_role,
        "中心值": float(center_value),
        "Ratio": ratio,
        "Cohen_d": cohen_d,
        "t值": t_stat,
        "p值": p_value,
        "显著性": significance_label(p_value),
        "样本量": n_total,
        "备注": notes.strip()
    }
    return result


def significance_label(p):
    """返回显著性标注符号"""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."
    
# ===============================
# 主分析函数：整合隶属度 + 结构
# ===============================
def analyze_structure(word: str):
    chars = list(word)
    results = []

    for ch in chars:
        prompt = f"请判断单字「{ch}」在现代汉语中的词类潜势，输出JSON，如{{'名词':0.5,'动词':0.1,'形容词':0.3}}。"
        response = query_deepseek(prompt)
        try:
            data = json.loads(re.findall(r"\{.*\}", response)[0])
        except:
            data = {"名词": 0, "动词": 0, "形容词": 0, "副词": 0, "助词": 0}
        results.append(data)

    df = pd.DataFrame(results, index=chars).fillna(0)
    df.index.name = "字"

    # 自动计算结构信息
    struct_info = compute_centripetal_degree(df)
    return df, struct_info
    
# ===============================
# 本地启发式兜底（简化/可扩展）
# ===============================
_NOUN_EXAMPLES = {"人","学生","老师","厂长","学科","人员","权力","木头","中国","历史","教育","塑料","工资","被窝","群众","质量","饭盒","桌子","椅子","教授","思想"}
_VERB_EXAMPLES = {"跑","吃","学习","工作","买","卖","走","调整","增加","下放","联系","鼓励","知道","说","做","写"}
_ADVERB_EXAMPLES = {"很","非常","尤其","特别","已经","马上","立刻"}
_FUNCTION_WORD_EX = {"的","了","在","和","也","不","没有","把","被"}

def heuristic_scores_for_pos(word: str, pos: str) -> Dict[str, int]:
    """
    简单启发式：针对常见词类给出一个可用的兜底评分向量。
    """
    rules = RULE_SETS.get(pos, [])
    res = {}
    w = word.strip()
    is_short = len(w) <= 2
    for r in rules:
        name = r["name"]
        if pos == "名词":
            cond = (w in _NOUN_EXAMPLES) or (is_short and w not in _VERB_EXAMPLES and w not in _ADVERB_EXAMPLES and w not in _FUNCTION_WORD_EX)
            res[name] = r["match_score"] if cond else r["mismatch_score"]
        elif pos == "动词":
            cond = (w in _VERB_EXAMPLES) or (w.endswith("化") or w.endswith("做") or w.endswith("动"))
            res[name] = r["match_score"] if cond else r["mismatch_score"]
        elif pos == "副词":
            cond = (w in _ADVERB_EXAMPLES) or (w.endswith("地") or w.endswith("然"))
            res[name] = r["match_score"] if cond else r["mismatch_score"]
        elif pos in ("介词","助词","连词","语气词"):
            cond = w in _FUNCTION_WORD_EX or len(w) <= 2
            res[name] = r["match_score"] if cond else r["mismatch_score"]
        else:
            cond = is_short
            res[name] = r["match_score"] if cond else r["mismatch_score"]
    return res

# ===============================
# LLM 调用：请求模型返回 predicted_pos 与所有词类规则得分
# 期望 JSON：
# { "predicted_pos": "名词", "scores": {"名词": {"N1_...": 10, ...}, "动词": {...}}, "explanation": "..." }
# ===============================
def ask_model_for_pos_and_scores(word: str, max_retries: int = 1) -> Tuple[Dict[str, Dict[str,int]], str, str]:
    # 构建规则说明文本（简化版）
    rules_summary_lines = []
    for pos, rules in RULE_SETS.items():
        rules_summary_lines.append(f"{pos}:")
        for r in rules:
            rules_summary_lines.append(f"  - {r['name']}: {r['desc']} (match={r['match_score']}, mismatch={r['mismatch_score']})")
    rules_text = "\n".join(rules_summary_lines)

    system_msg = (
        "你是语言学研究专家。输入一个中文词语，检索定义好的所有规则，请你分析判断该词最可能的词类，同时判断该词的词汇结构，确定是向心结构还是离心结构，根据定义好的规则计算该词的隶属度分值，同时计算该词的向心度，要确保分析的学术严谨性，科学性，计算值的准确性："
        + ", ".join(list(RULE_SETS.keys()))
        + ")，并严格返回 JSON。"
        " JSON 格式必须为："
        '{"predicted_pos":"<词类名>", "scores": {"<词类名>": {"<规则名>": <值>, ...}, ...}, "explanation":"可选说明"}。'
        " 值应为整数（最好等于规则定义的 match_score 或 mismatch_score，或能映射到它们）。不要返回任何额外的文字，除了 JSON。"
    )
    user_prompt = f"词语：『{word}』\n请基于下列规则（每条规则后给出符合/不符合/不确定及相应的数值）来判定并评分：\n\n{rules_text}\n\n只返回严格的 JSON。"

    ok, resp = call_deepseek_chat([{"role":"system","content":system_msg},{"role":"user","content":user_prompt}], temperature=0.0)
    raw_text = ""
    parsed_json = None
    if ok:
        raw_text = extract_text_from_response(resp)
        parsed_json, parsed_text = extract_json_from_text(raw_text)
    if parsed_json and isinstance(parsed_json, dict):
        scores_out = {}
        predicted_pos = parsed_json.get("predicted_pos")
        if not predicted_pos:
            predicted_pos = parsed_json.get("pos") or parsed_json.get("pred_pos") or None

        for pos in RULE_SETS.keys():
            scores_out[pos] = {r["name"]: 0 for r in RULE_SETS[pos]}

        raw_scores = parsed_json.get("scores", {})
        for pos, rules in RULE_SETS.items():
            raw_for_pos = raw_scores.get(pos, {})
            if not isinstance(raw_for_pos, dict):
                continue
            for k, v in raw_for_pos.items():
                nk = normalize_key(k, rules)
                if nk:
                    rule_def = next(filter(lambda rr: rr["name"]==nk, rules))
                    mapped = map_to_allowed_score(rule_def, v)
                    scores_out[pos][nk] = mapped
        nonzero = sum(1 for pos in scores_out for val in scores_out[pos].values() if val != 0)
        if nonzero < 1:
            parsed_json = None
        else:
            if not predicted_pos:
                totals = {pos: sum(abs(v) for v in vals.values()) for pos, vals in scores_out.items()}
                predicted_pos = max(totals.items(), key=lambda x: x[1])[0]
            return scores_out, raw_text, predicted_pos

    # 兜底：启发式
    scores_out = {}
    for pos in RULE_SETS.keys():
        scores_out[pos] = heuristic_scores_for_pos(word, pos)
    totals = {pos: sum(v for v in vals.values()) for pos, vals in scores_out.items()}
    predicted_pos = max(totals.items(), key=lambda x: x[1])[0]
    return scores_out, "(heuristic fallback)", predicted_pos

# ===============================
# 工具：增补/更新规则集合（便于后续扩展）
# ===============================
def add_new_rule_set(pos_name: str, rules: list):
    RULE_SETS[pos_name] = rules
    MAX_SCORES[pos_name] = sum(abs(r["match_score"]) for r in rules)

def update_rule_set(pos_name: str, rules: list):
    RULE_SETS[pos_name] = rules
    MAX_SCORES[pos_name] = sum(abs(r["match_score"]) for r in rules)

# ===============================
# 可视化：Plotly 雷达图（标准化 0~1）
# ===============================
def plot_radar_chart_streamlit(scores_norm: Dict[str, float], title: str = "词类隶属度雷达图"):
    categories = list(scores_norm.keys())
    values = [float(scores_norm[c]) for c in categories]
    # 闭合环
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="隶属度",
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0]
            )
        ),
        showlegend=False,
        title=dict(text=title, x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 雷达图可视化（高亮中心字 + 显示向心度）
# ===============================
def plot_radar(df: pd.DataFrame, struct_info: dict, word: str):
    categories = list(df.columns[:-2]) if 'max_role' in df.columns else list(df.columns)
    fig = go.Figure()

    center_char = struct_info.get("中心字", "")
    centripetal_degree = struct_info.get("向心度", 0)
    structure_type = struct_info.get("结构", "未知")

    for idx, row in df.iterrows():
        values = [float(v) for v in row[categories]]
        values += [values[0]]
        theta = categories + [categories[0]]

        # 高亮中心字
        color = 'red' if idx == center_char else None
        line_width = 3 if idx == center_char else 1
        fill_mode = 'toself' if idx == center_char else 'none'

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill=fill_mode,
            name=f"{idx}（中心）" if idx == center_char else idx,
            line=dict(width=line_width, color=color)
        ))

    fig.update_layout(
        title=f"『{word}』字本位隶属度雷达图｜{structure_type}｜向心度={centripetal_degree}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


# ===============================
# Streamlit UI（简洁居中输入 + 结果 + 折叠详细）
# ===============================
st.set_page_config(page_title="汉语词类隶属度检测判类", layout="wide")
# 顶部居中标题
st.markdown("<h1 style='text-align: center;'>汉语词类隶属度检测判类</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>输入单个词 → 模型自动判类并返回各词类规则得分与隶属度（标准化 0~1）</p>", unsafe_allow_html=True)
st.write("")

# 居中输入框
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    word_input = st.text_input("", placeholder="在此输入要分析的词（例如：很 / 跑 / 美丽）")
    confirm = st.button("确认")

# -------------------------------
# 辅助函数（中心字与向心度）
# -------------------------------
def detect_head_word(word: str, pos_scores: Dict[str, float]) -> str:
    if len(word) == 1:
        return word
    head_pos = max(pos_scores, key=pos_scores.get)
    if head_pos in ["名词", "动词"]:
        return word[0]
    elif head_pos in ["形容词", "副词"]:
        return word[-1]
    else:
        return word[len(word)//2]

def determine_structure_type(head_pos: str, pos_scores: Dict[str, float]) -> str:
    total = sum(pos_scores.values()) or 1
    if head_pos in ["名词", "动词"] and pos_scores[head_pos] > 0.5 * total:
        return "向心结构"
    else:
        return "离心结构"

def compute_centripetal_degree(scores: Dict[str, float], head_pos: str) -> float:
    total = sum(scores.values()) or 1
    head_score = scores.get(head_pos, 0)
    return round((head_score / total) * 100, 2)

# -------------------------------
# 主逻辑
# -------------------------------
if confirm:
    word = (word_input or "").strip()
    if not word:
        st.warning("请输入一个词语后确认。")
    else:
        with st.spinner("模型打分判类中……"):
            scores_all, raw_out, predicted_pos = ask_model_for_pos_and_scores(word)

        # 计算各词类隶属度
        pos_totals = {}
        pos_normed = {}
        for pos, score_map in scores_all.items():
            total = sum(score_map.values())
            pos_totals[pos] = total
            max_possible = MAX_SCORES.get(pos, sum(abs(x) for x in score_map.values()) or 1)
            norm = round(max(0, total) / max_possible, 3) if max_possible != 0 else 0.0
            pos_normed[pos] = norm

        # -------------------------------
        # 新增部分：中心字与向心度计算
        # -------------------------------
        head_word = detect_head_word(word, pos_normed)
        head_pos = max(pos_normed, key=pos_normed.get)
        structure_type = determine_structure_type(head_pos, pos_normed)
        centripetal_degree = compute_centripetal_degree(pos_normed, head_pos)

        # 输出摘要
        st.markdown("---")
        st.subheader("判定摘要")
        st.markdown(f"- **输入词**： `{word}`")
        st.markdown(f"- **模型预测词类**： **{predicted_pos}**")
        st.markdown(f"- **中心字（Head Word）**： `{head_word}`")
        st.markdown(f"- **中心词类（Head POS）**： `{head_pos}`")
        st.markdown(f"- **结构类型**： {structure_type}")
        st.markdown(f"- **向心度（Centripetal Degree）**： {centripetal_degree} %")
        st.markdown(f"- **解析摘要（模型原始响应）**： `{raw_out}`")

        # 排名与表格
        ranked = sorted(pos_normed.items(), key=lambda x: x[1], reverse=True)
        st.subheader("隶属度排行（前10）")
        for i, (p, s) in enumerate(ranked[:10]):
            st.write(f"{i+1}. **{p}** — 隶属度：{s}")

        # 雷达图
        st.subheader("词类隶属度与向心度雷达图（标准化 0~1）")
        radar_scores = {p: pos_normed[p] for p, _ in ranked}
        plot_radar_chart_streamlit(radar_scores, title=f"“{word}” 的词类隶属度分布")
        st.markdown(f"**注：** 雷达图中各维度为词类隶属度，圆心外圈代表向心度（{centripetal_degree}%）")

        # 表格
        st.subheader("各词类隶属度（标准化 0~1）")
        df_norm = pd.DataFrame([{"词类": p, "隶属度": pos_normed[p]} for p in pos_normed]).set_index("词类")
        st.dataframe(df_norm, use_container_width=True)

        # 折叠详细规则判断
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

        # 原始响应
        with st.expander("查看原始模型文本 / 响应"):
            st.code(raw_out if raw_out else "(无)")


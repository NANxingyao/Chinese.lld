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
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 自定义CSS样式
hide_streamlit_style = """
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.dataframe {font-size: 12px;}
[data-testid="stSidebar"] {display: none !important;}
.stApp > div:first-child {padding-top: 2rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===============================
# 模型配置 (支持多模型，仅从环境变量获取API Key)
# ===============================
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "endpoint": "/services/aigc/text-generation/generation",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "input": {"messages": messages}, 
            "parameters": {"max_tokens": kw.get("max_tokens", 4096), "temperature": kw.get("temperature", 0.0)}
        },
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "endpoint": "/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload": lambda model, messages, **kw: {
            "model": model, "messages": messages, "max_tokens": kw.get("max_tokens", 4096), 
            "temperature": kw.get("temperature", 0.0), "stream": False,
        },
    },
}

# 模型选项（需提前设置对应环境变量）
MODEL_OPTIONS = {
    "DeepSeek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "env_var": "DEEPSEEK_API_KEY"
    },
    "通义千问": {
        "provider": "qwen",
        "model": "qwen-turbo",
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "env_var": "QWEN_API_KEY"
    },
    "豆包": {
        "provider": "doubao",
        "model": "doubao-pro-32k",
        "api_key": os.getenv("DOUBAO_API_KEY", ""),
        "env_var": "DOUBAO_API_KEY"
    },
}

# ===============================
# 词类规则与最大得分配置
# ===============================
RULE_SETS = {
    "名词": [
        {"name": "能受数量短语修饰", "match_score": 2, "mismatch_score": -2},
        {"name": "不能受副词修饰", "match_score": 2, "mismatch_score": -2},
        {"name": "能作主语/宾语", "match_score": 2, "mismatch_score": -1},
        {"name": "不能带宾语", "match_score": 1, "mismatch_score": -1},
    ],
    "动词": [
        {"name": "能受副词修饰", "match_score": 2, "mismatch_score": -2},
        {"name": "能带宾语", "match_score": 2, "mismatch_score": -1},
        {"name": "能作谓语中心", "match_score": 2, "mismatch_score": -2},
        {"name": "不能受数量短语修饰", "match_score": 1, "mismatch_score": -1},
    ],
    "形容词": [
        {"name": "能受程度副词修饰", "match_score": 2, "mismatch_score": -2},
        {"name": "不能带宾语", "match_score": 2, "mismatch_score": -2},
        {"name": "能作谓语/定语", "match_score": 2, "mismatch_score": -1},
        {"name": "不能受数量短语修饰", "match_score": 1, "mismatch_score": -1},
    ],
    "副词": [
        {"name": "只能作状语", "match_score": 3, "mismatch_score": -3},
        {"name": "不能作主语/宾语", "match_score": 2, "mismatch_score": -2},
        {"name": "不能受程度副词修饰", "match_score": 2, "mismatch_score": -1},
    ],
}

# 计算每个词类的最大可能得分
MAX_SCORES = {pos: sum(rule["match_score"] for rule in rules) for pos, rules in RULE_SETS.items()}

# ===============================
# 核心功能函数
# ===============================
def calculate_membership(scores_all: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """计算每个词类的隶属度（保留两位小数，得分为0时显示为0）"""
    membership = {}
    for pos, scores in scores_all.items():
        total_score = sum(scores.values())
        max_score = MAX_SCORES.get(pos, 1)
        
        if max_score == 0:
            membership[pos] = 0
        else:
            # 归一化到 [0, 1] 区间
            normalized = (total_score + max_score) / (2 * max_score)
            clamped = max(0.0, min(1.0, normalized))
            # 保留两位小数，得分为0时显示为0而非0.00
            rounded = round(clamped, 2)
            membership[pos] = 0 if rounded == 0 else rounded
    return membership

def extract_json_from_text(text: str) -> Tuple[Dict[str, Any], str]:
    """从文本中提取JSON内容"""
    # 匹配JSON对象
    json_pattern = r'\{[^\}]+\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            # 尝试解析最后一个匹配的JSON（通常是模型返回的结果）
            json_str = matches[-1]
            # 修复可能的格式问题（如单引号转双引号）
            json_str = json_str.replace("'", '"').replace('\n', '').replace('\t', '')
            parsed = json.loads(json_str)
            return parsed, json_str
        except:
            pass
    
    # 若未提取到JSON，返回空字典和原始文本
    return {}, text

def call_llm_api(provider: str, model: str, api_key: str, messages: List[Dict[str, str]]) -> Tuple[str, bool]:
    """调用LLM API获取结果（兼容多模型格式）"""
    try:
        config = MODEL_CONFIGS[provider]
        url = f"{config['base_url']}{config['endpoint']}"
        headers = config['headers'](api_key)
        payload = config['payload'](model, messages)
        
        # 发送请求
        response = requests.post(
            url, 
            headers=headers, 
            data=json.dumps(payload, ensure_ascii=False), 
            timeout=60
        )
        response.raise_for_status()
        response_json = response.json()
        
        # 解析不同模型的响应格式
        if provider == "deepseek":
            result = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        elif provider == "qwen":
            result = response_json.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        elif provider == "doubao":
            result = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            result = ""
        
        return result, True
    
    except Exception as e:
        error_msg = f"API调用错误: {str(e)}"
        # 追加响应内容（便于调试）
        if 'response' in locals() and response is not None:
            error_msg += f"\n响应内容: {response.text[:500]}"
        return error_msg, False

# ===============================
# 页面渲染逻辑
# ===============================
def main():
    st.title("汉语词类隶属度检测划类工具")
    
    # 输入区域（分两列：词语输入 + 模型选择）
    col1, col2 = st.columns([3, 1])
    with col1:
        word = st.text_input("请输入要分析的词语:", placeholder="例如：苹果、跑步、美丽、非常...")
    with col2:
        model_name = st.selectbox("选择模型:", list(MODEL_OPTIONS.keys()))
    
    # 关键：跟踪当前输入词，变化时重置状态
    if 'current_word' not in st.session_state:
        st.session_state.current_word = None
    
    # 输入词变化时，清除所有计算结果
    if word != st.session_state.current_word:
        st.session_state.current_word = word
        # 重置相关状态变量
        for key in ['scores_all', 'membership', 'api_response', 'raw_result']:
            if key in st.session_state:
                del st.session_state[key]
    
    # 显示环境变量配置提示（若未设置）
    model_info = MODEL_OPTIONS[model_name]
    if not model_info["api_key"]:
        st.warning(
            f"请先设置环境变量 `{model_info['env_var']}`\n"
            "Linux/Mac: export {model_info['env_var']}='你的API Key'\n"
            "Windows: set {model_info['env_var']}='你的API Key'"
        )
    
    # 分析按钮（仅当词语和API Key都有效时可点击）
    if st.button("开始分析", disabled=not (word.strip() and model_info["api_key"])):
        with st.spinner(f"正在使用{model_name}模型分析...请稍候"):
            # 构建提示词（明确要求返回JSON格式）
            prompt = f"""
            请分析词语"{word}"的词类属性，根据以下规则计算每个词类的得分：
            
            规则说明：
            - 每个词类包含多条规则，每条规则匹配得对应正分，不匹配得对应负分
            - 仅返回JSON格式结果，无需其他解释
            - JSON结构：{{"词类名": {{"规则1": 得分, "规则2": 得分, ...}}, ...}}
            
            词类规则：
            {json.dumps(RULE_SETS, ensure_ascii=False, indent=2)}
            
            示例输出：
            {{"名词": {{"能受数量短语修饰": 2, "不能受副词修饰": -2, ...}}, "动词": {...}}}
            """
            
            # 调用API
            messages = [{"role": "user", "content": prompt}]
            raw_result, success = call_llm_api(
                provider=model_info["provider"],
                model=model_info["model"],
                api_key=model_info["api_key"],
                messages=messages
            )
            
            # 保存原始结果（便于调试）
            st.session_state.raw_result = raw_result
            
            if not success:
                st.error(f"分析失败: {raw_result}")
                return
            
            # 提取并解析JSON结果
            parsed_scores, _ = extract_json_from_text(raw_result)
            
            # 验证解析结果（确保所有词类和规则都存在）
            scores_all = {}
            for pos in RULE_SETS:
                scores_all[pos] = {}
                # 初始化所有规则得分为0（未匹配时默认0）
                for rule in RULE_SETS[pos]:
                    scores_all[pos][rule["name"]] = 0
                
                # 更新模型返回的得分
                if pos in parsed_scores and isinstance(parsed_scores[pos], dict):
                    for rule_name, score in parsed_scores[pos].items():
                        # 匹配规则名称（忽略大小写和空格）
                        normalized_rule_name = rule_name.strip().lower()
                        for rule in RULE_SETS[pos]:
                            if rule["name"].strip().lower() == normalized_rule_name:
                                scores_all[pos][rule["name"]] = int(score) if score != 0 else 0
                                break
            
            # 计算隶属度并保存状态
            st.session_state.scores_all = scores_all
            st.session_state.membership = calculate_membership(scores_all)
            st.success("分析完成！")
    
    # 显示结果区域（仅当有计算结果时）
    if 'membership' in st.session_state:
        st.subheader("一、词类隶属度结果")
        
        # 1. 表格显示（按隶属度降序排列）
        membership_df = pd.DataFrame(
            list(st.session_state.membership.items()),
            columns=["词类", "隶属度"]
        ).sort_values(by="隶属度", ascending=False)
        st.dataframe(membership_df, use_container_width=True)
        
        # 2. 柱状图可视化
        fig = go.Figure(data=[go.Bar(
            x=membership_df["词类"],
            y=membership_df["隶属度"],
            text=membership_df["隶属度"],
            textposition='auto',
            marker_color='#1f77b4'
        )])
        fig.update_layout(
            title="词类隶属度分布",
            xaxis_title="词类",
            yaxis_title="隶属度",
            yaxis_range=[0, 1],
            width=800,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. 详细得分展示（展开面板）
        with st.expander("查看详细规则得分"):
            scores_df = pd.DataFrame()
            for pos, scores in st.session_state.scores_all.items():
                temp_df = pd.DataFrame(list(scores.items()), columns=["规则", pos])
                if scores_df.empty:
                    scores_df = temp_df
                else:
                    scores_df = pd.merge(scores_df, temp_df, on="规则", how="outer")
            st.dataframe(scores_df, use_container_width=True)
    
    # 显示原始API响应（便于调试）
    if 'raw_result' in st.session_state:
        with st.expander("查看API原始响应"):
            st.text_area("原始响应内容", st.session_state.raw_result, height=200)

if __name__ == "__main__":
    main()

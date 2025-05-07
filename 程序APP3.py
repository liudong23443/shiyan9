import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="Gastric Cancer Survival Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Times New Roman', serif;
        padding: 1.5rem 0;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-family: 'Times New Roman', serif;
    }
    .description {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .feature-section {
        padding: 1.5rem;
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-section {
        padding: 2rem;
        background-color: #F0F9FF;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 2rem;
        border: 1px solid #93C5FD;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #6B7280;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        border: none;
        margin-top: 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# 加载保存的随机森林模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf.pkl')
        # 添加模型信息
        if hasattr(model, 'n_features_in_'):
            st.session_state['model_n_features'] = model.n_features_in_
            st.session_state['model_feature_names'] = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model
    except Exception as e:
        st.error(f"⚠️ 模型文件 'rf.pkl' 加载错误: {str(e)}。请确保模型文件在正确的位置。")
        return None

model = load_model()

# 添加调试信息
if model is not None and hasattr(model, 'n_features_in_'):
    st.sidebar.write(f"模型期望特征数量: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        st.sidebar.write("模型期望特征:", expected_features)

# 特征范围定义
feature_ranges = {
    "Intraoperative Blood Loss": {"type": "numerical", "min": 0.000, "max": 800.000, "default": 50, 
                                 "description": "手术期间的出血量 (ml)", "unit": "ml"},
    "CEA": {"type": "numerical", "min": 0, "max": 150.000, "default": 8.68, 
           "description": "癌胚抗原水平", "unit": "ng/ml"},
    "Albumin": {"type": "numerical", "min": 1.0, "max": 80.0, "default": 38.60, 
               "description": "血清白蛋白水平", "unit": "g/L"},
    "TNM Stage": {"type": "categorical", "options": [1, 2, 3, 4], "default": 2, 
                 "description": "肿瘤分期", "unit": ""},
    "Age": {"type": "numerical", "min": 25, "max": 90, "default": 76, 
           "description": "患者年龄", "unit": "岁"},
    "Max Tumor Diameter": {"type": "numerical", "min": 0.2, "max": 20, "default": 4, 
                          "description": "肿瘤最大直径", "unit": "cm"},
    "Lymphovascular Invasion": {"type": "categorical", "options": [0, 1], "default": 1, 
                              "description": "淋巴血管侵犯 (0=否, 1=是)", "unit": ""},
}

# 特征顺序定义 - 确保与模型训练时的顺序一致
# 如果模型有feature_names_in_属性，使用它来定义特征顺序
if model is not None and hasattr(model, 'feature_names_in_'):
    feature_input_order = list(model.feature_names_in_)
    feature_ranges_ordered = {}
    for feature in feature_input_order:
        if feature in feature_ranges:
            feature_ranges_ordered[feature] = feature_ranges[feature]
        else:
            # 模型需要但UI中没有定义的特征
            st.sidebar.warning(f"模型要求特征 '{feature}' 但在UI中未定义")
    
    # 检查UI中定义但模型不需要的特征
    for feature in feature_ranges:
        if feature not in feature_input_order:
            st.sidebar.warning(f"UI中定义的特征 '{feature}' 不在模型要求的特征中")
    
    # 使用排序后的特征字典
    feature_ranges = feature_ranges_ordered
else:
    # 如果模型没有feature_names_in_属性，使用原来的顺序
    feature_input_order = list(feature_ranges.keys())

# 应用标题和描述
st.markdown('<h1 class="main-header">胃癌术后三年生存预测模型</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
    该模型基于术后患者临床特征，预测胃癌患者术后三年内死亡的概率。
    请输入患者的临床参数，系统将提供预测结果并展示影响预测的关键因素。
</div>
""", unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">患者特征输入</h2>', unsafe_allow_html=True)
    
    # 动态生成输入项
    feature_values = {}
    
    for feature in feature_input_order:
        properties = feature_ranges[feature]
        
        # 显示特征描述 - 根据变量类型生成不同的帮助文本
        if properties["type"] == "numerical":
            help_text = f"{properties['description']} ({properties['min']}-{properties['max']} {properties['unit']})"
            
            # 为数值型变量创建滑块
            value = st.slider(
                label=f"{feature}",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                step=0.1,
                help=help_text
            )
        elif properties["type"] == "categorical":
            # 对于分类变量，只使用描述作为帮助文本
            help_text = f"{properties['description']}"
            
            # 为分类变量创建单选按钮
            if feature == "TNM Stage":
                options_display = {1: "I期", 2: "II期", 3: "III期", 4: "IV期"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            elif feature == "Lymphovascular Invasion":
                options_display = {0: "否", 1: "是"}
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    format_func=lambda x: options_display[x],
                    help=help_text,
                    horizontal=True
                )
            else:
                value = st.radio(
                    label=f"{feature}",
                    options=properties["options"],
                    help=help_text,
                    horizontal=True
                )
                
        feature_values[feature] = value
    
    # 预测按钮
    predict_button = st.button("开始预测", help="点击生成预测结果")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if predict_button and model is not None:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">预测结果与解释</h2>', unsafe_allow_html=True)
        
        # 准备模型输入
        features_df = pd.DataFrame([feature_values])
        
        # 确保特征顺序与模型训练时一致
        if hasattr(model, 'feature_names_in_'):
            # 检查是否所有需要的特征都有值
            missing_features = [f for f in model.feature_names_in_ if f not in features_df.columns]
            if missing_features:
                st.error(f"缺少模型所需的特征: {missing_features}")
                st.stop()
            
            # 按模型训练时的特征顺序重排列特征
            features_df = features_df[model.feature_names_in_]
        
        # 转换为numpy数组
        features_array = features_df.values
        
        # 添加调试信息
        st.sidebar.write("输入特征形状:", features_array.shape)
        st.sidebar.write("输入特征列名:", list(features_df.columns))
        
        with st.spinner("计算预测结果..."):
            try:
                # 模型预测
                predicted_class = model.predict(features_array)[0]
                predicted_proba = model.predict_proba(features_array)[0]
                
                # 提取预测的类别概率
                death_probability = predicted_proba[1] * 100  # 假设1表示死亡类
                survival_probability = 100 - death_probability
                
                # 创建概率显示
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = death_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "三年内死亡风险", 'font': {'size': 24, 'family': 'SimHei'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': 'green'},
                            {'range': [30, 70], 'color': 'orange'},
                            {'range': [70, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': death_probability}}))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="white",
                    font={'family': "SimHei"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 创建风险类别显示
                risk_category = "低风险"
                risk_color = "green"
                if death_probability > 30 and death_probability <= 70:
                    risk_category = "中等风险"
                    risk_color = "orange"
                elif death_probability > 70:
                    risk_category = "高风险"
                    risk_color = "red"
                
                # 显示风险类别
                st.markdown(f"""
                <div style="text-align: center; margin-top: -1rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem; font-family: 'Times New Roman'; color: {risk_color}; font-weight: bold;">
                        {risk_category}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示具体概率数值
                risk_cols = st.columns(2)
                with risk_cols[0]:
                    st.metric(label="三年生存概率", value=f"{survival_probability:.1f}%")
                with risk_cols[1]:
                    st.metric(label="三年死亡风险", value=f"{death_probability:.1f}%")
                
                # 计算 SHAP 值
                try:
                    with st.spinner("生成特征影响解释..."):
                        # 使用TreeExplainer计算SHAP值
                        explainer = shap.TreeExplainer(model)
                        
                        # 直接获取当前输入数据的SHAP值
                        feature_names = list(features_df.columns)
                        X = features_df.iloc[0:1]  # 只使用一个样本
                        
                        # 获取SHAP值并检查形状
                        shap_values = explainer.shap_values(X)
                        
                        # 输出SHAP值的形状用于调试
                        if isinstance(shap_values, list):
                            st.write(f"SHAP值类型: 列表，长度: {len(shap_values)}")
                            for i, sv in enumerate(shap_values):
                                st.write(f"第{i}个元素形状: {sv.shape}")
                        else:
                            st.write(f"SHAP值形状: {shap_values.shape}")
                        
                        # 创建一个特征重要性表，不使用具体的SHAP图
                        st.subheader("特征重要性分析")
                        
                        # 针对形状为(1, 7, 2)的SHAP值，特殊处理
                        if not isinstance(shap_values, list) and len(shap_values.shape) == 3:
                            # 这是一个三维数组，选择正类(索引为1)的值
                            # 选择第一个样本，所有特征，第二个输出(通常是正类)
                            class_idx = 1  # 假设索引1是死亡风险类
                            feature_shap_values = shap_values[0, :, class_idx]
                        elif isinstance(shap_values, list):
                            # 这是一个列表，可能包含每个类别的SHAP值
                            if len(shap_values) > 1:
                                # 多分类情况，选择正类(通常是第二个元素)
                                class_idx = 1
                                feature_shap_values = shap_values[class_idx][0]
                            else:
                                # 只有一个类别(或回归问题)
                                feature_shap_values = shap_values[0][0]
                        else:
                            # 其他情况(通常是回归问题)
                            feature_shap_values = shap_values[0]
                        
                        # 创建特征重要性数据框
                        feature_importance = pd.DataFrame({
                            '特征': feature_names,
                            'SHAP值': feature_shap_values,
                            '绝对值': np.abs(feature_shap_values)
                        })
                        
                        # 按绝对值大小排序
                        feature_importance = feature_importance.sort_values('绝对值', ascending=False)
                        
                        # 添加影响方向
                        feature_importance['影响方向'] = np.where(feature_importance['SHAP值'] > 0, 
                                                           '增加死亡风险', '降低死亡风险')
                        
                        # 为影响方向添加颜色
                        feature_importance['颜色'] = np.where(feature_importance['SHAP值'] > 0, 'red', 'green')
                        
                        # 使用st.dataframe展示结果
                        st.dataframe(
                            feature_importance[['特征', '绝对值', '影响方向']],
                            column_config={
                                "特征": st.column_config.TextColumn("特征名称"),
                                "绝对值": st.column_config.NumberColumn("影响强度", format="%.3f"),
                                "影响方向": st.column_config.TextColumn(
                                    "对风险的影响",
                                    width="medium",
                                ),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # 使用matplotlib创建自定义条形图
                        top_features = feature_importance.head(min(7, len(feature_importance)))
                        
                        # 创建图表
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 创建条形图，颜色根据影响方向确定
                        bars = ax.barh(
                            top_features['特征'],
                            top_features['SHAP值'],
                            color=[('red' if x > 0 else 'green') for x in top_features['SHAP值']]
                        )
                        
                        # 设置标题和标签
                        ax.set_title('特征对生存预测的影响', fontsize=15, fontname='SimHei')
                        ax.set_xlabel('SHAP值 (正值=增加风险, 负值=降低风险)', fontsize=12, fontname='SimHei')
                        ax.set_ylabel('特征', fontsize=12, fontname='SimHei')
                        
                        # 添加网格线使图表更易读
                        ax.grid(axis='x', linestyle='--', alpha=0.6)
                        
                        # 添加零线
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                        
                        # 为条形添加数值标签
                        for bar in bars:
                            width = bar.get_width()
                            label_x_pos = width + 0.01 if width > 0 else width - 0.01
                            label_x_pos = width + (0.01 * np.sign(width))
                            ax.text(
                                label_x_pos, 
                                bar.get_y() + bar.get_height()/2,
                                f'{width:.3f}',
                                va='center',
                                ha='left' if width > 0 else 'right',
                                fontsize=10
                            )
                        
                        # 设置背景颜色
                        ax.set_facecolor('#f8f9fa')
                        fig.patch.set_facecolor('#f8f9fa')
                        
                        # 自定义图例
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='red', label='增加死亡风险'),
                            Patch(facecolor='green', label='降低死亡风险')
                        ]
                        ax.legend(handles=legend_elements, loc='lower right')
                        
                        # 确保图表整洁
                        plt.tight_layout()
                        
                        # 显示图表
                        st.pyplot(fig)
                        
                        # 添加解释性文字
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                        <p style="font-size: 0.9rem;">
                        <strong>SHAP值解释:</strong><br>
                        - <span style="color:red;">红色条</span>: 特征值增加患者的死亡风险<br>
                        - <span style="color:green;">绿色条</span>: 特征值降低患者的死亡风险<br>
                        - 条的长度: 表示特征对预测的影响强度<br>
                        - 此解释基于当前输入的特征值，不同的输入值可能会产生不同的影响
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 添加个性化建议部分
                        st.subheader("个性化风险干预建议")
                        
                        # 找出最影响增加风险的可干预特征
                        risk_increase_features = feature_importance[feature_importance['SHAP值'] > 0].copy()
                        
                        # 检查是否有可干预特征
                        if not risk_increase_features.empty:
                            # 选择第一个最重要的可干预特征
                            top_risk_feature = risk_increase_features.iloc[0]['特征']
                            
                            # 根据特征类型提供不同的建议
                            intervention_advice = {
                                "Age": "年龄是不可干预因素，但应注意其他风险因素的控制。",
                                "CEA": "建议定期监测CEA水平，针对性治疗以降低肿瘤负担。",
                                "Albumin": "建议改善营养状况，提高蛋白质摄入，必要时考虑蛋白质补充剂。",
                                "TNM Stage": "TNM分期反映肿瘤进展，建议遵循相应分期的标准治疗方案。",
                                "Max Tumor Diameter": "肿瘤直径是已知情况，建议关注术后随访和复查。",
                                "Intraoperative Blood Loss": "术中出血已发生，建议加强术后恢复，注意补充营养。",
                                "Lymphovascular Invasion": "淋巴血管侵犯状况已确定，建议遵循医生推荐的辅助治疗。"
                            }
                            
                            feature_value = features_df[top_risk_feature].values[0]
                            
                            st.markdown(f"""
                            <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 4px solid #d32f2f;">
                            <h4 style="color: #d32f2f; margin-top: 0;">主要风险因素: {top_risk_feature}</h4>
                            <p>当前值: {feature_value}</p>
                            <p><strong>建议:</strong> {intervention_advice.get(top_risk_feature, "请咨询医生获取针对性建议。")}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 找出最帮助降低风险的特征
                        risk_decrease_features = feature_importance[feature_importance['SHAP值'] < 0].copy()
                        
                        if not risk_decrease_features.empty:
                            top_protective_feature = risk_decrease_features.iloc[0]['特征']
                            feature_value = features_df[top_protective_feature].values[0]
                            
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 4px solid #388e3c;">
                            <h4 style="color: #388e3c; margin-top: 0;">主要保护因素: {top_protective_feature}</h4>
                            <p>当前值: {feature_value}</p>
                            <p><strong>建议:</strong> 继续保持此良好指标，可能有助于改善预后。</p>
                            </div>
                            """, unsafe_allow_html=True)
                                            
                except Exception as e:
                    st.error(f"生成SHAP解释时发生错误: {str(e)}")
                    # 报错时添加详细的调试信息
                    st.warning("正在尝试显示SHAP值的形状以帮助调试...")
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(features_df)
                        
                        st.write("SHAP值类型:", type(shap_values))
                        
                        if isinstance(shap_values, list):
                            st.info(f"SHAP值是列表，长度为: {len(shap_values)}")
                            for i, sv in enumerate(shap_values):
                                st.info(f"第{i}个元素形状: {sv.shape}")
                        else:
                            st.info(f"SHAP值形状: {shap_values.shape}")
                            
                        # 打印模型的一些信息
                        st.write("模型类型:", type(model))
                        st.write("输入数据形状:", features_df.shape)
                        st.write("输入数据特征:", list(features_df.columns))
                        
                        # 尝试用另一种方式生成解释
                        st.warning("尝试使用特征重要性作为替代解释...")
                        
                        # 使用随机森林的内置特征重要性
                        if hasattr(model, 'feature_importances_'):
                            feature_importances = pd.DataFrame({
                                '特征': features_df.columns,
                                '重要性': model.feature_importances_
                            }).sort_values('重要性', ascending=False)
                            
                            st.dataframe(feature_importances)
                        
                    except Exception as debug_e:
                        st.error(f"获取调试信息时也出错: {str(debug_e)}")
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}")
                st.warning("请检查输入数据是否与模型期望的特征匹配，或联系开发人员获取支持。")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 添加临床建议部分
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">临床建议</h2>', unsafe_allow_html=True)
        
        # 根据不同风险级别提供建议
        if death_probability <= 30:
            st.markdown("""
            <div style="color: green;">
                <p>⭐ <strong>低风险患者建议:</strong></p>
                <ul>
                    <li>遵循标准的术后随访计划</li>
                    <li>每3-6个月进行一次常规检查</li>
                    <li>保持健康的生活方式和饮食习惯</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif death_probability <= 70:
            st.markdown("""
            <div style="color: orange;">
                <p>⚠️ <strong>中等风险患者建议:</strong></p>
                <ul>
                    <li>更频繁的随访计划，建议每2-3个月一次</li>
                    <li>考虑辅助治疗方案</li>
                    <li>密切监测CEA等肿瘤标志物的变化</li>
                    <li>注意营养支持和生活质量管理</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color: red;">
                <p>🔴 <strong>高风险患者建议:</strong></p>
                <ul>
                    <li>更积极的随访计划，建议每1-2个月一次</li>
                    <li>考虑更强化的辅助治疗方案</li>
                    <li>密切监测可能的复发和转移迹象</li>
                    <li>增强营养支持和症状管理</li>
                    <li>考虑多学科团队会诊</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 显示应用说明和使用指南
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">模型说明</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            本预测模型基于随机森林算法构建，通过分析胃癌患者的关键临床特征，预测术后三年内的死亡风险。
        </p>
        
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            <strong>模型特征包括:</strong>
        </p>
        <ul style="font-family: 'Times New Roman'; font-size: 1.1rem;">
            <li><strong>年龄</strong>: 患者年龄是影响胃癌预后的重要因素</li>
            <li><strong>TNM分期</strong>: 描述肿瘤大小、淋巴结侵犯和远处转移情况</li>
            <li><strong>肿瘤直径</strong>: 肿瘤的最大直径</li>
            <li><strong>血清白蛋白</strong>: 反映患者的营养状况</li>
            <li><strong>CEA</strong>: 癌胚抗原，是一种常用的肿瘤标志物</li>
            <li><strong>淋巴血管侵犯</strong>: 指肿瘤是否侵入淋巴或血管</li>
            <li><strong>术中出血量</strong>: 反映手术复杂性和患者耐受性</li>
        </ul>
        
        <p style="font-family: 'Times New Roman'; font-size: 1.1rem; margin-top: 1rem;">
            <strong>使用指南:</strong> 在左侧填写患者的临床参数，然后点击"开始预测"按钮获取结果。系统将生成死亡风险预测以及每个特征的影响程度分析。
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示示例案例
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">典型案例分析</h2>', unsafe_allow_html=True)
        
        # 创建示例数据
        case_data = {
            "案例": ["低风险案例", "中风险案例", "高风险案例"],
            "年龄": [55, 68, 76],
            "TNM分期": ["II期", "III期", "IV期"],
            "肿瘤直径(cm)": [2.5, 4.0, 8.5],
            "CEA": [3.2, 7.5, 25.8],
            "预测生存率": ["92%", "58%", "23%"]
        }
        
        case_df = pd.DataFrame(case_data)
        
        # 使用Streamlit的表格显示
        st.dataframe(
            case_df,
            column_config={
                "案例": st.column_config.TextColumn("案例类型"),
                "年龄": st.column_config.NumberColumn("年龄", format="%d岁"),
                "TNM分期": st.column_config.TextColumn("TNM分期"),
                "肿瘤直径(cm)": st.column_config.NumberColumn("肿瘤直径", format="%.1fcm"),
                "CEA": st.column_config.NumberColumn("CEA", format="%.1fng/ml"),
                "预测生存率": st.column_config.TextColumn("3年生存率", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )
                
        st.markdown('</div>', unsafe_allow_html=True)

# 添加页脚说明
st.markdown("""
<div class="disclaimer">
    <p>📋 免责声明：本预测工具仅供临床医生参考，不能替代专业医疗判断。预测结果应结合患者的完整临床情况进行综合评估。</p>
    <p>© 2023 胃癌术后预测研究团队 | 开发版本 v1.2.0</p>
</div>
""", unsafe_allow_html=True) 
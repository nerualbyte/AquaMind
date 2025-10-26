# aquamind_app.py
import streamlit as st
import pandas as pd
import numpy as np
from aquamind_core import AquaMind
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AquaMind Demo", layout="wide", page_icon="💧")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #0066cc; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #666; margin-top: 0;}
    .risk-high {background: #ff4444; color: white; padding: 10px; border-radius: 5px; font-weight: bold;}
    .risk-medium {background: #ffaa00; color: white; padding: 10px; border-radius: 5px; font-weight: bold;}
    .risk-low {background: #00cc66; color: white; padding: 10px; border-radius: 5px; font-weight: bold;}
    .metric-card {background: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .reasoning-step {background: #e8f4f8; padding: 10px; margin: 5px 0; border-left: 4px solid #0066cc; border-radius: 3px; color: #000000;}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "cooling_data.csv"

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.scenario_active = None

# Initialize core
@st.cache_resource
def init_aquamind():
    return AquaMind()

am = init_aquamind()

# Generate sample data if needed
if not os.path.exists(DATA_FILE):
    with st.spinner("Generating sample cooling system data..."):
        am.generate_sample_data(DATA_FILE, hours=12)

# Load dataframe
def load_data():
    return am.load_data(DATA_FILE)

df = load_data()

# Initialize session state with OPTIMAL LOW-RISK defaults
if not st.session_state.initialized:
    # Set to LOW RISK defaults from the start
    st.session_state.ambient = 38.0  # Normal ambient temp
    st.session_state.it_load = 75     # Moderate IT load
    st.session_state.flow_rate = 500  # Healthy flow rate
    st.session_state.tds = 850        # Good water quality
    st.session_state.ph = 7.3         # Optimal pH
    st.session_state.pump_eff = 90    # Good pump efficiency
    st.session_state.initialized = True

# Header
st.markdown('<div class="main-header">💧 AquaMind</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Cooling-Water Intelligence for Data Centers | Powered by K2 Think</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar controls
st.sidebar.header("🎛️ Live System Controls")

st.session_state.ambient = st.sidebar.slider("Ambient Temperature (°C)", 25.0, 55.0, st.session_state.ambient, step=0.5)
st.session_state.it_load = st.sidebar.slider("IT Load (%)", 0, 100, st.session_state.it_load)
st.session_state.flow_rate = st.sidebar.slider("Flow Rate (L/min)", 350, 700, st.session_state.flow_rate)
st.session_state.tds = st.sidebar.slider("TDS (ppm)", 200, 2000, st.session_state.tds)
st.session_state.ph = st.sidebar.slider("pH Level", 6.0, 9.0, st.session_state.ph, step=0.1)
st.session_state.pump_eff = st.sidebar.slider("Pump Efficiency (%)", 60, 100, st.session_state.pump_eff)

st.sidebar.markdown("---")
st.sidebar.header("🎬 Demo Scenarios")
st.sidebar.markdown("*Click to simulate realistic conditions*")

col_s1, col_s2 = st.sidebar.columns(2)

scenario_clicked = False

with col_s1:
    if st.button("🌡️ Heatwave", use_container_width=True):
        st.session_state.ambient = 48.0
        st.session_state.it_load = 90
        st.session_state.flow_rate = 420
        st.session_state.scenario_active = "Heatwave"
        scenario_clicked = True

    if st.button("🔧 Pump Strain", use_container_width=True):
        st.session_state.pump_eff = 78
        st.session_state.flow_rate = 380
        st.session_state.scenario_active = "Pump Strain"
        scenario_clicked = True

with col_s2:
    if st.button("💧 Poor Water Quality", use_container_width=True):
        st.session_state.tds = 1350
        st.session_state.ph = 6.3
        st.session_state.scenario_active = "Poor Water Quality"
        scenario_clicked = True

    if st.button("🔄 Reset to Normal", use_container_width=True):
        st.session_state.ambient = 38.0
        st.session_state.it_load = 75
        st.session_state.flow_rate = 500
        st.session_state.tds = 850
        st.session_state.ph = 7.3
        st.session_state.pump_eff = 90
        st.session_state.scenario_active = None
        scenario_clicked = True

# Force immediate rerun when scenario button clicked
if scenario_clicked:
    st.rerun()

if st.session_state.scenario_active:
    st.sidebar.success(f"✓ Scenario: {st.session_state.scenario_active}")

# Apply changes to dataframe
df_live = df.copy()
df_live.loc[df_live.index[-1], ['ambient_temp_c','it_load_pct','flow_rate_lpm','tds_ppm','ph_level','pump_efficiency_pct']] = \
    [st.session_state.ambient, st.session_state.it_load, st.session_state.flow_rate, 
     st.session_state.tds, st.session_state.ph, st.session_state.pump_eff]

# Recalculate derived columns
df_live['temp_delta'] = df_live['return_temp_c'] - df_live['supply_temp_c']
df_live['water_per_kw_l'] = df_live['water_consumed_l'] / (df_live['it_load_pct'] * 10)

# Run K2 Think reasoning with spinner
with st.spinner("🧠 K2 Think analyzing system state..."):
    time.sleep(1.5)  # Simulate API latency
    result = am.analyze_df(df_live)

# Main layout - REMOVED ML INSIGHTS TAB
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 K2 Think Reasoning", "💧 Water Footprint"])

# ========== TAB 1: DASHBOARD ==========
with tab1:
    # Risk Level Banner
    risk = result['risk_level']
    if risk == "HIGH":
        st.markdown(f'<div class="risk-high">🚨 RISK LEVEL: {risk}</div>', unsafe_allow_html=True)
    elif risk == "MEDIUM":
        st.markdown(f'<div class="risk-medium">⚠️ RISK LEVEL: {risk}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low">✅ RISK LEVEL: {risk}</div>', unsafe_allow_html=True)
    
    st.markdown(f"**System Summary:** {result['system_summary']}")
    st.markdown(f"**Prediction:** {result['predicted_risk']}")
    
    st.markdown("---")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📉 Critical Metrics (Last 48 readings)")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Flow Rate (L/min)", "TDS (ppm)", "Pump Efficiency (%)"),
            vertical_spacing=0.12
        )
        
        recent = df_live.tail(48)
        
        fig.add_trace(go.Scatter(x=recent.index, y=recent['flow_rate_lpm'], 
                                name="Flow Rate", line=dict(color='#0066cc', width=2)),
                     row=1, col=1)
        fig.add_hline(y=450, line_dash="dash", line_color="red", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=recent.index, y=recent['tds_ppm'],
                                name="TDS", line=dict(color='#ff6600', width=2)),
                     row=2, col=1)
        fig.add_hline(y=1000, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_hline(y=1200, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=recent.index, y=recent['pump_efficiency_pct'],
                                name="Pump Eff", line=dict(color='#00cc66', width=2)),
                     row=3, col=1)
        fig.add_hline(y=85, line_dash="dash", line_color="red", row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("🌡️ Environmental Conditions")
        
        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Ambient Temperature (°C)", "IT Load (%)"),
            vertical_spacing=0.15
        )
        
        fig2.add_trace(go.Scatter(x=recent.index, y=recent['ambient_temp_c'],
                                 fill='tozeroy', line=dict(color='#ff4444', width=2)),
                      row=1, col=1)
        
        fig2.add_trace(go.Scatter(x=recent.index, y=recent['it_load_pct'],
                                 fill='tozeroy', line=dict(color='#6600cc', width=2)),
                      row=2, col=1)
        
        fig2.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Actions
    st.subheader("🎯 Recommended Actions")
    for i, action in enumerate(result['recommended_actions'], 1):
        with st.expander(f"**Action {i}: {action['action']}**", expanded=(i==1)):
            st.markdown(f"**🎯 Goal:** {action['goal']}")
            st.markdown(f"**📊 Impact:** {action['impact']}")
            st.markdown(f"**✓ Confidence:** {action['confidence']}")
            if 'water_saved' in action:
                st.markdown(f"**💧 Water Saved:** {action['water_saved']}")
            if 'controls' in action:
                st.info(f"**{action['controls']}**")

# ========== TAB 2: K2 THINK REASONING ==========
with tab2:
    st.subheader("🧠 K2 Think Reasoning Chain")
    st.markdown("*Watch the AI think through the problem step-by-step*")
    
    for i, step in enumerate(result['reasoning_chain'], 1):
        st.markdown(f'<div class="reasoning-step"><strong>Step {i}:</strong> {step}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 Raw Analysis Output")
    with st.expander("View JSON Response"):
        st.json({
            "risk_level": result['risk_level'],
            "predicted_risk": result['predicted_risk'],
            "forecast_data": result.get('forecast_data', {}),
            "actions_count": len(result['recommended_actions'])
        })

# ========== TAB 3: WATER FOOTPRINT ==========
with tab3:
    wf = result['water_footprint']
    
    st.subheader("💧 Water Usage Intelligence")
    
    col_wf1, col_wf2, col_wf3, col_wf4 = st.columns(4)
    
    with col_wf1:
        st.metric("Water per kW", f"{wf['water_per_kw']:.2f} L", 
                 delta=f"{wf['efficiency_vs_baseline_pct']:.1f}% vs baseline",
                 delta_color="inverse")
    
    with col_wf2:
        st.metric("Per AI Task", f"{wf['water_per_ai_task_liters']:.2f} L")
    
    with col_wf3:
        st.metric("Sustainability Score", wf['sustainability_score'],
                 help="A = Excellent, B = Good, C = Needs Improvement")
    
    with col_wf4:
        st.metric("4h Consumption", f"{wf['total_consumed_4h_liters']:,.0f} L")
    
    st.markdown("---")
    
    # Water savings visualization
    st.subheader("💰 AquaMind Impact")
    
    col_save1, col_save2 = st.columns(2)
    
    with col_save1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Daily Water Savings")
        st.markdown(f"# {wf['estimated_daily_savings_liters']:,.0f} L")
        st.markdown("*With AquaMind optimizations*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_save2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Monthly Water Savings")
        st.markdown(f"# {wf['estimated_monthly_savings_liters']:,.0f} L")
        st.markdown(f"*≈ {wf['estimated_monthly_savings_liters'] // 1000:,.0f} m³ saved per month*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison chart
    st.markdown("---")
    st.subheader("📊 Baseline vs AquaMind")
    
    baseline_daily = wf['total_consumed_4h_liters'] * 6
    aquamind_daily = baseline_daily - wf['estimated_daily_savings_liters']
    
    comparison_data = {
        'System': ['Without AquaMind', 'With AquaMind'],
        'Daily Water (L)': [baseline_daily, aquamind_daily],
        'Color': ['#ff6666', '#00cc66']
    }
    
    fig_compare = go.Figure(data=[
        go.Bar(x=comparison_data['System'], 
               y=comparison_data['Daily Water (L)'],
               marker_color=comparison_data['Color'],
               text=[f"{int(v):,} L" for v in comparison_data['Daily Water (L)']],
               textposition='auto')
    ])
    
    fig_compare.update_layout(
        title="Daily Water Consumption Comparison",
        yaxis_title="Liters",
        height=400
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**AquaMind** | Powered by K2 Think | Built for Hackathon 2025")
st.markdown("*Protecting UAE's water resources through AI reasoning*")

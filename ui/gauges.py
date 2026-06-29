# ui/gauges.py

import streamlit as st

def render_pagasa_gauge(advisory, disable_flashing, flash_cb):
    if advisory:
        pagasa_severity = advisory["severity"]
        ts = advisory["timestamp"]
        label = advisory["label"]
        color = advisory["color"]
    else:
        pagasa_severity = None
        ts = "Advisory unavailable"
        label = "No data"
        color = "grey"

    flash_class = ""
    if pagasa_severity is not None and not disable_flashing:
        if pagasa_severity >= 0.60:
            flash_class = "fast-flash"
        else:
            flash_class = "slow-flash"
    warning_icon = "⚠️" if pagasa_severity and pagasa_severity >= 0.60 else ""

    gauge_html = f"""
    <div class="live-gauge {flash_class}" style="margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color: {color};">{warning_icon} {label}</span>
            <span style="font-size:0.8rem; color: #888;">Last updated: {ts}</span>
        </div>
        <div style="width:100%; height:25px; background:linear-gradient(to right, green 0%, green 25%, #FFC107 25%, #FFC107 50%, orange 50%, orange 75%, red 75%, red 100%); border-radius:5px; position:relative; margin-top:5px; border: 1px solid #555;">
            <div style="position:absolute; left:{pagasa_severity*100 if pagasa_severity else 0}%; top:-4px; width:4px; height:33px; background:black; border-radius:2px;"></div>
        </div>
        <p style="margin-top:5px; font-size:11px; color: #aaa;">Based on PAGASA advisory (updated daily). Simulation slider below.</p>
    </div>
    <style>
        @keyframes slowPulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        @keyframes fastPulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
            100% {{ opacity: 1; }}
        }}
        .slow-flash {{
            animation: slowPulse 2s infinite;
        }}
        .fast-flash {{
            animation: fastPulse 0.8s infinite;
        }}
    </style>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
    return pagasa_severity, label

def render_sim_gauge(flood_sev):
    if flood_sev <= 0.25:
        rainfall_mm = flood_sev * 40
        label = "Light rain"
    elif flood_sev <= 0.50:
        rainfall_mm = 10 + (flood_sev - 0.25) * 80
        label = "Moderate rain (Yellow)"
    elif flood_sev <= 0.75:
        rainfall_mm = 30 + (flood_sev - 0.50) * 120
        label = "Heavy rain (Orange)"
    else:
        rainfall_mm = 60 + (flood_sev - 0.75) * 160
        label = "Torrential rain (Red)"

    st.markdown(f"🌧️ **{rainfall_mm:.0f} mm** – {label}")
    gauge_html = f"""
    <div style="width:100%; height:30px; background:linear-gradient(to right, green 0%, green 25%, #FFC107 25%, #FFC107 50%, orange 50%, orange 75%, red 75%, red 100%); border-radius:5px; position:relative; margin-bottom:10px;">
        <div style="position:absolute; left:{flood_sev*100}%; top:-5px; width:4px; height:40px; background:black; border-radius:2px;"></div>
    </div>
    <p style="margin-top:5px; font-size:12px;">🟢 0-10 mm &nbsp; 🟡 10-30 mm &nbsp; 🟠 30-60 mm &nbsp; 🔴 >60 mm</p>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
    return label

def render_waterlevel_gauge():
    wl_gauge = """
    <div style="filter: grayscale(100%); opacity: 0.5; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color: grey;">🔒 No Data</span>
            <span style="font-size:0.8rem; color: #888;">Awaiting data feed</span>
        </div>
        <div style="width:100%; height:25px; background:linear-gradient(to right, green 0%, green 37%, #FFC107 37%, #FFC107 62%, orange 62%, orange 87%, red 87%, red 100%); border-radius:5px; position:relative; margin-top:5px; border: 1px solid #555;">
            <div style="position:absolute; left:0%; top:-4px; width:4px; height:33px; background:black; border-radius:2px;"></div>
        </div>
        <p style="margin-top:5px; font-size:11px; color: #aaa;">Water level data not yet available – awaiting PAGASA PREDICT / MDRRMO</p>
    </div>
    """
    st.markdown(wl_gauge, unsafe_allow_html=True)

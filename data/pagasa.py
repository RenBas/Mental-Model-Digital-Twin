# data/pagasa.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

@st.cache_data(ttl=3600)
def fetch_pagasa_advisory():
    url = "https://www.pagasa.dost.gov.ph/flood/tagoloan"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", class_="entry-content")
        if not content_div:
            content_div = soup.find("body")
        text = content_div.get_text(separator="\n") if content_div else soup.get_text()

        issued_match = re.search(r"ISSUED AT (.*?)\n", text)
        timestamp_str = issued_match.group(1).strip() if issued_match else "Unknown time"

        obs_match = re.search(r"OBSERVED 24-HR RAINFALL:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
        forecast_match = re.search(r"FORECAST 24-HR RAINFALL:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
        observed_text = obs_match.group(1).strip() if obs_match else ""
        forecast_text = forecast_match.group(1).strip() if forecast_match else ""

        rainfall_desc = forecast_text if forecast_text else observed_text
        if not rainfall_desc:
            return None

        severity = None
        label = "Unknown"
        color = "grey"
        if re.search(r"\b(torrential|intense|extreme)\b", rainfall_desc, re.IGNORECASE):
            severity = 0.90
            color = "red"
            label = "Torrential Rain (Red)"
        elif re.search(r"\bheavy\b", rainfall_desc, re.IGNORECASE):
            severity = 0.65
            color = "orange"
            label = "Heavy Rain (Orange)"
        elif re.search(r"\bmoderate\b", rainfall_desc, re.IGNORECASE):
            severity = 0.35
            color = "#FFC107"
            label = "Moderate Rain (Yellow)"
        elif re.search(r"\blight\b", rainfall_desc, re.IGNORECASE):
            severity = 0.10
            color = "green"
            label = "Light Rain (Green)"
        else:
            return None

        return {
            "severity": severity,
            "color": color,
            "label": label,
            "timestamp": timestamp_str,
            "raw_text": rainfall_desc
        }
    except Exception:
        return None

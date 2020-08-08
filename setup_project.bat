@echo off
title Recommendations to Lawmakers for Fighting Obesity in the USA
python -m venv venv && .\venv\Scripts\activate && pip install --disable-pip-version-check -q -r requirements.txt  && python policy_data_mining.py

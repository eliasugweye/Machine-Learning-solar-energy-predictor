#!/usr/bin/env python3
"""SolarML — run with: python run.py  →  http://localhost:5000"""
from app import app
if __name__ == "__main__":
    print("\n  ☀️  SolarML Solar Energy Prediction")
    print("  → http://localhost:5000")
    print("  Demo login: demo / demo\n")
    app.run(debug=False, port=5000, host="0.0.0.0")

Machine learning Solar Energy Prediction System
A fintech-style machine learning web application that predicts ATM dispense errors in real time.

 Prerequisites
Ensure you have Python 3.9+ installed.

 Installation
Install dependencies:

bash
pip install -r requirements.txt
Database setup:
The app uses SQLite. On first run, it will automatically create database.db.

Local Deployment (Auto-Launch)
Run:

bash
python app.py
Flask server starts automatically

Browser opens at http://127.0.0.1:5000

You’ll see the login page first

 Production Deployment (Render / Railway)
Push your project to GitHub

On Render or Railway, create a new Web Service

Set Start Command (from Procfile):

Code
web: python app.py
Configure environment variables:

PORT → provided by platform

SECRET_KEY → set securely

Flask runs with:

python
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
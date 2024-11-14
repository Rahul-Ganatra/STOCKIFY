# Stockify - Stock Price Prediction Web Application

## Overview
Stockify is a web-based application that uses machine learning to predict future stock prices for Indian stocks. The application provides real-time market indices tracking and detailed price predictions with interactive visualizations.

## Features
- Real-time tracking of major Indian market indices (SENSEX, NIFTY 50, NIFTY BANK)
- Stock price prediction for up to 30 days
- Interactive graphs showing:
  - Last 30 days historical data with predictions
  - Full historical data with predictions
- Detailed prediction table with dates and prices
- Responsive design for all device sizes

## Technology Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras (LSTM model)
- **Data Source**: Yahoo Finance API (yfinance)
- **Visualization**: Matplotlib
- **Additional Libraries**: 
  - pandas
  - numpy
  - scikit-learn

## Quick Start

**1. Clone and set up the project:**

git clone https://github.com/Rahul-Ganatra/STOCKIFY.git
cd stockify

**2. Create a Virtual Environment:**
   
python -m venv venv

**3. Activate the Virtual Environment:**
   
_For Windows:_
venv\Scripts\activate

_For MacOS/Linux:_
source venv/bin/activate

**4. Download Required Packages:**
   
pip install Flask yfinance pandas numpy matplotlib tensorflow

## Running the Application

**1. Start the Flask Server:**

python app.py

**2. Open the Web App:**
   
http://127.0.0.1:5000/
   

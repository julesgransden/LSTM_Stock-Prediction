# Stock Price Prediction with LSTM and Streamlit

## Overview
This project implements a Stock Price Prediction model using Long Short-Term Memory (LSTM) neural networks and presents the results through an interactive front-end created with Streamlit. The LSTM architecture is well-suited for capturing temporal dependencies in time series data, making it an effective choice for predicting stock prices.

## Features
- **LSTM Model:** Utilizes a deep learning model based on LSTM layers to analyze historical stock data and make future price predictions.
- **Streamlit Front-End:** Implements an interactive and user-friendly front-end using Streamlit for visualizing predictions and exploring historical data.

## Libraries Used
- **NumPy:** A library for numerical operations in Python.
- **Pandas:** A data manipulation library that provides data structures for efficiently storing large datasets.
- **Matplotlib:** A data visualization library for creating static, animated, and interactive visualizations.
- **Pandas DataReader:** A data retrieval library for extracting financial and economic data from various sources.
- **yfinance:** A library for downloading historical market data from Yahoo Finance.
- **Scikit-learn (MinMaxScaler):** A machine learning library providing tools for data preprocessing and scaling.
- **Keras:** An open-source deep learning library for building neural network models.
- **Streamlit:** A Python library for creating web applications with minimal effort.

## Project Structure
- `stock_prediction.py`: Python script containing the main code for data processing, LSTM model training, and Streamlit app creation.
- `keras_model.h5`: File holding the trained LSTM model.
- `requirements.txt`: Lists the necessary dependencies for running the project.

## Usage
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Run the Streamlit app:** `streamlit run stock_prediction.py`
3. **Access the interactive front-end in your web browser.**

## Instructions
1. Open `stock_prediction.py` and adjust parameters such as `start`, `end`, and the stock ticker in the Streamlit app section.
2. Ensure that the `keras_model.h5` file is present and contains your trained LSTM model.
3. Run the Streamlit app and interact with the front-end to visualize predictions and explore stock data.

Feel free to customize and enhance the project based on your specific needs. Happy forecasting!

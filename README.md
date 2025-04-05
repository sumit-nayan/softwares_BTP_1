# Time Series Analysis Application

A Streamlit-based interactive application for time series analysis and forecasting.

## Installation

1. Install required packages:

```bash
pip install streamlit pandas numpy matplotlib plotly statsmodels prophet tensorflow scikit-learn
```

2. Save the code to `time_series_app.py`

## Usage

Run the application:

```bash
streamlit run time_series_app.py
```

## Workflow

1. Upload your time series data (CSV or Excel)
2. Select the date and value columns
3. Explore different analysis pages

## Analysis Pages

- **Visualization**: View different plots of your data
- **Decomposition**: Break down the series into trend, seasonal, and residual components
- **Statistical Tests**: Check stationarity and autocorrelation
- **Forecasting**: Apply ARIMA, Prophet, or LSTM models
- **Model Comparison**: Compare different models' performance

## Features

### Data Handling
- Supports CSV and Excel files
- Column selection interface

### Visualization
- Interactive line plots
- Scatter plots
- Histograms
- Rolling statistics visualization

### Time Series Analysis
- Seasonal decomposition (additive/multiplicative models)
- ADF test for stationarity
- ACF/PACF plots

### Forecasting Models
- **ARIMA** with configurable parameters
- **Facebook's Prophet** with seasonality options
- **LSTM** neural network with look-back configuration

### Model Evaluation
- Compare RMSE of different models on test set
- Visual model performance comparison

## Interface
- Sidebar navigation
- Interactive visual feedback at each step
- Responsive design

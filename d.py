import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import io
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Time Series Analysis Tool", layout="wide")

# Title
st.title("Interactive Time Series Analysis Software")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Visualization", 
                                 "Decomposition", "Statistical Tests", 
                                 "Forecasting", "Model Comparison"])

# Initialize session state for data storage
if 'df' not in st.session_state:
    st.session_state.df = None
if 'date_col' not in st.session_state:
    st.session_state.date_col = None
if 'value_col' not in st.session_state:
    st.session_state.value_col = None

# Data Upload Page
if page == "Data Upload":
    st.header("Upload Your Time Series Data")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            st.success("Data uploaded successfully!")
            st.write("Preview of your data:")
            st.dataframe(df.head())
            
            # Let user select date and value columns
            cols = df.columns.tolist()
            date_col = st.selectbox("Select the date/time column", cols)
            value_col = st.selectbox("Select the value column", cols)
            
            if st.button("Set Columns"):
                st.session_state.date_col = date_col
                st.session_state.value_col = value_col
                
                # Convert date column to datetime
                try:
                    st.session_state.df[date_col] = pd.to_datetime(st.session_state.df[date_col])
                    st.session_state.df = st.session_state.df.sort_values(date_col)
                    st.success(f"Time series set with {date_col} as date and {value_col} as value")
                except Exception as e:
                    st.error(f"Error converting date column: {str(e)}")
    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Visualization Page
elif page == "Visualization" and st.session_state.df is not None:
    st.header("Time Series Visualization")
    
    df = st.session_state.df
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    
    # Plotting options
    plot_type = st.selectbox("Select plot type", 
                            ["Line Plot", "Scatter Plot", "Histogram", "Rolling Statistics"])
    
    if plot_type == "Line Plot":
        fig = px.line(df, x=date_col, y=value_col, title=f"Time Series of {value_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Scatter Plot":
        fig = px.scatter(df, x=date_col, y=value_col, title=f"Scatter Plot of {value_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=value_col, title=f"Distribution of {value_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Rolling Statistics":
        window = st.slider("Select window size", 1, 365, 30)
        
        df_rolling = df.set_index(date_col)[value_col].rolling(window=window)
        rolling_mean = df_rolling.mean()
        rolling_std = df_rolling.std()
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], label='Original')
        plt.plot(df[date_col], rolling_mean, label=f'Rolling Mean ({window} days)')
        plt.plot(df[date_col], rolling_std, label=f'Rolling Std ({window} days)')
        plt.legend()
        plt.title(f"Rolling Statistics for {value_col}")
        st.pyplot(fig)

# Decomposition Page
elif page == "Decomposition" and st.session_state.df is not None:
    st.header("Time Series Decomposition")
    
    df = st.session_state.df
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    
    # Set date as index
    ts = df.set_index(date_col)[value_col]
    
    # Decomposition options
    model = st.selectbox("Select decomposition model", ["additive", "multiplicative"])
    period = st.number_input("Seasonal period (if known)", min_value=1, value=7)
    
    if st.button("Perform Decomposition"):
        try:
            decomposition = seasonal_decompose(ts, model=model, period=period)
            
            st.subheader("Decomposition Results")
            
            # Plot decomposition
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
            decomposition.observed.plot(ax=ax1)
            ax1.set_ylabel('Observed')
            decomposition.trend.plot(ax=ax2)
            ax2.set_ylabel('Trend')
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_ylabel('Seasonal')
            decomposition.resid.plot(ax=ax4)
            ax4.set_ylabel('Residual')
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in decomposition: {str(e)}")

# Statistical Tests Page
elif page == "Statistical Tests" and st.session_state.df is not None:
    st.header("Statistical Tests for Time Series")
    
    df = st.session_state.df
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    
    ts = df.set_index(date_col)[value_col]
    
    test = st.selectbox("Select statistical test", 
                       ["Augmented Dickey-Fuller Test", "Autocorrelation", "Partial Autocorrelation"])
    
    if test == "Augmented Dickey-Fuller Test":
        if st.button("Run ADF Test"):
            result = adfuller(ts)
            st.write("ADF Statistic:", result[0])
            st.write("p-value:", result[1])
            st.write("Critical Values:")
            for key, value in result[4].items():
                st.write(f"   {key}: {value}")
            
            if result[1] <= 0.05:
                st.success("The series is stationary (p-value <= 0.05)")
            else:
                st.warning("The series is not stationary (p-value > 0.05)")
    
    elif test == "Autocorrelation":
        lags = st.slider("Number of lags", 1, 100, 40)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(ts, lags=lags, ax=ax)
        st.pyplot(fig)
    
    elif test == "Partial Autocorrelation":
        lags = st.slider("Number of lags", 1, 100, 40)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_pacf(ts, lags=lags, ax=ax)
        st.pyplot(fig)

# Forecasting Page
elif page == "Forecasting" and st.session_state.df is not None:
    st.header("Time Series Forecasting")
    
    df = st.session_state.df
    date_col = st.session_state.date_col
    value_col = st.session_state.value_col
    
    model_choice = st.selectbox("Select forecasting model", 
                              ["ARIMA", "Prophet", "LSTM Neural Network"])
    
    forecast_periods = st.number_input("Number of periods to forecast", min_value=1, value=30)
    
    if model_choice == "ARIMA":
        st.subheader("ARIMA Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR order (p)", min_value=0, value=1)
        with col2:
            d = st.number_input("Difference order (d)", min_value=0, value=1)
        with col3:
            q = st.number_input("MA order (q)", min_value=0, value=1)
        
        if st.button("Run ARIMA Model"):
            try:
                ts = df.set_index(date_col)[value_col]
                
                # Fit ARIMA model
                model = ARIMA(ts, order=(p, d, q))
                model_fit = model.fit()
                
                # Forecast
                forecast = model_fit.get_forecast(steps=forecast_periods)
                forecast_index = pd.date_range(ts.index[-1], periods=forecast_periods+1, closed='right')
                forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
                conf_int = forecast.conf_int()
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ts.plot(ax=ax, label='Observed')
                forecast_series.plot(ax=ax, label='Forecast', color='red')
                ax.fill_between(conf_int.index, 
                              conf_int.iloc[:, 0], 
                              conf_int.iloc[:, 1], 
                              color='pink', alpha=0.3)
                ax.legend()
                ax.set_title(f"ARIMA({p},{d},{q}) Forecast")
                st.pyplot(fig)
                
                # Show model summary
                st.subheader("Model Summary")
                st.text(str(model_fit.summary()))
                
            except Exception as e:
                st.error(f"Error in ARIMA model: {str(e)}")
    
    elif model_choice == "Prophet":
        st.subheader("Prophet Model Configuration")
        
        # Prophet options
        growth = st.selectbox("Growth model", ["linear", "logistic"])
        seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"])
        yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
        weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
        daily_seasonality = st.checkbox("Daily seasonality", value=False)
        
        if st.button("Run Prophet Model"):
            try:
                # Prepare data for Prophet
                prophet_df = df[[date_col, value_col]].copy()
                prophet_df.columns = ['ds', 'y']
                
                # Initialize and fit model
                model = Prophet(growth=growth,
                                seasonality_mode=seasonality_mode,
                                yearly_seasonality=yearly_seasonality,
                                weekly_seasonality=weekly_seasonality,
                                daily_seasonality=daily_seasonality)
                model.fit(prophet_df)
                
                # Make future dataframe
                future = model.make_future_dataframe(periods=forecast_periods)
                
                # Forecast
                forecast = model.predict(future)
                
                # Plot forecast
                fig1 = model.plot(forecast)
                st.pyplot(fig1)
                
                # Plot components
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"Error in Prophet model: {str(e)}")
    
    elif model_choice == "LSTM Neural Network":
        st.subheader("LSTM Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            look_back = st.number_input("Look back periods", min_value=1, value=30)
        with col2:
            epochs = st.number_input("Training epochs", min_value=1, value=50)
        
        if st.button("Train LSTM Model"):
            try:
                ts = df.set_index(date_col)[value_col].values
                ts = ts.reshape(-1, 1)
                
                # Normalize data
                scaler = MinMaxScaler(feature_range=(0, 1))
                ts_scaled = scaler.fit_transform(ts)
                
                # Create dataset for LSTM
                def create_dataset(dataset, look_back=1):
                    X, Y = [], []
                    for i in range(len(dataset)-look_back-1):
                        a = dataset[i:(i+look_back), 0]
                        X.append(a)
                        Y.append(dataset[i + look_back, 0])
                    return np.array(X), np.array(Y)
                
                X, y = create_dataset(ts_scaled, look_back)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Split train/test
                train_size = int(len(X) * 0.67)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Build LSTM model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                
                # Train model
                history = model.fit(X_train, y_train, 
                                  validation_data=(X_test, y_test), 
                                  epochs=epochs, 
                                  batch_size=32, 
                                  verbose=0)
                
                # Plot training history
                fig1, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history.history['loss'], label='train')
                ax.plot(history.history['val_loss'], label='test')
                ax.set_title('Model Loss')
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epoch')
                ax.legend()
                st.pyplot(fig1)
                
                # Make predictions
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)
                
                # Inverse transform
                train_predict = scaler.inverse_transform(train_predict)
                y_train = scaler.inverse_transform([y_train])
                test_predict = scaler.inverse_transform(test_predict)
                y_test = scaler.inverse_transform([y_test])
                
                # Calculate RMSE
                train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
                test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
                
                st.write(f"Train RMSE: {train_rmse:.2f}")
                st.write(f"Test RMSE: {test_rmse:.2f}")
                
                # Plot predictions
                train_predict_plot = np.empty_like(ts)
                train_predict_plot[:, :] = np.nan
                train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict
                
                test_predict_plot = np.empty_like(ts)
                test_predict_plot[:, :] = np.nan
                test_predict_plot[len(train_predict)+(look_back*2)+1:len(ts)-1, :] = test_predict
                
                fig2, ax = plt.subplots(figsize=(12, 6))
                ax.plot(ts, label='Original')
                ax.plot(train_predict_plot, label='Train Prediction')
                ax.plot(test_predict_plot, label='Test Prediction')
                ax.legend()
                ax.set_title("LSTM Predictions")
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"Error in LSTM model: {str(e)}")

# Model Comparison Page
elif page == "Model Comparison" and st.session_state.df is not None:
    st.header("Model Comparison")
    
    if st.button("Run Model Comparison"):
        try:
            df = st.session_state.df
            date_col = st.session_state.date_col
            value_col = st.session_state.value_col
            
            ts = df.set_index(date_col)[value_col]
            
            # Split data into train and test
            train_size = int(len(ts) * 0.8)
            train, test = ts[:train_size], ts[train_size:]
            
            # Initialize results dictionary
            results = {}
            
            # ARIMA Model
            try:
                arima_model = ARIMA(train, order=(1,1,1))
                arima_fit = arima_model.fit()
                arima_forecast = arima_fit.forecast(steps=len(test))
                arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
                results['ARIMA(1,1,1)'] = arima_rmse
            except Exception as e:
                st.warning(f"ARIMA failed: {str(e)}")
                results['ARIMA(1,1,1)'] = None
            
            # Prophet Model
            try:
                prophet_df = pd.DataFrame({
                    'ds': train.index,
                    'y': train.values
                })
                prophet_model = Prophet()
                prophet_model.fit(prophet_df)
                
                future = prophet_model.make_future_dataframe(periods=len(test))
                prophet_forecast = prophet_model.predict(future)
                prophet_forecast = prophet_forecast.set_index('ds')['yhat'][-len(test):]
                
                prophet_rmse = np.sqrt(mean_squared_error(test, prophet_forecast))
                results['Prophet'] = prophet_rmse
            except Exception as e:
                st.warning(f"Prophet failed: {str(e)}")
                results['Prophet'] = None
            
            # Simple Moving Average
            try:
                window = 7  # 7-day moving average
                history = train.copy()
                predictions = []
                
                for t in range(len(test)):
                    yhat = history[-window:].mean()
                    predictions.append(yhat)
                    history = pd.concat([history, pd.Series([test[t]], index=[test.index[t]])])
                
                sma_rmse = np.sqrt(mean_squared_error(test, predictions))
                results['SMA (7)'] = sma_rmse
            except Exception as e:
                st.warning(f"SMA failed: {str(e)}")
                results['SMA (7)'] = None
            
            # Display results
            st.subheader("Model Comparison Results (RMSE)")
            results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE'])
            st.dataframe(results_df.sort_values('RMSE'))
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            ts.plot(ax=ax, label='Actual')
            
            if 'ARIMA(1,1,1)' in results and results['ARIMA(1,1,1)'] is not None:
                arima_forecast.plot(ax=ax, label='ARIMA Forecast')
            
            if 'Prophet' in results and results['Prophet'] is not None:
                prophet_forecast.plot(ax=ax, label='Prophet Forecast')
            
            if 'SMA (7)' in results and results['SMA (7)'] is not None:
                pd.Series(predictions, index=test.index).plot(ax=ax, label='SMA Forecast')
            
            ax.legend()
            ax.set_title("Model Forecast Comparison")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in model comparison: {str(e)}")

# Display message if no data uploaded
elif st.session_state.df is None:
    st.warning("Please upload your data first from the Data Upload page.")
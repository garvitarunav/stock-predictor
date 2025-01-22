import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np
import streamlit as st

# Function to fetch historical data for training
def fetch_historical_data(stock_symbol):
    stock_data = yf.Ticker(stock_symbol)
    df = stock_data.history(period="1y", interval="1d")
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)  # Remove timezone info
    return df

# Function to add new technical features (e.g., moving averages)
def add_technical_indicators(df):
    df['5_day_MA'] = df['Close'].rolling(window=5).mean()  # 5-day moving average
    df['10_day_MA'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
    df['20_day_MA'] = df['Close'].rolling(window=20).mean()  # 20-day moving average
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().gt(0).rolling(window=14).sum() /
                                   df['Close'].diff().lt(0).rolling(window=14).sum())))
    df = df.dropna()  # Drop rows with NaN values due to moving averages or RSI
    return df

# Prepare data for training
def prepare_data(df, target_feature):
    features = df[['Open', 'High', 'Low', 'Close', 'Volume', '5_day_MA', '10_day_MA', '20_day_MA', 'RSI']]
    target = df[target_feature].shift(-1).dropna()  # Predict the next day's feature
    features = features[:-1]  # Remove the last row to match target size
    return features, target

# Function to create the pipeline with configurable parameters
def create_pipeline(n_estimators=100, max_depth=None):
    preprocessor = ColumnTransformer(
        transformers=[('num', MinMaxScaler(), ['Open', 'High', 'Low', 'Close', 'Volume', '5_day_MA', '10_day_MA', '20_day_MA', 'RSI'])]
    )
    # Create the pipeline with imputer and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
                                ('model', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42))])
    return pipeline

# Function to perform grid search for hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    pipeline = create_pipeline()
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Fetch live data for prediction
def fetch_live_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    live_data = stock.history(period="1d")
    live_data_dict = {
        'Open': live_data['Open'][0],
        'High': live_data['High'][0],
        'Low': live_data['Low'][0],
        'Close': live_data['Close'][0],
        'Volume': live_data['Volume'][0],
        '5_day_MA': live_data['Close'].rolling(window=5).mean().iloc[-1],
        '10_day_MA': live_data['Close'].rolling(window=10).mean().iloc[-1],
        '20_day_MA': live_data['Close'].rolling(window=20).mean().iloc[-1],
        'RSI': 100 - (100 / (1 + (live_data['Close'].diff().gt(0).rolling(window=14).sum() / 
                                  live_data['Close'].diff().lt(0).rolling(window=14).sum())))
    }
    return live_data_dict

# Main function for Streamlit
def main():
    st.title("Stock Price Predictor")
    
    stock_symbol = st.text_input("Enter the stock symbol (e.g., 'ADANIPOWER.NS'):")
    
    if stock_symbol:
        df = fetch_historical_data(stock_symbol)
        df = add_technical_indicators(df)
        
        target_feature = st.selectbox("Which feature would you like to predict?", ['Open', 'Close', 'Volume', 'High', 'Low'])
        
        features, target = prepare_data(df, target_feature)
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        choice = st.selectbox("Would you like to:", ["Fetch live data", "Manually input custom data"])
        
        if choice == "Fetch live data":
            live_data = fetch_live_data(stock_symbol)
            st.write("Live Data Fetched:", live_data)
            live_df = pd.DataFrame([live_data])
            
            with st.spinner("Training the model... Please wait."):
                pipeline = tune_hyperparameters(X_train, y_train)
                joblib.dump(pipeline, 'stock_predictor_model.pkl')
            
            prediction = pipeline.predict(live_df)
            st.write(f"Predicted {target_feature} for the next day based on live data: {prediction[0]}")
        
        elif choice == "Manually input custom data":
            with st.form("manual_input_form"):
                st.write("Enter all values:")
                open_price = st.number_input("Enter the 'Open' price:", value=0.0)
                high_price = st.number_input("Enter the 'High' price:", value=0.0)
                low_price = st.number_input("Enter the 'Low' price:", value=0.0)
                close_price = st.number_input("Enter the 'Close' price:", value=0.0)
                volume = st.number_input("Enter the 'Volume' for the day:", value=0)
                submitted = st.form_submit_button("Submit")
            
            if submitted:
                manual_data = {
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume,
                    '5_day_MA': (open_price + high_price + low_price + close_price) / 4,  # Example of custom MA calculation
                    '10_day_MA': (open_price + high_price + low_price + close_price) / 4,  # Example of custom MA calculation
                    '20_day_MA': (open_price + high_price + low_price + close_price) / 4,  # Example of custom MA calculation
                    'RSI': 50  # Placeholder for RSI
                }
                
                manual_df = pd.DataFrame([manual_data])
                
                with st.spinner("Training the model... Please wait."):
                    pipeline = tune_hyperparameters(X_train, y_train)
                    joblib.dump(pipeline, 'stock_predictor_model.pkl')
                
                prediction = pipeline.predict(manual_df)
                st.write(f"Predicted {target_feature} for the next day based on the provided manual data: {prediction[0]}")

if __name__ == "__main__":
    main()

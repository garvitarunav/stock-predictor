import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pyttsx3
import threading

# Predefined stock list for NSE and BSE
def get_stock_list():
    data = {
        "Stock Symbol": [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ADANIPOWER.NS", 
            "SBIN.BO", "ITC.NS", "ICICIBANK.NS", "WIPRO.NS", "HCLTECH.NS",
            "ONGC.NS", "TATASTEEL.NS", "BHARTIARTL.NS", "MARUTI.NS", "AXISBANK.NS",
            "LT.NS", "BAJAJFINSV.NS", "HINDUNILVR.NS", "ULTRACEMCO.NS", "KOTAKBANK.NS",
            "ADANIENT.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS", "GAIL.NS",
            "COALINDIA.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
            "GRASIM.NS", "NESTLEIND.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS",
            "M&M.NS", "TITAN.NS", "SHREECEM.NS", "TECHM.NS", "HAVELLS.NS",
            "BERGEPAINT.NS", "PIDILITIND.NS", "BRITANNIA.NS", "DLF.NS", "GODREJCP.NS",
            "JSWSTEEL.NS", "HINDALCO.NS", "APOLLOHOSP.NS", "ICICIPRULI.NS", "SBICARD.NS",
            "AUROPHARMA.NS", "BIOCON.NS", "ABBOTINDIA.NS", "GLAXO.NS", "LUPIN.NS",
            "TORNTPHARM.NS", "ZYDUSLIFE.NS", "MCDOWELL-N.NS", "UBL.NS", "INDIGO.NS",
            "IRCTC.NS", "BANDHANBNK.NS", "YESBANK.NS", "IDFCFIRSTB.NS", "PNB.NS",
            "CANBK.NS", "FEDERALBNK.NS", "BANKBARODA.NS", "IOB.NS", "INDUSINDBK.NS",
            "IEX.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "TATACONSUM.NS", "BOSCHLTD.NS",
            "SIEMENS.NS", "HDFCLIFE.NS", "BEL.NS", "CUMMINSIND.NS", "ASIANPAINT.NS",
            "MARICO.NS", "TATAMOTORS.NS", "ASHOKLEY.NS", "TVSMOTOR.NS", "AMBUJACEM.NS",
            "ACC.NS", "RAMCOCEM.NS", "IDBI.NS", "NAM-INDIA.NS", "MFSL.NS",
            "ADANIPORTS.NS", "INDHOTEL.NS", "TRENT.NS", "VBL.NS", "CHOLAFIN.NS",
            "SRF.NS", "DABUR.NS", "ESCORTS.NS", "PAGEIND.NS", "PVRINOX.NS"
        ],
        "Stock Name": [
            "Reliance Industries", "Tata Consultancy Services", "Infosys", "HDFC Bank", "Adani Power", 
            "State Bank of India", "ITC Limited", "ICICI Bank", "Wipro", "HCL Technologies",
            "ONGC", "Tata Steel", "Bharti Airtel", "Maruti Suzuki", "Axis Bank",
            "Larsen & Toubro", "Bajaj Finserv", "Hindustan Unilever", "UltraTech Cement", "Kotak Mahindra Bank",
            "Adani Enterprises", "Power Grid Corporation", "NTPC", "Bharat Petroleum", "GAIL",
            "Coal India", "Sun Pharmaceutical", "Dr. Reddy's", "Cipla", "Divi's Laboratories",
            "Grasim Industries", "Nestle India", "Hero MotoCorp", "Eicher Motors", "Bajaj Auto",
            "Mahindra & Mahindra", "Titan Company", "Shree Cement", "Tech Mahindra", "Havells India",
            "Berger Paints", "Pidilite Industries", "Britannia Industries", "DLF Limited", "Godrej Consumer Products",
            "JSW Steel", "Hindalco Industries", "Apollo Hospitals", "ICICI Prudential Life Insurance", "SBI Cards",
            "Aurobindo Pharma", "Biocon", "Abbott India", "GlaxoSmithKline Pharma", "Lupin",
            "Torrent Pharma", "Zydus Lifesciences", "United Spirits", "United Breweries", "IndiGo",
            "IRCTC", "Bandhan Bank", "Yes Bank", "IDFC First Bank", "Punjab National Bank",
            "Canara Bank", "Federal Bank", "Bank of Baroda", "Indian Overseas Bank", "IndusInd Bank",
            "Indian Energy Exchange", "Adani Green Energy", "Adani Transmission", "Tata Consumer Products", "Bosch Limited",
            "Siemens", "HDFC Life Insurance", "Bharat Electronics", "Cummins India", "Asian Paints",
            "Marico", "Tata Motors", "Ashok Leyland", "TVS Motor Company", "Ambuja Cements",
            "ACC", "Ramco Cements", "IDBI Bank", "Nippon Life India AMC", "Max Financial Services",
            "Adani Ports", "Indian Hotels", "Trent", "Varun Beverages", "Cholamandalam Finance",
            "SRF Limited", "Dabur", "Escorts Kubota", "Page Industries", "PVR Inox"
        ]
    }
    return pd.DataFrame(data)

# Fetch historical data
def fetch_historical_data(stock_symbol):
    stock_data = yf.Ticker(stock_symbol)
    df = stock_data.history(period="1y", interval="1d")
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df

# Add technical indicators
def add_technical_indicators(df):
    df['5_day_MA'] = df['Close'].rolling(window=5).mean()
    df['10_day_MA'] = df['Close'].rolling(window=10).mean()
    df['20_day_MA'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().gt(0).rolling(window=14).sum() / 
                                   df['Close'].diff().lt(0).rolling(window=14).sum())))

    df = df.dropna()
    return df

# Prepare data
def prepare_data(df, target_feature):
    features = df[['Open', 'High', 'Low', 'Close', '5_day_MA', '10_day_MA', '20_day_MA', 'RSI']]
    target = df[target_feature].shift(-1).dropna()
    features = features[:-1]
    return features, target

# Create pipeline
def create_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[('num', MinMaxScaler(), ['Open', 'High', 'Low', 'Close', '5_day_MA', '10_day_MA', '20_day_MA', 'RSI'])])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('imputer', SimpleImputer(strategy='mean')),
                                ('model', RandomForestRegressor(random_state=42))])
    return pipeline

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'model__n_estimators': [50, 75],
        'model__max_depth': [10, 20, None]
    }
    pipeline = create_pipeline()
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def speak_text(text):
    try:
        import os
        if "STREAMLIT" not in os.environ:  # Check if the app is running locally
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
            engine.say(text)
            engine.runAndWait()
        else:
            st.warning("Text-to-speech is not supported in the deployed environment.")
    except Exception as e:
        st.error(f"An error occurred with text-to-speech: {str(e)}")

# Modify fetch_stock_info function to shorten, speak and print the information
def fetch_stock_info(stock_name):
    search_query = f"{stock_name} site:en.wikipedia.org"
    response = requests.get(f"https://en.wikipedia.org/wiki/{stock_name.replace(' ', '_')}")
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")

    stock_info = ""
    word_count = 0

    if paragraphs:
        # Get the first 4 bullet points, limiting to one line each and ensuring total word count is under 100
        for para in paragraphs:
            text = para.get_text().strip()
            words_in_para = text.split()
            if word_count + len(words_in_para) <= 100:
                stock_info += f"• {text}\n"
                word_count += len(words_in_para)
            else:
                break
    
    # Return stock info with a maximum of 100 words
    return stock_info if stock_info else "No information available."

# Main app
def main():
    # Changing the Title 
    st.markdown("""
    <span style='font-family:BODONI POSTER; font-size:48px; font-weight:bold;'>Stock Price Predictor</span>
    """, unsafe_allow_html=True)


    st.sidebar.markdown("""
    <span style='font-family:Dom Casual; font-size:28px; font-weight:bold;'>Stock Symbols</span>
    """, unsafe_allow_html=True)

    
    stock_list = get_stock_list()
    st.sidebar.dataframe(stock_list, height=400, width=300)
    
    selected_stock = st.sidebar.selectbox("Select a stock:", [""] + stock_list["Stock Symbol"].tolist())
    
    live_data = None  # Initialize live_data variable
    target_feature = None
    
    if selected_stock:
        stock_name = stock_list.loc[stock_list['Stock Symbol'] == selected_stock, 'Stock Name'].values[0]
        st.markdown(f"""
        <span style='font-family:Georgia; font-size:20px; font-weight:bold;'>Selected Stock: </span>
        <span style='font-family:Arial; font-size:18px;'>{stock_name}</span>
        """, unsafe_allow_html=True)

        
        # Store stock info in session state if it's not already there
        if 'stock_info' not in st.session_state or st.session_state.selected_stock != selected_stock:
            # Fetch stock info and store it
            stock_info = fetch_stock_info(stock_name)
            st.session_state.stock_info = stock_info
            st.session_state.selected_stock = selected_stock  # Store the selected stock to track changes
            
            # Print the stock info first
            stock_info_placeholder = st.empty()  # Placeholder for stock info
            stock_info_placeholder.markdown(f"""
            <span style='font-family:Georgia; font-size:18px; font-weight:bold;'>Stock Information:</span>
            <span style='font-family:Arial; font-size:16px;'>{stock_info}</span>
            """, unsafe_allow_html=True)
            # Print info on the screen
            
            # Start a separate thread to speak the stock info (so that it does not block the app)
            threading.Thread(target=speak_text, args=(stock_info,)).start()
        
        else:
            # If the stock is already selected, just display the stored info
            st.write("Stock Information:")
            st.write(st.session_state.stock_info)
        
        # Immediately show prediction options after stock info is being spoken
        target_feature = st.selectbox("Which feature would you like to predict?", ['', 'Open', 'Close', 'High', 'Low'])

        if target_feature:
            choice = st.radio("Choose Prediction Mode:", ["", "Fetch live data", "Manually input custom data"])
            
            if choice == "Fetch live data":
                df = fetch_historical_data(selected_stock)
                df = add_technical_indicators(df)
                live_data = df.tail(1)  # Fetch the latest data row
                st.write("Live Data Fetched:")
                st.dataframe(live_data)
                
                # Train the model and show prediction immediately
                with st.spinner("Training the model..."):
                    features, target = prepare_data(df, target_feature)
                    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                    pipeline = tune_hyperparameters(X_train, y_train)
                    # Calculate the R-squared score
                    r_squared_score = pipeline.score(X_test, y_test)

                    # Improved UI with a card-like display
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f9f9f9; 
                            padding: 15px; 
                            border-radius: 10px; 
                            border: 1px solid #e0e0e0; 
                            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        ">
                            <h4 style="color:rgb(92, 74, 226); margin-bottom: 10px;">Model Performance</h4>
                            <p style="font-size: 18px; color: #333; margin: 0;">
                                <b>R-squared Score on Test Data:</b> {r_squared_score:.4f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    
                    if live_data is not None:
                        prediction = pipeline.predict(live_data)
                        # Display the predicted value with enhanced UI
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #f9f9f9; 
                                padding: 15px; 
                                border-radius: 10px; 
                                border: 1px solid #e0e0e0; 
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                            ">
                                <h4 style="color:rgb(226, 74, 74); margin-bottom: 10px;">Prediction Result</h4>
                                <p style="font-size: 18px; color: #333; margin: 0;">
                                    <b>Predicted {target_feature} for the next day:</b> {prediction[0]:.4f}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
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
                    # Prepare manual data
                    manual_data = {
                        'Open': open_price,
                        'High': high_price,
                        'Low': low_price,
                        'Close': close_price,
                        'Volume': volume,
                        '5_day_MA': (open_price + high_price + low_price + close_price) / 4,
                        '10_day_MA': (open_price + high_price + low_price + close_price) / 4,
                        '20_day_MA': (open_price + high_price + low_price + close_price) / 4,
                        'RSI': 50  # Placeholder RSI
                    }
                    
                    live_data = pd.DataFrame([manual_data])

                    # Re-train the model with manual data
                    with st.spinner("Training the model..."):
                        df = fetch_historical_data(selected_stock)
                        df = add_technical_indicators(df)
                        df = df.append(live_data, ignore_index=True)  # Add manual data to historical data
                        
                        features, target = prepare_data(df, target_feature)
                        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                        pipeline = tune_hyperparameters(X_train, y_train)
                        
                        # Calculate the R-squared score for the new model
                        r_squared_score = pipeline.score(X_test, y_test)

                        st.markdown(
                            f"""
                            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                                <h4 style="color:rgb(84, 74, 226); margin-bottom: 10px;">Model Performance</h4>
                                <p style="font-size: 18px; color: #333; margin: 0;">
                                    <b>R-squared Score on Test Data:</b> {r_squared_score:.4f}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if live_data is not None:
                            prediction = pipeline.predict(live_data)
                            st.markdown(
                                f"""
                                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                                    <h4 style="color:rgb(226, 74, 74); margin-bottom: 10px;">Prediction Result</h4>
                                    <p style="font-size: 18px; color: #333; margin: 0;">
                                        <b>Predicted {target_feature} for the next day:</b> {prediction[0]:.4f}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )



if __name__ == "__main__":
    main()
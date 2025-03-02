import streamlit as st
import requests
import time
from bs4 import BeautifulSoup
from transformers import pipeline
from collections import Counter
import pandas as pd

def stock_sentiment_analysis():
    # Load Hugging Face Sentiment Analysis Model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def fetch_news_gnews(stock_name, max_articles=50, retries=3, wait_time=5):
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": stock_name,
            "lang": "en",
            "country": "in",
            "max": max_articles,
            "apikey": "08eb58904eb6b537bc9e4c9c6d5ddd42"
        }
        
        for attempt in range(retries):
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("articles", [])
            time.sleep(wait_time)
        return []

    def fetch_news_google(stock_name):
        google_url = f"https://news.google.com/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            response = requests.get(google_url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")
            news_list = []
            for item in soup.find_all("article"):
                title_element = item.find("h3")
                if title_element:
                    title = title_element.text.strip()
                    link = "https://news.google.com" + title_element.a["href"][1:]
                    news_list.append({"title": title, "url": link})
            return news_list[:20]
        except Exception as e:
            return []

    def get_stock_news(stock_name):
        return fetch_news_gnews(stock_name) + fetch_news_google(stock_name)

    def analyze_news_sentiment(news_articles):
        sentiment_counts = Counter()
        if not news_articles:
            return [], Counter(), "Neutral"
        
        headlines = [article["title"] for article in news_articles]
        sentiment_results = sentiment_pipeline(headlines)
        sentiment_data = []
        
        for article, sentiment in zip(news_articles, sentiment_results):
            sentiment_label = "Positive" if sentiment["label"] == "POSITIVE" else "Negative"
            sentiment_counts[sentiment_label] += 1
            sentiment_data.append({
                "Title": article["title"], 
                "Sentiment": sentiment_label, 
                "URL": f"[Click Here]({article['url']})"
            })
        
        total_articles = sum(sentiment_counts.values())
        most_common_sentiment = sentiment_counts.most_common(1)[0][0] if total_articles > 0 else "Neutral"
        
        return sentiment_data, sentiment_counts, most_common_sentiment

    # Apply custom CSS for larger text and improved readability
    st.markdown("""
        <style>
        .big-font { font-size:20px !important; }
        .stDataFrame { height: 600px !important; }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("📈 Stock Market News Sentiment Analysis")
    stock_name = st.sidebar.text_input("Enter the stock name (e.g., 'Reliance Industries'):")
    
    if st.sidebar.button("Analyze Sentiment"):
        if stock_name:
            with st.spinner("Fetching and analyzing news..."):
                news_articles = get_stock_news(stock_name)
                sentiment_data, sentiment_counts, overall_sentiment = analyze_news_sentiment(news_articles)
            
            st.subheader(f"🔍 {len(news_articles)} News Articles Found")
            
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
            
            st.subheader("📊 Sentiment Analysis Summary")
            st.write(f"✅ Positive: {sentiment_counts['Positive']} | ❌ Negative: {sentiment_counts['Negative']}")
            st.markdown(f"**📈 Overall Market Sentiment: `{overall_sentiment}`**", unsafe_allow_html=True)
        else:
            st.warning("Please enter a stock name to analyze.")

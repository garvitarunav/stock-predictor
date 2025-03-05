import streamlit as st
import requests
import time
from bs4 import BeautifulSoup
from transformers import pipeline
from collections import Counter
import pandas as pd
from tabulate import tabulate  # Ensure this is installed

def stock_sentiment_analysis():
    """Perform sentiment analysis on stock news articles."""
    
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"⚠️ Error loading sentiment analysis model: {e}")
        return

    def fetch_news_google(stock_name):
        """Fetch stock news articles from Google News with error handling."""
        google_url = f"https://news.google.com/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            response = requests.get(google_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            news_list = []
            for item in soup.find_all("h3"):
                if item.a:  # Ensure <a> tag exists
                    title = item.text.strip()
                    link = "https://news.google.com" + item.a["href"][1:]
                    news_list.append({"title": title, "url": link})
            
            return news_list[:20]  # Limit results to top 20 articles

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Network error. Please check your internet connection.")
        except requests.exceptions.Timeout:
            st.warning("⚠️ Request to Google News timed out. Try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Failed to fetch Google News: {e}")

        return []


    def fetch_news_google(stock_name):
        """Fetch stock news articles from Google News."""
        google_url = f"https://news.google.com/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            response = requests.get(google_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return [{"title": item.text.strip(), "url": f"https://news.google.com{item.a['href'][1:]}"} for item in soup.find_all("h3")]
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Failed to fetch Google News: {e}")
            return []

    def analyze_news_sentiment(news_articles):
        """Perform sentiment analysis on news articles."""
        sentiment_counts = Counter()
        if not news_articles:
            return [], Counter(), "Neutral"

        headlines = [article["title"] for article in news_articles]
        try:
            sentiment_results = sentiment_pipeline(headlines)
        except Exception as e:
            st.error(f"⚠️ Sentiment analysis failed: {e}")
            return [], Counter(), "Neutral"

        sentiment_data = []
        for article, sentiment in zip(news_articles, sentiment_results):
            sentiment_label = "Positive" if sentiment["label"] == "POSITIVE" else "Negative"
            sentiment_counts[sentiment_label] += 1
            sentiment_data.append({"Title": article["title"], "Sentiment": sentiment_label, "URL": f"[Click Here]({article['url']})"})

        overall_sentiment = sentiment_counts.most_common(1)[0][0] if sum(sentiment_counts.values()) > 0 else "Neutral"
        return sentiment_data, sentiment_counts, overall_sentiment

    st.sidebar.title("📈 Stock Market News Sentiment Analysis")
    stock_name = st.sidebar.text_input("Enter the stock name (e.g., 'Reliance Industries'):")

    if st.sidebar.button("Analyze Sentiment"):
        if stock_name:
            with st.spinner("Fetching and analyzing news..."):
                news_articles = fetch_news_gnews(stock_name) + fetch_news_google(stock_name)
                sentiment_data, sentiment_counts, overall_sentiment = analyze_news_sentiment(news_articles)

            st.subheader(f"🔍 {len(news_articles)} News Articles Found")

            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                
                # **Use `st.dataframe()` for a better display**
                st.dataframe(df)

                # Alternative: Convert DataFrame to Markdown table
                table_md = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
                st.markdown(table_md)

            st.subheader("📊 Sentiment Analysis Summary")
            st.write(f"✅ Positive: {sentiment_counts['Positive']} | ❌ Negative: {sentiment_counts['Negative']}")
            st.markdown(f"**📈 Overall Market Sentiment: `{overall_sentiment}`**", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a stock name to analyze.")

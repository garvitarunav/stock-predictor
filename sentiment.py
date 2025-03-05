import streamlit as st
import requests
import time
from bs4 import BeautifulSoup
from transformers import pipeline
from collections import Counter
import pandas as pd

def stock_sentiment_analysis():
    # Load Hugging Face Sentiment Analysis Model
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"⚠️ Error loading sentiment analysis model: {e}")
        return

    def fetch_news_gnews(stock_name, max_articles=50, retries=3, wait_time=5):
        """Fetch news articles from GNews API with error handling."""
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": stock_name,
            "lang": "en",
            "country": "in",
            "max": max_articles,
            "apikey": "08eb58904eb6b537bc9e4c9c6d5ddd42"
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()  # Raises an error for 4xx/5xx responses
                articles = response.json().get("articles", [])
                if articles:
                    return articles
                else:
                    st.warning("⚠️ No news articles found from GNews.")
                    return []
            except requests.exceptions.Timeout:
                st.warning(f"⚠️ Timeout occurred, retrying ({attempt+1}/{retries})...")
                time.sleep(wait_time)
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Network issue detected. Please check your internet connection.")
                return []
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ API request failed: {e}")
                return []
        st.error("⚠️ Failed to fetch news from GNews after multiple attempts.")
        return []

    def fetch_news_google(stock_name):
        """Fetch stock news articles from Google News with error handling."""
        google_url = f"https://news.google.com/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            response = requests.get(google_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            news_list = []
            for item in soup.find_all("article"):
                title_element = item.find("h3")
                if title_element and title_element.a:
                    title = title_element.text.strip()
                    link = "https://news.google.com" + title_element.a["href"][1:]
                    news_list.append({"title": title, "url": link})
            if not news_list:
                st.warning("⚠️ No articles found on Google News.")
            return news_list[:20]
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Network error. Please check your internet connection.")
        except requests.exceptions.Timeout:
            st.warning("⚠️ Request to Google News timed out. Try again later.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Failed to fetch Google News: {e}")
        return []

    def get_stock_news(stock_name):
        """Fetch news from both GNews and Google News."""
        gnews_articles = fetch_news_gnews(stock_name)
        google_news_articles = fetch_news_google(stock_name)
        return gnews_articles + google_news_articles

    def analyze_news_sentiment(news_articles):
        """Perform sentiment analysis on news articles with exception handling."""
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
            st.warning("⚠️ Please enter a stock name to analyze.")

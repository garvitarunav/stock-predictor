import streamlit as st
import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from collections import Counter
# Download necessary resources
nltk.download("vader_lexicon")

# Load FinBERT for finance-specific sentiment analysis
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    elif parsed_url.netloc in ("youtu.be",):
        return parsed_url.path.lstrip("/")
    return None

def fetch_transcript(video_id, lang="en"):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def analyze_sentiment_finbert(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = finbert_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    sentiment_labels = ["Bearish 📉", "Neutral 😐", "Bullish 📈"]
    return sentiment_labels[torch.argmax(scores).item()]

def identify_stock_terms(text):
    return list(set(re.findall(r'\b[A-Z]{2,5}\b', text)))

def plot_sentiment_trend(transcript):
    sia = SentimentIntensityAnalyzer()
    times, scores = [], []
    for segment in transcript:
        times.append(segment['start'] / 60)
        scores.append(sia.polarity_scores(segment['text'])["compound"])
    fig, ax = plt.subplots()
    ax.plot(times, scores, marker='o', linestyle='-', color='blue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Sentiment Score (-1 to 1)")
    ax.set_title("Stock Sentiment Trend Over Time")
    ax.grid(True)
    st.pyplot(fig)

def generate_stock_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Stock Market Word Cloud")
    st.pyplot(fig)

def analyze_emotions(text):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    emotion_counts = Counter()
    for chunk in chunks:
        emotions = emotion_classifier(chunk)
        for result in emotions:
            for emotion in result:
                emotion_counts[emotion['label']] += emotion['score']
    total = sum(emotion_counts.values())
    for key in emotion_counts:
        emotion_counts[key] = round((emotion_counts[key] / total) * 100, 2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'blue', 'green', 'orange', 'purple'])
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Percentage")
    ax.set_title("Emotion Analysis of Video Subtitles")
    ax.set_xticklabels(emotion_counts.keys(), rotation=30)
    st.pyplot(fig)

def analyze_youtube_video(video_url):
    st.title("📈 YouTube Stock Sentiment Analyzer")
    st.write("Analyze sentiment, stocks, and emotions from YouTube financial videos.")
    
    video_id = extract_video_id(video_url)
    if video_id:
        transcript = fetch_transcript(video_id)
        if transcript:
            full_text = " ".join([t["text"] for t in transcript])
            df = pd.DataFrame(transcript)
            st.subheader("🎬 Video Transcript")
            st.write(df)
            finbert_sentiment = analyze_sentiment_finbert(full_text)
            st.subheader("📈 Stock Market Sentiment (FinBERT)")
            st.success(finbert_sentiment)
            stock_terms = identify_stock_terms(full_text)
            st.subheader("📊 Sentiment Trend Over Time")
            plot_sentiment_trend(transcript)
            st.subheader("☁️ Stock Market Word Cloud")
            generate_stock_wordcloud(full_text)
            st.subheader("😃 Emotion Analysis")
            analyze_emotions(full_text)
        else:
            st.error("No transcript available.")
    else:
        st.error("Invalid YouTube URL.")

def yt_video():
    st.sidebar.title("For Youtube Analysis")
    video_url = st.sidebar.text_input("Enter YouTube Video URL:")

    if st.sidebar.button("Analyze Video"):
        if video_url:
            analyze_youtube_video(video_url)  # ✅ Correct function name
        else:
            st.warning("Please enter a YouTube video URL.")


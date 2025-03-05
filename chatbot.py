import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load API key from .env file
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def search_google(query):
    """Fetches top search result from Google using SerpAPI."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "en",
        "gl": "us",
        "num": 1  # Fetch only the first result
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        results = response.json().get("organic_results", [])
        return results[0] if results else None
    except requests.exceptions.ConnectionError:
        return "API failed. Check network connection."
    except requests.exceptions.RequestException:
        return "Network error. Try later."

def scrape_website(url):
    """Scrapes the main content of the given URL more accurately."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        main_content = None

        # Try different methods to extract meaningful content
        if soup.find("div", class_="article-content"):
            main_content = soup.find("div", class_="article-content")
        elif soup.find("article"):
            main_content = soup.find("article")
        elif soup.find("div", class_="content"):
            main_content = soup.find("div", class_="content")
        else:
            main_content = soup

        # Extract and clean up text
        paragraphs = main_content.find_all("p")
        text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
        
        if text:
            return text[:2000]  # Return up to 2000 characters to avoid excessive length
        
    except requests.exceptions.ConnectionError:
        return "API failed. Check network connection."
    except requests.exceptions.RequestException:
        return "Network error. Try later."

    return "Failed to fetch detailed content. The website may be blocking scraping."

def typing_effect(text, speed=0.02):
    """Displays text with a typing effect in Streamlit."""
    placeholder = st.empty()
    typed_text = ""

    for char in text:
        typed_text += char
        placeholder.markdown(typed_text)  # Update the displayed text
        time.sleep(speed)  # Adjust typing speed

def get_detailed_answer(query):
    """Fetches a detailed answer with typing effect from Google search result."""
    result = search_google(query)
    
    if isinstance(result, str):  # If an error message is returned
        typing_effect(result)
        return
    
    if not result:
        typing_effect("No results found. Try rephrasing your query.")
        return
    
    title = result.get("title", "No title")
    link = result.get("link", "#")

    # Scrape full answer from the first website
    full_content = scrape_website(link)
    
    response_text = f"🔹 **{title}**\n\n{full_content}\n\n[Read more]({link})"
    typing_effect(response_text)



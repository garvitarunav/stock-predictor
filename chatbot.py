import streamlit as st
import requests
import os
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
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        return results[0] if results else None
    return None

def scrape_website(url):
    """Scrapes the main content of the given URL more accurately."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try different methods to extract meaningful content
        main_content = None
        
        # Check for specific div containers
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
        text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])  # Avoid short lines
        
        if text:
            return text[:2000]  # Return up to 2000 characters to avoid excessive length
        
    return "Failed to fetch detailed content. The website may be blocking scraping."

def get_detailed_answer(query):
    """Gets a full answer from the first search result."""
    result = search_google(query)
    if not result:
        return "No results found. Try rephrasing your query."
    
    title = result.get("title", "No title")
    link = result.get("link", "#")
    
    # Scrape full answer from the first website
    full_content = scrape_website(link)
    
    return f"🔹 **{title}**\n\n{full_content}\n\n[Read more]({link})"

    
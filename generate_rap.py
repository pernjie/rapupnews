from datetime import date, timedelta, datetime
from pydantic import BaseModel, Field
from scrapegraph_py import Client
import feedparser
import requests
import json
import time
import openai
from openai.types.audio import TranscriptionWord
import difflib
import re
import os
from typing import List, NamedTuple, Optional
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip, ImageClip # Install 1.0.3
import numpy as np
from PIL import Image
import librosa
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

today = date.today()

# Define a function to load keys from a file
def load_keys(file_path):
    keys = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore empty lines and comments
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                keys[key] = value
    return keys

# Load keys from the text file
file_path = 'keys.txt'  # Path to your keys file
keys = load_keys(file_path)

# Assign keys to environment variables or directly to variables
OPENROUTER_API_KEY = keys.get('OPENROUTER_API_KEY')
SUNO_COOKIE = keys.get('SUNO_COOKIE')
openai.api_key = keys.get('OPENAI_API_KEY')
SCRAPEGRAPH_API_KEY = keys.get('SCRAPEGRAPH_API_KEY')


class NewsArticle(BaseModel):
    title: str = Field(description="The news article title")
    url: str = Field(description="The news article URL")

class NewsContent(BaseModel):
    content: str = Field(description="The news content")
    image_url: str = Field(description="The URL of the article image")
    news_source: str = Field(description="The source of the news article")

def default_serializer(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # or obj.dict() in Pydantic v1
    raise TypeError(f"Type {type(obj)} not serializable")


###############################################################################
# 1. Gather Articles
###############################################################################

def gather_rss_articles(rss_urls):
    articles = []
    for url in rss_urls:
        print(f"Fetching feed from: {url}")
        articles.extend(fetch_recent_news(url))
        print("=" * 60)
    
    return articles


def gather_reddit_articles(reddit_urls):
    articles = []
    for url in reddit_urls:
        print(f"Fetching feed from: {url}")
        articles.extend(get_reddit_content(url))
        print("=" * 60)
    
    return articles


def fetch_rss_feed(url):
    """
    Fetch RSS feed content using requests with custom headers.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text  # Return the XML content as a string
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the RSS feed: {url}\n{e}")
        return None


def get_actual_url(google_url):
    """Use Selenium to get the actual URL after Google News redirect"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        # Set page load timeout to 10 seconds
        driver.set_page_load_timeout(10)
        driver.get(google_url)
        
        # Wait for redirect to complete (max 10 seconds)
        WebDriverWait(driver, 10).until(
            lambda driver: "news.google.com" not in driver.current_url
        )
        
        actual_url = driver.current_url
        return actual_url
    except Exception as e:
        print(f"Failed to get actual URL: {e}")
        return None
    finally:
        # Ensure driver is always closed, even if an error occurs
        if driver:
            try:
                driver.quit()
            except Exception as e:
                print(f"Failed to close driver: {e}")


def fetch_recent_news(url, max_results=10):
    """
    Fetch and parse an RSS feed, filtering for articles from the last 24 hours.
    """
    # Try parsing the feed directly using feedparser
    feed = feedparser.parse(url)
    if feed.bozo == 0:  # Successfully parsed
        print(f"Parsed successfully with feedparser: {url}")
    else:
        print(f"Direct parsing failed. Attempting fallback with requests: {url}")
        rss_content = fetch_rss_feed(url)
        if rss_content:
            feed = feedparser.parse(rss_content)
        else:
            print(f"Fallback failed for: {url}")
            return []
    
    articles = []

    n = 0
    if 'title' in feed.feed:
        print(f"Feed Title: {feed.feed.title}")
        print("-" * 40)

        for entry in feed.entries[:max_results]:
            published = entry.get('published_parsed')  # Parsed publication time
            if published:
                published_datetime = datetime.fromtimestamp(time.mktime(published))
                # Compare the publication time with the last 24 hours
                if datetime.now() - published_datetime < timedelta(days=1):
                    # For Google News, we need to make a request to get the actual URL
                    if 'news.google.com' in url:
                        google_url = entry.get('link')
                        source_url = get_actual_url(google_url)
                        if not source_url:
                            print(f"Failed to get actual URL for: {entry.title}")
                            continue
                    else:
                        source_url = entry.get('link')

                    print(f"Title: {entry.title.strip()}")
                    print(f"Link: {source_url}")
                    print(f"Published: {published_datetime}")
                    print("-" * 40)
                    n+=1

                    if source_url:  # Only add if we found a valid URL
                        articles.append({"title": entry.title, "url": source_url})
        print(f"{n} articles from {url}")
        print("\n")
    else:
        print(f"Could not retrieve title for feed: {url}\n")
    
    return articles


def categorise_articles(articles):
    print("[INFO] Categorising articles...")

    # Build a prompt that instructs the model to create a rap-style verse based on summaries
    prompt = (
        "You are tasked with categorising news articles from a list of titles:\n"
        "Category 1: 'global' - Articles with global geopolitical significance\n"
        "Category 2: 'local' - Articles with localized impact or relevance\n"
        "Category 3: 'stem' - Articles on science, technology, or related trends\n"
        "Category 4: 'random' - Miscellaneous, interesting or funny articles\n"
        "Return a JSON list with an additional 'category' field for each article, with value being one of the above tags. \n\n"
        "Articles:\n"
    )
    
    prompt += '['
    for idx, article in enumerate(articles):
        prompt += '{'
        prompt += '"id": {}, "title": "{}"'.format(idx, article['title'])
        prompt += '}'
        if idx < len(articles) - 1:  # Add a comma unless it's the last item
            prompt += ',\n'
    prompt += ']'
    
    prompt += (
        "\n\nReturn ONLY the articles in the same format (with additional 'category' field), without preamble or introduction. \n"
        "Cleaned articles in JSON [no prose]:\n"
    )

    # Call your custom OpenRouter function with the prompt
    response = call_openrouter(query=prompt, is_json=True)

    print(response)

    # Parse the string into a Python object
    articles_json = json.loads(response)

    if isinstance(articles_json, dict):
        print("Processing as a dictionary...")
        articles_json = articles_json[next(iter(articles_json))]
    elif isinstance(articles_json, list):
        print("Processing as a list...")

    cleaned_articles = []

    print(articles_json)

    for article in articles_json:
        original_article = articles[article['id']]
        if not article.get('category'):
            continue

        clean_article = {
            'title': original_article['title'],
            'url': original_article['url'],
            'category': article['category']
        }
        cleaned_articles.append(clean_article)

    print("\n")
    print(cleaned_articles)

    # Initialize a dictionary to hold articles by category
    categories = {}

    # Group articles by category
    for article in cleaned_articles:
        category = article["category"]
        if category not in categories:
            categories[category] = []  # Create a new list for this category
        categories[category].append(article)

    # Print the results
    for category, articles in categories.items():
        print(f"Category: {category}")
        for article in articles:
            print(f"  - {article['title']}")

    return categories


def get_news_website(client, news_url):
    try:
        response = client.smartscraper(
            website_url=news_url,
            user_prompt="Extract the news titles and URLs from every section",
            output_schema=NewsArticle
        )
        
        news_data = response['result']['news']
        print("[INFO] Retrieved {} articles from {}".format(len(news_data), news_url))

        # Add articles to the main list
        return [{
                    "title": item["title"],
                    "url": item["url"]
                } for item in news_data]
    except Exception as e:
        print(f"[ERROR] Failed to fetch articles from {news_url}: {e}")
        return []


def get_reddit_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NewsScript/1.0)"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request fails
        
        data = response.json()
        
        articles_list = []
        n = 0

        print("Reddit Feed:")
        print("-" * 40)
        
        # Traverse the children array to fetch each post's title and URL
        for child in data["data"]["children"]:
            child_data = child["data"]
            article = {
                "title": child_data["title"].strip(),
                "url": child_data["url"]
            }
            articles_list.append(article)
            n += 1

            print(f"Title: {child_data['title']}")
            print(f"Link: {child_data['url']}")
            print("-" * 40)

        print(f"{n} articles from Reddit feed.")
        print("\n")

        # Add Reddit articles to the main list
        return articles_list
    except Exception as e:
        print(f"[ERROR] Failed to fetch Reddit articles: {e}")
        return []

###############################################################################
# 2. Filter articles with an LLM for Global Relevance
###############################################################################
def filter_top_articles(category, articles, top_n, yesterday_articles=[]):
    """
    Filters the top N article based on the highest ratings.
    Returns a filtered list of article in the original JSON format.
    """
    print("\n[INFO] Filtering articles for {} with LLM...".format(category))
    prompt = (
        f"From the selection, choose {top_n} articles for a global user.\n"
        "Provide the latest real-time breaking news articles with specific events, updates, or incidents. "
        "AVOID general analysis, historical retrospectives, or articles explaining why things happen. To reiterate:\n"
        "Good: Article is about SPECIFIC EVENTS.\n"
        "Good: Article is interesting to general audience.\n"
        "Bad: Article is a summary, video, compilation, opinion piece, or non-timely articles.\n"
        "Bad: Article is about something too localised with less global relevance.\n"
        "You should give preference to content that have multiple articles on it, but DO NOT select multiple entries of the same content, "
        "just choose the most informative one if needed.\n\n"
        "Avoid Russian-Ukraine war and Israel-Palestine conflict UNLESS there is significant updates.\n"
        "Also avoid any articles already covered yesterday, which are: {}.\n"

        "Articles to select from:\n"
    ).format(yesterday_articles)

    prompt += '['
    for idx, article in enumerate(articles):
        prompt += '{'
        prompt += '"id": {}, "title": "{}"'.format(idx, article['title'])
        prompt += '}'
        if idx < len(articles) - 1:  # Add a comma unless it's the last item
            prompt += ',\n'
    prompt += ']'
    
    prompt += (
        f"\nOnly return a JSON array with the {top_n} entries, and nothing else:\n"
        "[{\"id\": <id>, \"title\": <article title>, \"explanation\": <reasoning>}, ...]\n\n"
        "Articles in JSON:\n"
    )
    print(prompt)

    try:
        # Call the LLM with the constructed prompt
        response = call_openrouter(prompt, model='deepseek/deepseek-chat', is_json=True)
        print("[INFO] LLM successfully rated articles.")
        print(response)

        # Parse the LLM response
        rated_articles = json.loads(response)

        if isinstance(rated_articles, dict):
            print("Processing as a dictionary...")
            # If have multiple keys, wrap it in a list
            if len(rated_articles.keys()) >1:
                rated_articles = [rated_articles]
            # Otherwise, assume list is captured in key
            else:
                rated_articles = rated_articles[next(iter(rated_articles))]
        elif isinstance(rated_articles, list):
            print("Processing as a list...")
        
        # Combine info
        filtered_articles = []
        for rated_article in rated_articles:
            original_article = articles[rated_article['id']]
            filtered_article = {
                'category': category,
                'title': original_article['title'],
                'url': original_article['url'],
                'explanation': rated_article['explanation']
            }
            filtered_articles.append(filtered_article)

        return filtered_articles
    except Exception as e:
        print(f"[ERROR] Failed to rate articles with LLM: {e}")
        return []


###############################################################################
# 3. Scrape Full Article Text
###############################################################################
def scrape_article(url):
    """
    Given a URL, scrape or fetch the full article text.
    Return a string with the main content of the article.
    """
    print(f"\n[INFO] Scraping article from {url} ...")

    client = Client(api_key=SCRAPEGRAPH_API_KEY)
    try:
        response = client.smartscraper(
            website_url=url,
            user_prompt="Summarise the main news content in 1 paragraph, extract the URL of the main image of the article, and the news source (e.g. 'Al Jazeera', 'BBC', 'CNN')",
            output_schema=NewsContent
        )

        print("[INFO] Retrieved scraped data from {}: {}".format(url, response))

        # {
        #     'request_id': '48580a14-a956-4008-ac01-21d8c07b72de', 
        #     'status': 'completed', 
        #     'website_url': 'https://www.aljazeera.com/news/2024/12/31/us-sanctions-russia-and-iran-over-accusations-of-election-interference?traffic_source=rss', 
        #     'user_prompt': 'Summarise the main news content in 1 paragraph, and extract the URL of the main image of the article', 
        #     'result': {
        #         'content': "The United States has imposed new sanctions on Russia and Iran, accusing them of attempting to interfere in the 2024 elections by sowing division among the American populace through disinformation campaigns. The US Treasury Department stated that affiliates of Russia's military intelligence and Iran's Islamic Revolutionary Guard Corps were involved in these efforts. The sanctions target specific organizations and individuals, freezing their US-based assets and prohibiting American entities from engaging in business with them. This move is part of ongoing US efforts to counter foreign influence in its democratic processes.", 
        #         'image_url': 'https://www.aljazeera.com/wp-content/uploads/2024/11/2024-11-04T073531Z_254737801_RC27SAAI7NZU_RTRMADP_3_INDIA-RUPEE-US-ELECTIONS-1-1730714524.jpg?resize=770%2C513&quality=80'
        #     }, 
        #     'error': ''
        # }

        return {
            'content': response['result'].get('content'),
            'image_url': response['result'].get('image_url'),
            'news_source': response['result'].get('news_source')
        }
    except Exception as e:
        print(f"[ERROR] Failed to fetch content from {url}: {e}")
        return None


###############################################################################
# 5. Generate Rap Lyrics from Summaries
###############################################################################
def generate_rap_lyrics(news_summaries):
    """
    Combines multiple article summaries into a single rap verse or multiple verses.
    Returns a string of the final lyrics.
    """
    print("[INFO] Generating rap lyrics from summaries...")

    # Build a prompt that instructs the model to create a rap-style verse based on summaries
    prompt = (
        "You are a master rap lyricist writing for a YouTube channel named 'Rap Up News'. "
        "Use the following news summaries to create lyrics for a rap, delivering the news. "
        "Use one stanza for each news story, try to be more informational, include more details if possible. "
        "Focus on weaving them into a cohesive rap with rhythm and flow. \n\n"
        "Summaries:\n"
    )
    
    for idx, news_summary in enumerate(news_summaries, 1):
        prompt += f"{idx}) {news_summary['title']}\n{news_summary['content']}\n\n"
    
    prompt += (
        "\nPlease keep the tone energetic and the style reminiscent of classic hip-hop, "
        f"but still family-friendly and informational. Please be politically neutral.\n\n"
        "Use the following outline: \n"
        f"Verse 1) Intro, mentioning the date, which is {today}. DO NOT mention the year in the intro.\n"
    )

    for idx, news_summary in enumerate(news_summaries, 1):
        prompt += f"Verse {idx+1}) Article {idx}\n"

    prompt += (f"Verse {len(news_summaries)+2}) End with a catchy outro and thanking the users for watching.\n\n"
        "The first and last stanza should be four lines long, while the other stanzas should be 6 lines long\n"
        "Return ONLY the rap lyrics with no preamble or title, in a JSON object with key 'lyrics', the whole rap all in one string\n"
        "The whole rap should be less than 3000 characters long\n"
        "Rap lyrics in JSON:"
    )

    print(prompt)

    # Call your custom OpenRouter function with the prompt
    rap_lyrics = call_openrouter(query=prompt, model='openai/gpt-4o-2024-11-20', is_json=True)

    # Parse the string into a Python object
    rap_lyrics = json.loads(rap_lyrics)

    if isinstance(rap_lyrics, dict):
        print("Processing as a dictionary...")
        rap_lyrics = rap_lyrics[next(iter(rap_lyrics))]
    
    print(rap_lyrics)

    return rap_lyrics

###############################################################################
# 6. Send Lyrics to Suno (TTS) to Generate Audio
###############################################################################
def transcribe_audio():
    import openai


def transcribe_audio_with_timestamps(audio_file_path):
    try:
        # Open the audio file in binary mode
        with open(audio_file_path, "rb") as audio_file:
            # Send the audio file to Whisper API for transcription
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",  # Ensure we receive timestamps
                timestamp_granularities=["word"]
            )

            print("=============")

            # Parse word-level timestamps
            word_timestamps = []
            for word_info in transcription.words:
                word_timestamps.append({
                    "word": word_info.word,
                    "start": word_info.start,
                    "end": word_info.end
                })
                print(f"Word: {word_info.word}, Start: {word_info.start}s, End: {word_info.end}s")

            return word_timestamps

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class LyricTokenWithLine:
    """
    Holds info about each lyric token:
      - original_token (string with punctuation, etc.)
      - cleaned_token (lowercased, punctuation stripped for matching)
      - line_idx (which line of the lyrics this token belongs to)
      - verse_idx (which verse this token belongs to - verses are separated by double line breaks)
      - end_of_line (bool: is this the last token in that line?)
    """
    def __init__(self, original_token, cleaned_token, line_idx, verse_idx, end_of_line):
        self.original_token = original_token
        self.cleaned_token = cleaned_token
        self.line_idx = line_idx
        self.verse_idx = verse_idx
        self.end_of_line = end_of_line


def tokenize_lyrics_with_lines(lyrics: str) -> List[LyricTokenWithLine]:
    """
    Splits the lyrics into verses (separated by blank lines), then lines, then tokens.
    For each token, store:
      - original_token,
      - cleaned_token (for fuzzy matching),
      - line_idx,
      - verse_idx (based on blank line separation),
      - end_of_line (bool).
    """
    # Split lyrics into lines first
    lines = lyrics.splitlines()
    
    # Group lines into verses by detecting blank lines (lines with only whitespace)
    verses = []
    current_verse = []
    
    for line in lines:
        if not line.strip():  # If line is empty or only contains whitespace
            if current_verse:  # If we have collected some lines
                verses.append(current_verse)
                current_verse = []
        else:
            current_verse.append(line)
    
    # Don't forget the last verse if it exists
    if current_verse:
        verses.append(current_verse)
    
    token_pattern = re.compile(r"[^\w\s']")  # used to strip punctuation for matching
    all_tokens = []
    current_line = 0  # Track actual line number across all verses
    
    for verse_idx, verse_lines in enumerate(verses):
        for line in verse_lines:
            # Skip empty lines (shouldn't happen after our verse splitting)
            if not line.strip():
                continue
            
            # Split line into raw tokens
            raw_tokens = line.split()
            for i, raw in enumerate(raw_tokens):
                # Cleaned token (remove punctuation, lowercase)
                cleaned = token_pattern.sub("", raw).lower()
                end_of_line = (i == len(raw_tokens) - 1)  # last token in this line?
                token_obj = LyricTokenWithLine(raw, cleaned, current_line, verse_idx, end_of_line)
                all_tokens.append(token_obj)
            
            current_line += 1  # Increment line counter only for non-empty lines
    
    return all_tokens


def align_lyrics_to_transcript(
        lyrics: str, 
        recognized_words: List[dict], 
        match_threshold=0.6
    ):
    """
    Align lyric tokens to recognized words using SequenceMatcher for approximate matching.
    Then perform timestamp interpolation for unmatched tokens.
    Ensures no overlapping timestamps by adjusting start times.

    Returns a list of tuples:
       (original_lyric_token, start_time, end_time, line_idx, verse_idx, end_of_line).
    """
    # 1) Get lyric tokens with line info
    lyric_tokens = tokenize_lyrics_with_lines(lyrics)

    # We'll keep just the "cleaned_token" sequence for alignment
    lyric_cleaned_tokens = [t.cleaned_token for t in lyric_tokens]

    # 2) Clean recognized tokens
    recognized_cleaned = []
    for tw in recognized_words:
        clean_word = re.sub(r"[^\w\s']", "", tw['word']).lower()
        recognized_cleaned.append(clean_word)

    # 3) difflib.SequenceMatcher
    matcher = difflib.SequenceMatcher(None, recognized_cleaned, lyric_cleaned_tokens)

    # We'll store alignment as a list, length = len(lyric_tokens),
    # each element is either (start_time, end_time) or None.
    aligned_times = [None] * len(lyric_tokens)

    # Process the opcodes
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # recognized_cleaned[i1:i2] == lyric_cleaned_tokens[j1:j2]
            for idx_r, idx_l in zip(range(i1, i2), range(j1, j2)):
                aligned_times[idx_l] = (
                    recognized_words[idx_r]['start'],
                    recognized_words[idx_r]['end']
                )
        else:
            # handle replace, insert, delete
            rec_segment = recognized_cleaned[i1:i2]
            lyr_segment = lyric_cleaned_tokens[j1:j2]
            for offset_l, lyric_word in enumerate(lyr_segment):
                best_score = 0.0
                best_idx = None
                for offset_r, rec_word in enumerate(rec_segment):
                    score = difflib.SequenceMatcher(None, lyric_word, rec_word).ratio()
                    if score > best_score:
                        best_score = score
                        best_idx = offset_r
                
                if best_idx is not None and best_score >= match_threshold:
                    actual_rec_index = i1 + best_idx
                    aligned_times[j1 + offset_l] = (
                        recognized_words[actual_rec_index]['start'],
                        recognized_words[actual_rec_index]['end']
                    )

    # 4) Interpolate timestamps for any None entries
    aligned_times = fill_unmatched_times(aligned_times)

    # 5) Adjust timestamps to prevent overlaps
    for i in range(1, len(aligned_times)):
        prev_end = aligned_times[i-1][1]
        curr_start, curr_end = aligned_times[i]
        
        # If current start is before previous end, adjust it
        if curr_start < prev_end:
            # Set current start to previous end
            aligned_times[i] = (prev_end, curr_end)
            
            # If this causes start to be after end, adjust end too
            if prev_end > curr_end:
                # Add a small buffer (e.g., 0.05 seconds)
                aligned_times[i] = (prev_end, prev_end + 0.05)

    # 6) Build final results
    results = []
    for (token_obj, times) in zip(lyric_tokens, aligned_times):
        if times is not None:
            start_time, end_time = times
        else:
            start_time, end_time = None, None
        results.append((
            token_obj.original_token,  # keep original punctuation/casing
            start_time,
            end_time,
            token_obj.line_idx,
            token_obj.verse_idx,
            token_obj.end_of_line
        ))

    print(json.dumps(results, indent=2))

    return results

def fill_unmatched_times(times_list: List[Optional[tuple]]) -> List[Optional[tuple]]:
    """
    Given a list of (start, end) or None, interpolate to fill the None values.
    Rules:
      - If tokens at the start are None, assign them the time of the first non-None token.
      - If tokens at the end are None, assign them the time of the last non-None token.
      - For runs of None in the middle, linearly interpolate start/end times 
        between the matched tokens on either side.
    """
    n = len(times_list)
    if n == 0:
        return times_list
    
    times = list(times_list)  # copy

    # 1) Find first non-None
    first_matched = None
    for i in range(n):
        if times[i] is not None:
            first_matched = i
            break
    if first_matched is None:
        # means no matched tokens at all
        return times

    # Fill from 0..first_matched-1
    for i in range(first_matched):
        times[i] = times[first_matched]

    # 2) Find last non-None
    last_matched = None
    for i in range(n - 1, -1, -1):
        if times[i] is not None:
            last_matched = i
            break
    # Fill from last_matched+1..end
    for i in range(last_matched + 1, n):
        times[i] = times[last_matched]

    # 3) Interpolate any runs of None in the middle
    i = 0
    while i < n:
        if times[i] is None:
            start_run = i - 1
            while i < n and times[i] is None:
                i += 1
            end_run = i
            # times[start_run] and times[end_run] are not None
            t1_start, t1_end = times[start_run]
            t2_start, t2_end = times[end_run]

            gap_size = end_run - (start_run + 1)
            for pos in range(1, gap_size + 1):
                fraction = pos / (gap_size + 1)
                interp_start = t1_start + fraction * (t2_start - t1_start)
                interp_end   = t1_end   + fraction * (t2_end   - t1_end)
                times[start_run + pos] = (interp_start, interp_end)
        else:
            i += 1

    return times


###############################################################################
# 7. Post-processing, Upload, etc.
###############################################################################
def download_image(url: str, save_path: str) -> bool:
    """
    Downloads an image from a URL and saves it to the specified path.
    Returns True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set up headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Download the image
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if the response is actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            print(f"[WARNING] URL does not point to an image: {url}")
            return False
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        # Verify the image can be opened
        try:
            with Image.open(save_path) as img:
                # Convert to RGB if needed (handles PNG with transparency)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                    img.save(save_path, 'JPEG', quality=85)
                
                # Resize if too large
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(save_path, 'JPEG', quality=85)
                
        except Exception as e:
            print(f"[ERROR] Failed to process downloaded image: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return False
            
        print(f"[INFO] Successfully downloaded and processed image: {save_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download image from {url}: {e}")
        return False

def download_images(news_summaries, selected_date=None):
    """
    Downloads images for each news summary and saves them to the video/images directory.
    Creates a mapping of article indices to their image paths.
    """
    print("\n[INFO] Downloading article images...")
    
    # Use selected_date if provided, otherwise use today
    image_date = selected_date if selected_date else str(date.today())
    
    # Create images directory if it doesn't exist
    os.makedirs("video/images", exist_ok=True)
    
    # Track successful downloads
    image_paths = {}
    
    for idx, news_summary in enumerate(news_summaries):
        image_url = news_summary.get('image_url')
        if not image_url:
            print(f"[WARNING] No image URL for article {idx}")
            continue
            
        # Generate save path using the selected date
        image_path = f"video/images/{image_date}-{idx}.jpg"
        
        # Skip if file already exists
        if os.path.exists(image_path):
            print(f"[INFO] Image already exists at {image_path}, skipping download")
            image_paths[idx] = image_path
            continue
        
        # Download and process the image
        if download_image(image_url, image_path):
            image_paths[idx] = image_path
        else:
            print(f"[WARNING] Failed to download image for article {idx}")
            # If download fails, try to use a default image based on category
            category = news_summary.get('category', 'default')
            default_image = f"video/default_images/{category}.jpg"
            if os.path.exists(default_image):
                image_paths[idx] = default_image
    
    print(f"[INFO] Downloaded {len(image_paths)} images successfully")
    return image_paths

def detect_beats(audio_file_path, threshold=0.5):
    """
    Detect beats in the audio file using librosa.
    Returns a list of beat timestamps in seconds.
    Uses tempo detection to find consistent beats.
    """
    print("[INFO] Detecting beats from audio...")
    
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=85, tightness=100)
    print(f"[INFO] Detected tempo: {tempo} BPM")
    
    # Calculate beat times from frames
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Get onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Find peaks in onset strength that align with beat frames
    peak_idxs = []
    window_size = int(sr * 0.05)  # 50ms window to look for peaks
    
    for beat_frame in beat_frames:
        # Look for the strongest onset near each beat
        start = max(0, beat_frame - window_size)
        end = min(len(onset_env), beat_frame + window_size)
        
        # Get the maximum onset strength in this window
        window_max = np.max(onset_env[start:end])
        
        # Only keep beats with significant onset strength
        if window_max > threshold * np.max(onset_env):
            peak_idxs.append(beat_frame)
    
    # Convert frames to times
    beat_times = librosa.frames_to_time(peak_idxs, sr=sr)
    
    # Ensure beats are evenly spaced based on tempo
    beat_period = 60.0 / tempo  # Time between beats in seconds
    
    # Create a regular grid of beats based on the tempo
    num_beats = int(librosa.get_duration(y=y, sr=sr) / beat_period)
    regular_beats = np.arange(num_beats) * beat_period
    
    # Match detected beats to the regular grid
    final_beats = []
    for reg_beat in regular_beats:
        # Find the closest detected beat
        closest_beats = [b for b in beat_times if abs(b - reg_beat) < beat_period * 0.25]
        if closest_beats:
            # Use the detected beat if it exists
            final_beats.append(closest_beats[0])
        else:
            # Use the regular beat if no detected beat is close enough
            final_beats.append(reg_beat)
    
    print(f"[INFO] Detected {len(final_beats)} regular beats at {tempo} BPM")
    return np.array(final_beats)

def create_rapper_animation(rapper_paths, beat_times, video_duration, video_height):
    """
    Creates a list of rapper clips that alternate on beats.
    Returns a list of positioned and timed ImageClips.
    """
    print("[INFO] Creating rapper animation...")
    
    rapper_clips = []
    
    # Load and process both rapper images
    rapper_images = []
    for path in rapper_paths:
        img = Image.open(path)
        # Calculate dimensions for rapper
        rapper_height = int(video_height * 0.5)
        rapper_width = int(rapper_height * img.width / img.height)
        # Resize rapper image
        img = img.resize((rapper_width, rapper_height), Image.Resampling.LANCZOS)
        rapper_images.append(img)
    
    # Save temporary resized images
    temp_paths = []
    for idx, img in enumerate(rapper_images):
        temp_path = f"video/temp_rapper_{idx}.png"
        img.save(temp_path)
        temp_paths.append(temp_path)
    
    # Create clips that alternate on beats
    current_time = 0
    use_alternate = False
    
    for beat_time in beat_times:
        # Create clip for duration until this beat
        if beat_time > current_time:
            # Create clip with current image
            clip = (ImageClip(temp_paths[1 if use_alternate else 0])
                   .set_start(current_time)
                   .set_duration(beat_time - current_time)
                   .set_position((-100, video_height - rapper_height + 50)))
            rapper_clips.append(clip)
        
        # Create longer clip for the beat
        beat_duration = 0.1  # Increased from 0.1 to 0.2 seconds
        if beat_time + beat_duration <= video_duration:
            clip = (ImageClip(temp_paths[0 if use_alternate else 1])
                   .set_start(beat_time)
                   .set_duration(beat_duration)
                   .set_position((-100, video_height - rapper_height + 50)))
            rapper_clips.append(clip)
        
        current_time = beat_time + beat_duration
        use_alternate = not use_alternate
    
    # Fill remaining time if needed
    if current_time < video_duration:
        clip = (ImageClip(temp_paths[1 if use_alternate else 0])
               .set_start(current_time)
               .set_duration(video_duration - current_time)
               .set_position((-100, video_height - rapper_height + 50)))
        rapper_clips.append(clip)
    
    # Clean up temporary files
    for temp_path in temp_paths:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return rapper_clips

def generate_video(news_summaries, audio_file_path, timestamped_lyrics, debug=False):
    # Parameters
    video_width = 1280
    video_height = 720
    font_color = "black"
    box_margin = 40
    debug_duration = 3.0  # Debug mode will generate 3 seconds
    
    # Add audio to the video and determine duration first
    audio_clip = AudioFileClip(audio_file_path)
    video_duration = debug_duration if debug else audio_clip.duration
    
    if debug:
        # In debug mode, only keep lyrics that appear in first 3 seconds
        timestamped_lyrics = [
            (word, start, end, line, verse, is_end) 
            for word, start, end, line, verse, is_end in timestamped_lyrics 
            if start < debug_duration
        ]
        # If no lyrics in first 3 seconds, add a dummy lyric for testing
        if not timestamped_lyrics:
            timestamped_lyrics = [
                ("Debug Test Lyric", 0.0, debug_duration, 0, 0, True)
            ]
        
        # Trim audio to debug duration
        audio_clip = audio_clip.subclip(0, debug_duration)

    # Convert hex color to RGB (needed for background)
    def hex_to_rgb(hex_color):
        # Remove the '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Format the date (e.g., "30th December 2024")
    def ordinal(n):
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return str(n) + suffix
    
    formatted_date = today.strftime(f"{ordinal(today.day)} %B %Y").upper()
    
    # Create persistent date text at the top with transparent background
    date_clip = (TextClip(formatted_date,
                         fontsize=26,
                         color='#7C878A',
                         bg_color='transparent',  # Use 'transparent' instead of None
                         font='./video/fonts/classica_3/Classica-Book.ttf',
                         kerning=12)
                 .set_position(('center', 125)))
    
    def find_optimal_font_size(text, max_width, max_height, font_path, initial_size=70):
        """
        Binary search to find the largest font size that fits the text within given dimensions
        """
        min_size = 30  # Increased minimum size for readability
        max_size = 120  # Increased maximum size
        optimal_size = min_size
        
        while min_size <= max_size:
            mid_size = (min_size + max_size) // 2
            
            # Create test clip with explicit width constraint
            test_clip = TextClip(
                text,
                fontsize=mid_size,
                font=font_path,
                color=font_color,
                bg_color='transparent',
                align="West",
                method="caption",
                size=(max_width, None)
            )
            
            # Get actual dimensions after text wrapping
            clip_width = test_clip.w
            clip_height = test_clip.h
            
            # Check if text fits within bounds
            if clip_width <= max_width and clip_height <= max_height:
                optimal_size = mid_size
                min_size = mid_size + 1
            else:
                max_size = mid_size - 1
            
            # Clean up the test clip
            test_clip.close()
        
        return optimal_size

    # Text box dimensions and position
    text_box_width = video_width * 0.55  # 55% of screen width
    text_box_height = video_height * 0.3  # Reduced height to prevent too many lines
    left_position = video_width * 0.45  # Moved from 0.4 to 0.45 (more to the right)
    vertical_position = 300  # This is where both text and images should align
    
    output_path = f"video/Rap Up News {today}.mp4"
    
    # Newspaper background image
    # Load and resize background image using PIL first
    img = Image.open("video/newspaper.png")
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = img.height / img.width
    new_height = int(video_width * aspect_ratio)
    # Resize image
    img = img.resize((video_width, new_height), Image.Resampling.LANCZOS)
    # Center crop if needed
    if new_height > video_height:
        top = (new_height - video_height) // 2
        bottom = top + video_height
        img = img.crop((0, top, video_width, bottom))
    # Save temporary resized image
    temp_path = "video/temp_resized.png"
    img.save(temp_path)
    
    # Create ImageClip from resized image
    background_image = ImageClip(temp_path)
    
    # Detect beats from audio
    beat_times = detect_beats(audio_file_path)
    
    # Create rapper animation clips
    rapper_paths = ["video/rapper.png", "video/rapper_alt.png"]  # Make sure you have both images
    rapper_clips = create_rapper_animation(
        rapper_paths=rapper_paths,
        beat_times=beat_times,
        video_duration=video_duration,
        video_height=video_height
    )

    # DEBUG: Create a semi-transparent box to show text boundaries
    debug_box = ColorClip(
        size=(int(text_box_width - (2 * box_margin)), int(text_box_height)),
        color=(255, 0, 0)  # Pure red
    ).set_position((left_position + box_margin, vertical_position))
    debug_box = debug_box.set_duration(video_duration)
    
    # Set duration for clips
    date_clip = date_clip.set_duration(video_duration)
    background_image = background_image.set_duration(video_duration)

    # Group lyrics by line
    from collections import defaultdict
    lines = defaultdict(list)
    for word, start, end, line, verse, is_end in timestamped_lyrics:
        lines[line].append((word, start, end))

    # Create a list of clips
    text_clips = []
    
    # Create clips line by line with transparent background
    for line_num, words in lines.items():
        # Get the full text for the line
        full_line = " ".join(word for word, _, _ in words)
        
        # Get timing for the whole line
        line_start = min(start for _, start, _ in words)
        line_end = max(end for _, _, end in words)
        
        # Find the optimal font size for this line
        optimal_font_size = find_optimal_font_size(
            text=full_line,
            max_width=text_box_width - (2 * box_margin),
            max_height=text_box_height,
            # font_path='./video/fonts/classica_3/Classica-Bold.ttf'
            font_path='./video/fonts/rokkitt/Rokkitt-SemiBold.ttf'
        )
        
        # Create a clip for the full line with transparent background
        line_clip = TextClip(
            full_line,
            fontsize=optimal_font_size,
            color=font_color,
            size=(text_box_width - (2 * box_margin), None),
            bg_color='transparent',
            align="West",
            method="caption",
            # font='./video/fonts/classica_3/Classica-Bold.ttf',
            font='./video/fonts/rokkitt/Rokkitt-SemiBold.ttf',
            kerning=-2
        )
        
        # Calculate vertical position to center the text in the box
        text_height = line_clip.h
        centered_y = vertical_position + (text_box_height - text_height) // 2
        
        # Set position with centered y-coordinate
        line_clip = line_clip.set_position((left_position + box_margin, centered_y))
        
        # Set the timing for this line
        line_clip = line_clip.set_start(line_start).set_duration(line_end - line_start)
        text_clips.append(line_clip)

    # Create background clip with solid color (using RGB)
    background_color_clip = ColorClip(size=(video_width, video_height), 
                                    color=hex_to_rgb("#1F324B"))
    background_color_clip = background_color_clip.set_duration(video_duration)
    
    # Article image
    # Calculate dimensions for article image
    article_width = int(video_width * 0.35)  # 35% of screen width
    article_height = int(video_height * 0.3)  # Match text box height
    article_x = 80  # Increased from 40 to 80 (more to the right)
    article_y = vertical_position  # Use same vertical position as text box

    # Create article image clips for each verse
    article_clips = []
    current_verse = -1
    
    # Function to create bordered image
    def add_border_to_image(img, border_size=8):
        # Create new image with border
        new_width = img.width + (2 * border_size)
        new_height = img.height + (2 * border_size)
        bordered = Image.new('RGB', (new_width, new_height), 'black')
        # Paste original image in center
        bordered.paste(img, (border_size, border_size))
        return bordered

    # Load placeholder image once
    try:
        placeholder_img = Image.open("video/placeholder.png")
        
        # Calculate dimensions maintaining aspect ratio
        img_aspect = placeholder_img.width / placeholder_img.height
        if img_aspect > (article_width / article_height):
            # Image is wider than target - fit to height
            new_height = article_height
            new_width = int(article_height * img_aspect)
        else:
            # Image is taller than target - fit to width
            new_width = article_width
            new_height = int(article_width / img_aspect)
        
        # Resize image preserving aspect ratio
        placeholder_img = placeholder_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Add border
        placeholder_img = add_border_to_image(placeholder_img)
        
        # Save temporary placeholder
        placeholder_temp_path = "video/temp_placeholder.png"
        placeholder_img.save(placeholder_temp_path)
    except Exception as e:
        print(f"[ERROR] Failed to load placeholder image: {e}")
        placeholder_temp_path = None

    # Function to create image clip for a verse
    def create_verse_image_clip(verse_idx, verse_start, verse_end):
        # Skip middle verses (they use article images)
        if verse_idx > 0 and verse_idx < len(news_summaries) + 1:
            article_idx = verse_idx - 1
            news_summary = news_summaries[article_idx]
            news_source = news_summary.get('news_source', 'News')  # Get news source, default to 'News'
            
            # Get image path from the downloaded images using selected_date from audio_file_path
            selected_date = os.path.basename(audio_file_path).split(' ')[3].replace('.mp3', '')
            image_path = f"video/images/{selected_date}-{article_idx}.jpg"
            if not os.path.exists(image_path):
                print("Article image not found")
                # Try default image if article image doesn't exist
                category = news_summary.get('category', 'default')
                image_path = f"video/default_images/{category}.jpg"
                if not os.path.exists(image_path):
                    return create_placeholder_clip(verse_start, verse_end)

        else:
            # For intro/outro verses, use placeholder
            return create_placeholder_clip(verse_start, verse_end)

        try:
            # Load and process article image using PIL
            article_img = Image.open(image_path)
            
            # Calculate dimensions maintaining aspect ratio
            img_aspect = article_img.width / article_img.height
            if img_aspect > (article_width / article_height):
                new_height = article_height
                new_width = int(article_height * img_aspect)
            else:
                new_width = article_width
                new_height = int(article_width / img_aspect)
            
            # Resize image
            article_img = article_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Add border
            article_img = add_border_to_image(article_img)
            
            # Save temporary resized image
            article_temp_path = f"video/temp_article_{verse_idx}.jpg"
            article_img.save(article_temp_path)
            
            # Create ImageClip from resized image
            img_clip = ImageClip(article_temp_path)
            
            # Center the image vertically
            y_offset = (article_height - new_height) // 2
            img_clip = img_clip.set_position((article_x, article_y + y_offset))
            
            # Create source text clip
            source_clip = TextClip(
                news_source,
                fontsize=24,
                color=font_color,
                bg_color='transparent',
                font='./video/fonts/rokkitt/Rokkitt-Medium.ttf'
            )
            
            # Position source text above the image
            source_x = article_x
            source_y = article_y + y_offset - source_clip.h - 10  # 10px padding
            source_clip = source_clip.set_position((source_x, source_y))
            
            # Set duration for both clips
            duration = verse_end - verse_start
            img_clip = img_clip.set_start(verse_start).set_duration(duration)
            source_clip = source_clip.set_start(verse_start).set_duration(duration)
            
            # Return both clips as a list
            return [img_clip, source_clip]
            
        except Exception as e:
            print(f"[ERROR] Failed to create image clip for verse {verse_idx}: {e}")
            return create_placeholder_clip(verse_start, verse_end)

    def create_placeholder_clip(start_time, end_time):
        if not placeholder_temp_path:
            return None
            
        try:
            img_clip = ImageClip(placeholder_temp_path)
            img_clip = img_clip.set_position((article_x, article_y))
            img_clip = img_clip.set_start(start_time).set_duration(end_time - start_time)
            return img_clip
        except Exception as e:
            print(f"[ERROR] Failed to create placeholder clip: {e}")
            return None

    # Create image clips for each verse
    for verse_idx in range(max(v for _, _, _, _, v, _ in timestamped_lyrics) + 1):
        # Get timing for the verse
        verse_words = [(s, e) for _, s, e, _, v, _ in timestamped_lyrics if v == verse_idx]
        if verse_words:
            verse_start = verse_words[0][0]
            verse_end = verse_words[-1][1]
            
            # Create and add image clip(s)
            clips = create_verse_image_clip(verse_idx, verse_start, verse_end)
            if isinstance(clips, list):
                article_clips.extend(clips)
            elif clips:
                article_clips.append(clips)

    # Get the end time of the last lyric
    last_lyric_end = max(end for _, _, end, _, _, _ in timestamped_lyrics)
    
    # Create "Like and Subscribe" clip that appears after lyrics end
    subscribe_text = "Subscribe for your daily dose of Rap Up News\nStay Informed, Stay Entertained!"
    subscribe_clip = TextClip(
        subscribe_text,
        fontsize=50,
        color=font_color,
        size=(video_width * 0.8, None),
        bg_color='transparent',
        align="center",
        method="caption",
        font='./video/fonts/rokkitt/Rokkitt-Medium.ttf',
        kerning=-2
    )
    
    # Center the clip horizontally and vertically
    subscribe_x = (video_width - subscribe_clip.w) // 2
    subscribe_y = (video_height - subscribe_clip.h) // 2
    
    # Position and time the clip to appear after lyrics end
    subscribe_clip = subscribe_clip.set_position((subscribe_x, subscribe_y))
    subscribe_clip = subscribe_clip.set_start(last_lyric_end).set_duration(video_duration - last_lyric_end)
    
    # Layer the clips: background color -> background image -> article images -> rapper animation -> date -> debug box -> lyrics text -> subscribe clip
    composite_clip = CompositeVideoClip([
        background_color_clip,
        background_image,
        *article_clips,
        *rapper_clips,
        date_clip,
        # debug_box,
        *text_clips,
        subscribe_clip  # Add the subscribe clip last so it appears on top
    ], size=(video_width, video_height)).set_duration(video_duration)

    final_video = composite_clip.set_audio(audio_clip)

    # Export the video
    if debug:
        output_path = f"video/Debug_Rap Up News {today}.mp4"
    else:
        output_path = f"video/Rap Up News {today}.mp4"
    final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    
    # Clean up temporary files
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if os.path.exists(placeholder_temp_path):
        os.remove(placeholder_temp_path)
    # Clean up any temporary article images
    for f in os.listdir("video"):
        if f.startswith("temp_article_"):
            os.remove(os.path.join("video", f))

    return output_path

def get_youtube_credentials():
    """
    Handles YouTube API authentication and validates credentials
    """
    # Constants
    CLIENT_SECRETS_FILE = 'client_secrets.json'
    TOKEN_PICKLE_FILE = 'token.pickle'
    
    print("[INFO] Checking YouTube credentials...")
    
    # First verify client secrets file exists
    if not os.path.exists(CLIENT_SECRETS_FILE):
        raise FileNotFoundError(
            f"Missing {CLIENT_SECRETS_FILE}. Please download it from Google Cloud Console:\n"
            "1. Go to https://console.cloud.google.com\n"
            "2. Create a project or select an existing project\n"
            "3. Enable the YouTube Data API v3\n"
            "4. Go to Credentials\n"
            "5. Create an OAuth 2.0 Client ID\n"
            "6. Download the client configuration file\n"
            "7. Save it as 'client_secrets.json' in the project directory"
        )

    creds = None
    
    # Remove existing token if it exists to force re-authentication
    if os.path.exists(TOKEN_PICKLE_FILE):
        try:
            # Try to load existing credentials
            with open(TOKEN_PICKLE_FILE, 'rb') as token:
                creds = pickle.load(token)
            
            # Test if credentials are valid
            if creds and creds.valid:
                try:
                    # Build service and make a simple API call to test credentials
                    youtube = build('youtube', 'v3', credentials=creds)
                    youtube.channels().list(part='snippet', mine=True).execute()
                    print("[INFO] Existing credentials are valid")
                    return creds
                except Exception as e:
                    print(f"[WARNING] Existing credentials failed: {str(e)}")
                    creds = None
                    # Delete invalid token file
                    os.remove(TOKEN_PICKLE_FILE)
            else:
                print("[WARNING] Existing credentials are invalid")
                creds = None
                os.remove(TOKEN_PICKLE_FILE)
                
        except Exception as e:
            print(f"[WARNING] Error loading credentials: {str(e)}")
            creds = None
            # Delete corrupted token file
            if os.path.exists(TOKEN_PICKLE_FILE):
                os.remove(TOKEN_PICKLE_FILE)

    # If no valid credentials available, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] Refreshing expired credentials")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[WARNING] Failed to refresh credentials: {str(e)}")
                creds = None
        
        if not creds:
            print("[INFO] Getting new credentials")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRETS_FILE,
                    ['https://www.googleapis.com/auth/youtube.upload']
                )
                creds = flow.run_local_server(port=0)
                
                # Save new credentials
                with open(TOKEN_PICKLE_FILE, 'wb') as token:
                    pickle.dump(creds, token)
                print("[INFO] New credentials saved successfully")
                
            except Exception as e:
                raise Exception(
                    f"Failed to get new credentials: {str(e)}\n"
                    "Please verify your client_secrets.json is valid and "
                    "the YouTube Data API is enabled in your Google Cloud Console"
                )
    
    return creds

def upload_to_youtube(video_path, thumbnail_path, title, video_description):
    """
    Uploads video to YouTube with optimized metadata for maximum visibility.
    Schedules the video to be published at 00:30 the next day.
    """
    print("[INFO] Preparing YouTube upload...")
    
    try:
        # Get credentials and build service
        creds = get_youtube_credentials()
        youtube = build('youtube', 'v3', credentials=creds)
        
        # Calculate publish time (00:30 next day Singapore time)
        current_date = datetime.now()
        next_day = current_date + timedelta(days=1)
        # Convert to Singapore time (UTC+8)
        publish_date = next_day.replace(hour=0, minute=30, second=0, microsecond=0) - timedelta(hours=8)
        
        # Convert to RFC3339 format required by YouTube API
        publish_time = publish_date.isoformat() + 'Z'
        
        # Prepare video metadata
        body = {
            'snippet': {
                'title': title,
                'description': video_description,
                'tags': ['news', 'rap', 'daily news', 'news summary', 'rap news'],
                'categoryId': '25'  # News category
            },
            'status': {
                'privacyStatus': 'private',  # Set to private initially
                'publishAt': publish_time,  # Schedule publish time
                'selfDeclaredMadeForKids': False
            }
        }
        
        # Upload video file
        print("[INFO] Uploading video file...")
        print(f"[INFO] Video will be published at: {publish_time}")
        
        # Create MediaFileUpload object with retry parameters
        media = MediaFileUpload(
            video_path,
            chunksize=1024*1024,  # 1MB chunks
            resumable=True,
            mimetype='video/mp4'
        )
        
        insert_request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        # Execute upload with retry logic
        response = None
        retries = 3
        retry_delay = 5  # seconds
        
        while response is None and retries > 0:
            try:
                status, response = insert_request.next_chunk()
                if status:
                    print(f"[INFO] Uploaded {int(status.progress() * 100)}%")
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e
                print(f"[WARNING] Upload chunk failed, retrying in {retry_delay} seconds... ({retries} retries left)")
                print(f"[WARNING] Error details: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        if response is None:
            raise Exception("Upload failed after all retries")
        
        video_id = response['id']
        print(f"[INFO] Video upload complete! Video ID: {video_id}")
        
        # Upload thumbnail with retry logic
        if thumbnail_path and os.path.exists(thumbnail_path):
            print("[INFO] Uploading thumbnail...")
            thumbnail_retries = 3
            while thumbnail_retries > 0:
                try:
                    youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(thumbnail_path)
                    ).execute()
                    print("[INFO] Thumbnail upload complete!")
                    break
                except Exception as e:
                    thumbnail_retries -= 1
                    if thumbnail_retries == 0:
                        print(f"[WARNING] Failed to upload thumbnail: {e}")
                        break
                    print(f"[WARNING] Thumbnail upload failed, retrying... ({thumbnail_retries} retries left)")
                    time.sleep(2)
        
        video_url = f"https://youtu.be/{video_id}"
        print(f"[INFO] Video uploaded successfully and scheduled for {publish_time}: {video_url}")
        return video_url
        
    except Exception as e:
        print(f"[ERROR] An error occurred during YouTube upload: {str(e)}")
        print("[ERROR] Full error details:")
        import traceback
        traceback.print_exc()
        raise

###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    pass

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

def call_openrouter(query, model=None, is_json=False):
    print("Calling openrouter with query length: " + str(len(query)))
    if not model:
        model = "openai/gpt-4o-mini"

    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
      },
      data=json.dumps({
        "model": model,
        "response_format": { "type": "json_object" } if is_json else None,
        "messages": [
          { "role": "user", "content": query }
        ]
      })
    )

    # Check if the request was successful
    try:
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()

            # Print the results
            return response_data['choices'][0]['message']['content']
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")

            raise Exception(response.text)
    except:
        print(response_data)
        raise Exception(response.text)

# Rap Up News

An automated news aggregation and rap generation system that creates daily rap summaries of top news stories.

## Overview

Rap Up News automatically:
1. Gathers news from various sources (RSS feeds and Reddit)
2. Categorizes and filters the most important stories
3. Generates rap lyrics summarizing the news
4. Creates a video with synchronized lyrics and news imagery
5. Uploads the final video to YouTube

## Prerequisites

- Python 3.8+
- FFmpeg installed and in system PATH
- YouTube API credentials
- Various API keys (OpenAI, OpenRouter, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pernjie/rapupnews.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up configuration files:
   - Create `config.json` and customize news sources
   - Create `keys.txt` with your API keys
   - Set up YouTube credentials:
     - Create a project in Google Cloud Console
     - Enable YouTube Data API v3
     - Create OAuth 2.0 credentials
     - Download and rename credentials to `client_secrets.json`

## Usage

1. Start the web interface:
```bash
python web_interface.py
```

2. Open `http://localhost:5000` in your browser

3. Follow the pipeline steps:
   - Gather Articles
   - Categorize Articles
   - Filter Top Stories
   - Generate Rap Lyrics
   - Process Audio
   - Generate Video
   - Upload to YouTube

## Project Structure

- `web_interface.py`: Main Flask application and pipeline controller
- `generate_rap.py`: Core news processing and content generation logic
- `templates/`: Flask HTML templates
- `pipeline_data/`: Storage for pipeline state and outputs
- `video/`: Video generation assets and output
- `audio/`: Audio files for rap generation

## Disclaimer

This project is only for fun and not meant to be a good news outlet. Do what you want with it.
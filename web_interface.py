from flask import Flask, render_template, jsonify, request, send_from_directory
import json
from datetime import date, datetime, timedelta
import os
from generate_rap import (
    gather_rss_articles, 
    gather_reddit_articles,
    categorise_articles,
    filter_top_articles,
    scrape_article,
    generate_rap_lyrics,
    transcribe_audio_with_timestamps,
    align_lyrics_to_transcript,
    download_images,
    generate_video,
    upload_to_youtube,
    Client
)
import glob
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from googleapiclient.discovery import build
import requests
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Initialize state
DATA_DIR = 'pipeline_data'  # New base directory for all pipeline data
OLD_STATE_FILE = 'pipeline_state.json'  # For backward compatibility

def ensure_data_dir():
    """Ensure the data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_state_file(selected_date=None):
    if selected_date is None:
        selected_date = str(date.today())
    return os.path.join(DATA_DIR, f'pipeline_state_{selected_date}.json')

def get_output_dir(selected_date=None):
    if selected_date is None:
        selected_date = str(date.today())
    return os.path.join(DATA_DIR, f'outputs_{selected_date}')

def migrate_old_state():
    """Migrate data from old state file if it exists"""
    if os.path.exists(OLD_STATE_FILE):
        try:
            with open(OLD_STATE_FILE, 'r') as f:
                old_state = json.load(f)
                old_date = old_state.get('date', str(date.today()))
                
            # Save to new location
            new_state_file = get_state_file(old_date)
            if not os.path.exists(new_state_file):
                os.makedirs(os.path.dirname(new_state_file), exist_ok=True)
                with open(new_state_file, 'w') as f:
                    json.dump(old_state, f, indent=2)
                
            # Move old outputs if they exist
            old_output_dir = 'outputs'
            if os.path.exists(old_output_dir):
                new_output_dir = get_output_dir(old_date)
                if not os.path.exists(new_output_dir):
                    os.makedirs(new_output_dir, exist_ok=True)
                    # Move all files from old output dir to new
                    for file in os.listdir(old_output_dir):
                        old_path = os.path.join(old_output_dir, file)
                        new_path = os.path.join(new_output_dir, file)
                        if os.path.isfile(old_path):
                            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Error migrating old state: {e}")

def get_available_dates():
    """Get list of dates that have state files"""
    ensure_data_dir()
    migrate_old_state()  # Migrate old data if it exists
    
    dates = set()
    
    # Look for state files in the data directory
    state_files = glob.glob(os.path.join(DATA_DIR, 'pipeline_state_*.json'))
    for file in state_files:
        try:
            # Extract date from filename
            date_str = os.path.basename(file).replace('pipeline_state_', '').replace('.json', '')
            # Validate it's a proper date
            datetime.strptime(date_str, '%Y-%m-%d')
            dates.add(date_str)
        except ValueError:
            continue
            
    # If no dates found and old state file exists, add its date
    if not dates and os.path.exists(OLD_STATE_FILE):
        try:
            with open(OLD_STATE_FILE, 'r') as f:
                old_state = json.load(f)
                dates.add(old_state.get('date', str(date.today())))
        except:
            pass
            
    # Always include today's date
    dates.add(str(date.today()))
    
    return sorted(list(dates), reverse=True)

def load_state(selected_date=None):
    ensure_data_dir()
    
    if selected_date is None:
        selected_date = str(date.today())
        
    state_file = get_state_file(selected_date)
    
    # Try to load from new location first
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
    # Fall back to old state file if it exists and date matches
    elif os.path.exists(OLD_STATE_FILE):
        with open(OLD_STATE_FILE, 'r') as f:
            old_state = json.load(f)
            if old_state.get('date') == selected_date:
                state = old_state
            else:
                state = create_empty_state(selected_date)
    else:
        state = create_empty_state(selected_date)
    
    # Convert rap_lyrics to string if it's in old format
    if isinstance(state.get('rap_lyrics', None), dict):
        state['rap_lyrics'] = state['rap_lyrics'].get('lyrics', '')
    
    return state

def create_empty_state(selected_date):
    """Create a new empty state for the given date"""
    return {
        'current_step': 0,
        'date': selected_date,
        'articles': [],
        'categories': {},
        'filtered_articles': [],
        'news_summaries': [],
        'rap_lyrics': '',
        'word_timestamps': None,
        'aligned_timestamps': None,
        'video_path': None,
        'youtube_url': None
    }

def save_state(state, selected_date=None):
    ensure_data_dir()
    if selected_date is None:
        selected_date = state['date']
    
    state_file = get_state_file(selected_date)
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def save_step_output(step_name, data, selected_date=None):
    """Save step output to a JSON file"""
    ensure_data_dir()
    output_dir = get_output_dir(selected_date)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"step_{step_name}_{selected_date or date.today()}.json")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: x.__dict__)

def check_youtube_credentials():
    """Check if YouTube credentials are properly set up and try to refresh if needed"""
    creds = None
    token_path = 'token.pickle'
    secrets_path = 'client_secrets.json'
    
    # First check if client_secrets.json exists
    if not os.path.exists(secrets_path):
        return False, "client_secrets.json not found in the project directory"
    
    # Check for existing token
    if os.path.exists(token_path):
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            return False, f"Error loading token.pickle: {str(e)}"
    
    # If no credentials or if expired
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed credentials
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                return True, "Credentials refreshed successfully"
            except Exception as e:
                return False, f"Error refreshing credentials: {str(e)}"
        else:
            try:
                # Load client secrets and create new credentials
                flow = InstalledAppFlow.from_client_secrets_file(
                    secrets_path,
                    ['https://www.googleapis.com/auth/youtube.upload']
                )
                creds = flow.run_local_server(port=0)
                # Save new credentials
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                return True, "New credentials created successfully"
            except Exception as e:
                return False, f"Error creating new credentials: {str(e)}"
    
    return True, "Credentials are valid"

def check_youtube_quota():
    """Check current quota usage and limits"""
    try:
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            return False, "Invalid credentials"
        
        youtube = build('youtube', 'v3', credentials=creds)
        
        # Get quota information
        channels_response = youtube.channels().list(
            part='snippet',
            mine=True
        ).execute()
        
        # Get quota usage from response headers
        quota_remaining = channels_response.get('quotaUsed', 0)
        
        return True, f"Quota used today: {quota_remaining}"
        
    except Exception as e:
        if 'quotaExceeded' in str(e):
            return False, "Daily quota exceeded. Please request an increase at: https://console.cloud.google.com/"
        return False, f"Error checking quota: {str(e)}"

def clean_url(url):
    """Remove tracking parameters and fragments from URLs, and follow redirects for Google News"""
    # Handle Google News URLs
    if 'news.google.com' in url:
        try:
            # Make a request to follow the redirect
            response = requests.get(url, allow_redirects=True, timeout=10)
            # Get the final URL after redirects
            url = response.url
            print("Followed Google News redirect:", url)
        except Exception as e:
            print(f"Failed to follow Google News redirect: {e}")
            # If redirect fails, keep original URL
            pass
    
    # Remove everything after ? or #
    base_url = url.split('?')[0].split('#')[0]
    # Remove trailing slashes
    return base_url.rstrip('/')

def load_config():
    """Load URLs from config file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: config.json not found, using default configuration")
        return {
            "rss_feeds": [],
            "reddit_feeds": []
        }
    except json.JSONDecodeError:
        print("Warning: config.json is invalid, using default configuration")
        return {
            "rss_feeds": [],
            "reddit_feeds": []
        }

@app.route('/')
def index():
    selected_date = request.args.get('date', str(date.today()))
    available_dates = get_available_dates()
    if not available_dates:
        available_dates = [str(date.today())]
    if selected_date not in available_dates:
        selected_date = available_dates[0]
    
    state = load_state(selected_date)
    
    # Check YouTube credentials status
    youtube_creds_valid, youtube_creds_message = check_youtube_credentials()
    
    return render_template('index.html', 
                         state=state, 
                         available_dates=available_dates,
                         selected_date=selected_date,
                         youtube_creds_valid=youtube_creds_valid,
                         youtube_creds_message=youtube_creds_message)

@app.route('/step/<int:step>', methods=['POST'])
def execute_step(step):
    selected_date = request.args.get('date', str(date.today()))
    state = load_state(selected_date)
    
    try:
        if step == 1:  # Gather articles
            config = load_config()
            rss_urls = config['rss_feeds']
            reddit_urls = config['reddit_feeds']
            
            articles = []
            
            # Gather RSS articles and clean URLs
            rss_articles = gather_rss_articles(rss_urls)
            for article in rss_articles:
                article['url'] = clean_url(article['url'])
            articles.extend(rss_articles)
            
            # Gather Reddit articles and clean URLs
            reddit_articles = gather_reddit_articles(reddit_urls)
            for article in reddit_articles:
                article['url'] = clean_url(article['url'])
            articles.extend(reddit_articles)
            
            state['articles'] = articles
            save_step_output('gather_articles', state['articles'])
            
        elif step == 2:  # Categorize articles
            state['categories'] = categorise_articles(state['articles'])
            save_step_output('categories', state['categories'])
            
        elif step == 3:  # Filter articles
            filtered_articles = []
            # Only filter categories that have articles
            if state['categories'].get('global', []):
                filtered_articles.extend(filter_top_articles('global', state['categories']['global'], 4))
            if state['categories'].get('local', []):
                filtered_articles.extend(filter_top_articles('local', state['categories']['local'], 2))
            if state['categories'].get('stem', []):
                filtered_articles.extend(filter_top_articles('stem', state['categories']['stem'], 1))
            if state['categories'].get('random', []):
                filtered_articles.extend(filter_top_articles('random', state['categories']['random'], 1))
            
            state['filtered_articles'] = filtered_articles
            save_step_output('filtered_articles', filtered_articles)
            
        elif step == 4:  # Scrape and summarize
            news_summaries = []
            for article in state['filtered_articles']:
                scraped_data = scrape_article(article['url'])
                if scraped_data:
                    news_summaries.append({
                        "url": article['url'], 
                        "title": article['title'],
                        "image_url": scraped_data['image_url'],
                        "content": scraped_data['content'],
                        "news_source": scraped_data['news_source']
                    })
                else:
                    news_summaries.append({
                        "url": article['url'], 
                        "title": article['title'],
                        "image_url": "TODO",
                        "content": "TODO",
                        "news_source": "TODO"
                    })
            
            state['news_summaries'] = news_summaries
            save_step_output('news_summaries', news_summaries)
            
        elif step == 5:  # Generate rap lyrics
            rap_lyrics = generate_rap_lyrics(state['news_summaries'])
            
            state['rap_lyrics'] = rap_lyrics
            save_step_output('rap_lyrics', rap_lyrics)
            
        elif step == 6:  # Process audio
            audio_file = f"./audio/Rap Up News {selected_date}.mp3"
            print("Looking for audio file:", audio_file)
            if os.path.exists(audio_file):
                word_timestamps = transcribe_audio_with_timestamps(audio_file)
                # Get rap_lyrics directly as it's now stored as a string
                lyrics = state.get('rap_lyrics', '')
                if not lyrics:
                    raise ValueError("No rap lyrics found. Please generate lyrics first.")
                    
                aligned_timestamps = align_lyrics_to_transcript(lyrics, word_timestamps)
                
                state['word_timestamps'] = word_timestamps
                state['aligned_timestamps'] = aligned_timestamps
                save_step_output('audio_processing', {
                    'word_timestamps': word_timestamps,
                    'aligned_timestamps': aligned_timestamps
                })
            else:
                return jsonify({'error': 'Audio file not found'}), 400
            
        elif step == 7:  # Generate video
            audio_file = f"./audio/Rap Up News {selected_date}.mp3"
            print("Looking for audio file:", audio_file)
            if os.path.exists(audio_file):
                download_images(state['news_summaries'], selected_date)
                video_path = generate_video(state['news_summaries'], audio_file, state['aligned_timestamps'], debug=False)
                state['video_path'] = video_path
                save_step_output('video', {'video_path': video_path})
            else:
                return jsonify({'error': 'Audio file not found'}), 400
        
        elif step == 8:  # Upload to YouTube
            video_path = state.get('video_path')
            if not video_path or not os.path.exists(video_path):
                return jsonify({
                    'error': 'Video file not found',
                    'details': 'Please generate video first.'
                }), 400

            # Generate thumbnail
            try:
                # Format date (e.g., "JAN 15, 2024")
                date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
                formatted_date = date_obj.strftime("%b %d, %Y").upper()  # %b gives abbreviated month name
                
                # Open template and create draw object
                img = Image.open("video/thumbnail-template.png")
                # Convert RGBA to RGB
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else img)
                    img = background
                
                # Create a transparent layer for the text
                txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(txt_layer)
                
                # Load Aileron font
                try:
                    font = ImageFont.truetype("video/fonts/aileron/Aileron-Black.otf", 80)
                except:
                    # Fallback to default font if Aileron not found
                    print("Aileron font not found, using default")
                    font = ImageFont.load_default()
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Get text size
                text_bbox = draw.textbbox((0, 0), formatted_date, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Calculate position (bottom right with padding)
                padding = 200
                x = img_width - text_width - padding
                y = img_height - text_height - padding
                
                # Add text to transparent layer
                draw.text(
                    (x, y),
                    formatted_date,
                    fill='yellow',
                    font=font
                )
                
                # Rotate the text layer
                txt_layer = txt_layer.rotate(-3, expand=True, resample=Image.Resampling.BICUBIC)
                
                # Calculate new position after rotation
                new_x = img_width - txt_layer.width - padding + 320  # Adjust x position to compensate for rotation
                new_y = img_height - txt_layer.height - padding + 270  # Adjust y position to compensate for rotation
                
                # Paste rotated text onto main image
                img.paste(txt_layer, (new_x, new_y), txt_layer)
                
                # Save thumbnail
                img.save(f"video/thumbnail_{selected_date}.jpg", "JPEG", quality=95)
                print(f"Thumbnail generated: video/thumbnail_{selected_date}.jpg")
                
            except Exception as e:
                print(f"Error generating thumbnail: {e}")
                thumbnail_path = None

            # Create title and description
            title = f"Daily News in Rap - Headlines for {selected_date}"
            
            description = "🎤 Welcome to Rap Up News!\n"
            description += "Today's headlines meet hard-hitting rhymes as we break down the biggest stories in a way you've never heard before.\n\n"
            description += "📰 TODAY'S HEADLINES:\n"
            
            # Add article titles and URLs from news summaries
            for idx, article in enumerate(state['news_summaries'], 1):
                description += f"{idx}. {article['title']}\n"
                description += f"   Source: {article['url']}\n\n"
            
            description += "\n🔔 Don't forget to Subscribe for your daily dose of Rap News!"

            # Just print and return the title and description
            print("Title:", title)
            print("\nDescription:", description)
            
            # return jsonify({
            #     'success': True,
            #     'message': 'Title and description generated',
            #     'title': title,
            #     'description': description
            # })

            # Check YouTube credentials first
            creds_valid, creds_message = check_youtube_credentials()
            if not creds_valid:
                return jsonify({
                    'error': 'YouTube credentials not properly set up',
                    'details': creds_message
                }), 400
            
            # Create thumbnail path
            thumbnail_path = f"video/thumbnail_{selected_date}.jpg"
            
            try:
                video_url = upload_to_youtube(video_path, thumbnail_path, title, description)
                if not video_url:
                    raise Exception("Upload failed - no video URL returned")
                    
                state['youtube_url'] = video_url
                save_step_output('youtube', {'youtube_url': video_url})
                
            except Exception as e:
                import traceback
                error_details = {
                    'error': 'Failed to upload to YouTube',
                    'details': str(e),
                    'traceback': traceback.format_exc()
                }
                return jsonify(error_details), 500
        
        state['current_step'] = step
        save_state(state, selected_date)
        
        return jsonify({'success': True, 'message': f'Step {step} completed successfully'})
        
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'step': step
        }
        return jsonify(error_details), 500

@app.route('/get_step_output/<int:step>')
def get_step_output(step):
    selected_date = request.args.get('date', str(date.today()))
    state = load_state(selected_date)
    output = {}
    
    if step == 1:
        output = {'articles': state.get('articles', [])}
    elif step == 2:
        output = {'categories': state.get('categories', {})}
    elif step == 3:
        output = {'filtered_articles': state.get('filtered_articles', [])}
    elif step == 4:
        output = {'news_summaries': state.get('news_summaries', [])}
    elif step == 5:
        return jsonify(state.get('rap_lyrics', ''))
    elif step == 6:
        output = {
            'word_timestamps': state.get('word_timestamps', []),
            'aligned_timestamps': state.get('aligned_timestamps', [])
        }
    elif step == 7:
        output = {'video_path': state.get('video_path')}
    elif step == 8:
        # Calculate scheduled publish time
        current_date = datetime.now()
        next_day = current_date + timedelta(days=1)
        publish_date = next_day.replace(hour=0, minute=30, second=0, microsecond=0)
        
        output = {
            'youtube_url': state.get('youtube_url'),
            'scheduled_publish_time': publish_date.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return jsonify(output)

@app.route('/update_step_output/<int:step>', methods=['POST'])
def update_step_output(step):
    selected_date = request.args.get('date', str(date.today()))
    state = load_state(selected_date)
    data = request.json
    
    try:
        if step == 1:
            state['articles'] = data['articles']
        elif step == 2:
            state['categories'] = data['categories']
        elif step == 3:
            # Update filtered articles and ensure they have all necessary fields
            filtered_articles = data['filtered_articles']
            # Make sure we preserve any existing fields that might be needed
            state['filtered_articles'] = []
            for article in filtered_articles:
                state['filtered_articles'].append({
                    'category': article['category'],
                    'url': article['url'],
                    'title': article['title']
                })
        elif step == 4:
            state['news_summaries'] = data['news_summaries']
        elif step == 5:
            state['rap_lyrics'] = data
        elif step == 6:
            state['word_timestamps'] = data['word_timestamps']
            state['aligned_timestamps'] = data['aligned_timestamps']
        elif step == 7:
            state['video_path'] = data['video_path']
        elif step == 8:
            state['youtube_url'] = ''
        
        save_state(state, selected_date)
        save_step_output(f'step_{step}', data, selected_date)
        return jsonify({'success': True, 'message': 'Content updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_pipeline():
    try:
        selected_date = request.args.get('date', str(date.today()))
        new_state = create_empty_state(selected_date)
        save_state(new_state, selected_date)
        
        # Clear the outputs directory for this date
        output_dir = get_output_dir(selected_date)
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')
        
        return jsonify({'success': True, 'message': 'Pipeline reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_to_step', methods=['POST'])
def reset_to_step():
    try:
        step = request.json.get('step', 0)
        selected_date = request.args.get('date', str(date.today()))
        state = load_state(selected_date)
        
        # Keep data up to the specified step
        new_state = {
            'current_step': step,
            'date': state['date'],
            'articles': state['articles'] if step >= 1 else [],
            'categories': state['categories'] if step >= 2 else {},
            'filtered_articles': state['filtered_articles'] if step >= 3 else [],
            'news_summaries': state['news_summaries'] if step >= 4 else [],
            'rap_lyrics': state['rap_lyrics'] if step >= 5 else '',
            'word_timestamps': state['word_timestamps'] if step >= 6 else None,
            'aligned_timestamps': state['aligned_timestamps'] if step >= 6 else None,
            'video_path': state['video_path'] if step >= 7 else None,
            'youtube_url': state['youtube_url'] if step >= 8 else None
        }
        
        # Save the modified state
        save_state(new_state, selected_date)
        
        # Clear output files for steps after the reset point
        output_dir = get_output_dir(selected_date)
        if os.path.exists(output_dir):
            # Map step numbers to their names
            step_names = {
                1: 'gather_articles',
                2: 'categories',
                3: 'filtered_articles',
                4: 'news_summaries',
                5: 'rap_lyrics',
                6: 'audio_processing',
                7: 'video',
                8: 'youtube'
            }
            
            for file in os.listdir(output_dir):
                try:
                    # Check each step name
                    for step_num, step_name in step_names.items():
                        if file.startswith(f'step_{step_name}_') and step_num > step:
                            file_path = os.path.join(output_dir, file)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                break
                except Exception as e:
                    print(f'Error processing file {file}: {str(e)}')
                    continue
        
        return jsonify({'success': True, 'message': f'Pipeline reset to step {step}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    # Assuming videos are stored in a 'videos' directory
    return send_from_directory('video', filename)

@app.route('/debug/youtube_credentials')
def debug_youtube_credentials():
    status = {
        'client_secrets_exists': os.path.exists('client_secrets.json'),
        'token_pickle_exists': os.path.exists('token.pickle'),
    }
    
    try:
        valid, message = check_youtube_credentials()
        status.update({
            'credentials_valid': valid,
            'message': message
        })
        
        # Add more detailed information if client_secrets.json exists
        if status['client_secrets_exists']:
            with open('client_secrets.json', 'r') as f:
                secrets = json.load(f)
                status['client_secrets_format_valid'] = 'installed' in secrets
                status['has_client_id'] = bool(secrets.get('installed', {}).get('client_id'))
    except Exception as e:
        status['error'] = str(e)
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)
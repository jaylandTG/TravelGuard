import os
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
import jwt
from functools import wraps
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import threading

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Gemini API
client = genai.Client(api_key=os.getenv('GEMINI_TOKEN'))
google_search_tool = Tool(google_search=GoogleSearch())

# JWT Configuration
JWT_SECRET = os.getenv('JWT_SECRET')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXP_DELTA_SECONDS = int(os.getenv('JWT_EXPIRATION_HOURS', 24)) * 3600

# AI Chat Configurations 
chat_sessions = {}
chat_sessions_lock = threading.Lock()

# Initialize Firebase Admin SDK
cred = credentials.Certificate({
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
})
firebase_admin.initialize_app(cred)

# ThreadPool for non-blocking Firebase calls
executor = ThreadPoolExecutor(max_workers=10)

# Frontend Firebase config
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "")
}

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('jwt_token') or request.headers.get('Authorization')
        if not token:
            response = redirect(url_for('home'))
            response.delete_cookie('jwt_token')
            return response
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            current_user = data['user_id']
        except Exception:
            response = redirect(url_for('home'))
            response.delete_cookie('jwt_token')
            return response
        return f(current_user, *args, **kwargs)
    return decorated

# Background function to fetch user info
def fetch_user_info(uid):
    user_record = auth.get_user(uid)
    return {
        'id': user_record.uid,
        'email': user_record.email,
        'name': user_record.display_name or user_record.email.split('@')[0]
    }

# Background function to verify token
def verify_google_token(id_token):
    return auth.verify_id_token(id_token)

### PAGE ROUTES

@app.route('/')
def home():
    token = request.cookies.get('jwt_token')
    if token:
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return redirect(url_for('dashboard'))
        except:
            pass
    return render_template('login.html', firebase_config=FIREBASE_CONFIG)

@app.route('/dashboard')
@token_required
def dashboard(current_user):
    try:
        future = executor.submit(fetch_user_info, current_user)
        user_data = future.result(timeout=5)
        return render_template('dashboard.html', user=user_data)
    except Exception:
        response = redirect(url_for('home'))
        response.delete_cookie('jwt_token')
        return response

### UTILITY ROUTES

@app.route('/logout')
def logout():
    response = redirect('/')
    response.delete_cookie('jwt_token')
    return response

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    id_token = request.json.get('id_token')
    try:
        future = executor.submit(verify_google_token, id_token)
        decoded_token = future.result(timeout=5)

        payload = {
            'user_id': decoded_token['uid'],
            'email': decoded_token['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
        }
        jwt_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        response = make_response(jsonify({
            'user': {
                'id': decoded_token['uid'],
                'email': decoded_token['email'],
                'name': decoded_token.get('name', '')
            }
        }), 200)

        response.set_cookie(
            'jwt_token',
            jwt_token,
            httponly=True,
            secure=True if os.getenv('FLASK_ENV') == 'production' else False,
            samesite='Lax',
            max_age=JWT_EXP_DELTA_SECONDS
        )
        return response

    except TimeoutError:
        return jsonify({'error': 'Authentication timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 401

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    data = request.json
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    user_id = current_user  # current_user is already the ID string
    print(f"Processing chat for user ID: {user_id}")

    # Get or create chat session
    with chat_sessions_lock:
        if user_id not in chat_sessions:
            chat_sessions[user_id] = {'history': []}
        
        session = chat_sessions[user_id]
    
    try:
        # Get user email for the username
        user_record = auth.get_user(user_id)
        email = user_record.email
        username = email.split('@')[0]
        
        # Rest of your existing chat implementation...
        contents = []
        
        if not session['history']:
            system_prompt = f"You are VoyagerAI an AI Assistant of the company Travel Guard and you will address me as {username}. You are a friendly and knowledgeable travel companion. You Respond very concise, no filler words, no referencing other company names in your chat such as waze, google or other local news."
            contents.append({
                'role': 'user',
                'parts': [{'text': system_prompt}]
            })
        
        for msg in session['history']:
            contents.append({
                'role': msg['role'],
                'parts': [{'text': msg['content']}]
            })
        
        contents.append({
            'role': 'user',
            'parts': [{'text': user_message}]
        })

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=['TEXT'],
                system_instruction=system_prompt if not session['history'] else None
            )
        )

        session['history'].append({'role': 'user', 'content': user_message})
        session['history'].append({'role': 'model', 'content': response.text})
        session['history'] = session['history'][-20:]

        return jsonify({
            'response': response.text,
            'grounding_metadata': getattr(response, 'grounding_metadata', None)
        })
    
    except TimeoutError:
        return jsonify({'error': 'Response timed out'}), 504
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/protected')
@token_required
def protected_route(current_user):
    return jsonify({'message': f'Hello {current_user}, this is a protected route!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('FLASK_ENV') == 'development', threaded=True)

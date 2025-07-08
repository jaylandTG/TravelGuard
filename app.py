# ==============================================================================
# I. IMPORTS
# ==============================================================================

# --- Standard Library Imports ---
import os
import threading
import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# --- Third-Party Imports ---
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
from flask_caching import Cache
from flask_compress import Compress
import jwt
from dotenv import load_dotenv

# Firebase & Google Cloud
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.cloud.firestore_v1.client import Client
from google.auth.credentials import Credentials

# Google AI & Twilio
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from twilio.rest import Client as TwilioClient

# ==============================================================================
# II. CONFIGURATION
# ==============================================================================

# --- Environment Variables ---
load_dotenv()

# --- Flask App Configuration ---
APP_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
FLASK_ENV = os.getenv('FLASK_ENV', 'production')

# --- JWT Configuration ---
JWT_SECRET = os.getenv('JWT_SECRET')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXP_DELTA_SECONDS = int(os.getenv('JWT_EXPIRATION_HOURS', 24)) * 3600

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_TOKEN')
SYSTEM_PROMPT_TEMPLATE = (
    "You are VoyagerAI an AI Assistant of the company Travel Guard and you will address me as {}. "
    "You are a friendly and knowledgeable travel companion. You Respond very concise, no filler words, "
    "no referencing other company names in your chat such as waze, google or other local news. "
    "You are limited to answer about traveling inquiries or emergency situations."
)

GOOGLE_SEARCH_TOOL = Tool(google_search=GoogleSearch())
GEMINI_CONFIG = GenerateContentConfig(
    tools=[GOOGLE_SEARCH_TOOL],
    response_modalities=['TEXT']
)

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# --- Firebase Admin Configuration ---
FIREBASE_CREDENTIALS = credentials.Certificate({
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

# --- Frontend Firebase Configuration ---
FIREBASE_CONFIG_FRONTEND = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "")
}

# --- Application Globals & Constants ---
MAX_CHAT_HISTORY = 20
EXECUTOR_MAX_WORKERS = 10
CACHE_TIMEOUT_SECONDS = 300

# ==============================================================================
# III. INITIALIZATION
# ==============================================================================

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates')
app.secret_key = APP_SECRET_KEY
Compress(app)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# --- Service Clients Initialization ---
firebase_admin.initialize_app(FIREBASE_CREDENTIALS)
db: Client = firestore.client()
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
TWILIO = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Firestore Collection References ---
USER_COLLECTION = db.collection('users')
LOCATION_SHARING_COLLECTION = db.collection('location_sharing_sessions')

# --- Thread Pool & Chat Session Management ---
executor = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)
chat_sessions = {}
chat_sessions_lock = threading.Lock()

# ==============================================================================
# IV. DECORATORS & HELPERS
# ==============================================================================

def token_required(f):
    """Decorator to protect routes that require JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('jwt_token') or request.headers.get('Authorization')
        if not token:
            return redirect(url_for('home'))
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={'verify_exp': True})
            return f(data['user_id'], *args, **kwargs)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            response = redirect(url_for('home'))
            response.delete_cookie('jwt_token')
            return response
    return decorated

@cache.memoize(timeout=CACHE_TIMEOUT_SECONDS)
def fetch_user_info(uid: str) -> dict:
    """Fetches and caches user information from Firebase Auth."""
    user_record = auth.get_user(uid)
    return {
        'id': user_record.uid,
        'email': user_record.email,
        'name': user_record.display_name or user_record.email.split('@')[0]
    }

def send_travel_notification(user_name: str, emergency_contacts: list, puv_details: dict, destination: dict):
    """Sends SMS notifications to emergency contacts in the background."""
    for contact in emergency_contacts:
        message = (
            f"🚨 Travel Update from Travel Guard 🚨\n\n"
            f"Your {contact.get('relationship', 'contact')} {user_name} is en route to {destination['address']}.\n\n"
            f"🚌 PUV Info:\n"
            f"Plate: {puv_details.get('plate_number', 'N/A')}\n"
            f"Type: {puv_details.get('type', 'N/A')}\n"
            f"Departed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Note: {puv_details.get('notes', 'None')}\n\n"
            f"Travel Guard is monitoring this journey. #TravelSafeWithTravelGuard"
        )
        try:
            # Uncomment the following lines to enable live SMS sending
            # TWILIO.messages.create(
            #     to=contact.get('phone'),
            #     from_=TWILIO_FROM_NUMBER,
            #     body=message
            # )
            app.logger.info(f"SMS notification would be sent to {contact.get('phone')}")
        except Exception as e:
            app.logger.error(f"Failed to send SMS to {contact.get('phone')}: {e}", exc_info=True)


# ==============================================================================
# V. ROUTE DEFINITIONS
# ==============================================================================

# --- Page Rendering Routes ---

@app.route('/')
def home():
    """Renders the login page or redirects to the dashboard if already logged in."""
    token = request.cookies.get('jwt_token')
    if token:
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={'verify_signature': False})
            return redirect(url_for('dashboard'))
        except jwt.PyJWTError:
            pass  # Invalid or expired token, show login page
    return render_template('login.html', firebase_config=FIREBASE_CONFIG_FRONTEND)

@app.route('/dashboard')
@token_required
def dashboard(current_user: str):
    """Renders the main application dashboard."""
    user_ref = USER_COLLECTION.document(current_user)
    user_doc = user_ref.get()

    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_data['is_new_user'] = not user_data.get('profile_complete', False)
    else:
        user_data = fetch_user_info(current_user)
        user_data['is_new_user'] = True

    return render_template('dashboard.html', user=user_data)

# --- Authentication & User Management API Routes ---

@app.route('/logout')
def logout():
    """Logs the user out by clearing the JWT cookie."""
    response = redirect(url_for('home'))
    response.delete_cookie('jwt_token')
    return response

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Authenticates a user via Google ID token and issues a JWT."""
    id_token = request.json.get('id_token')
    if not id_token:
        return jsonify({'error': 'Missing token'}), 400

    try:
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        user_ref = USER_COLLECTION.document(user_id)
        user_doc = user_ref.get()
        is_new_user = not user_doc.exists

        user_data = {
            'email': decoded_token['email'],
            'name': decoded_token.get('name', ''),
            'last_login': firestore.SERVER_TIMESTAMP,
            'profile_complete': not is_new_user
        }
        user_ref.set(user_data, merge=True)

        if is_new_user:
            executor.submit(user_ref.collection('emergency_contacts').document('default').set({'placeholder': True}))
            executor.submit(user_ref.collection('travel_history').document('default').set({'placeholder': True}))

        payload = {
            'user_id': user_id,
            'email': decoded_token['email'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
        }
        jwt_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        response_data = {
            'token': jwt_token,
            'is_new_user': is_new_user,
            'user': {'id': user_id, 'email': decoded_token['email']}
        }
        response = jsonify(response_data)
        response.set_cookie(
            'jwt_token', jwt_token,
            httponly=True, secure=(FLASK_ENV == 'production'),
            samesite='Lax', max_age=JWT_EXP_DELTA_SECONDS
        )
        return response

    except Exception as e:
        app.logger.error(f"Google auth error: {e}", exc_info=True)
        return jsonify({'error': 'Authentication failed'}), 401

@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile(current_user: str):
    """Retrieves the complete user profile."""
    try:
        user_doc = USER_COLLECTION.document(current_user).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        return jsonify(user_doc.to_dict())
    except Exception as e:
        app.logger.error(f"Profile fetch error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/complete-profile', methods=['POST'])
@token_required
def complete_profile(current_user: str):
    """Completes a new user's profile with additional details."""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    try:
        update_data = {
            'phone': data.get('phone'),
            'profile_complete': True,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        user_ref = USER_COLLECTION.document(current_user)
        executor.submit(user_ref.update, update_data)

        if contact := data.get('contact'):
            contact_data = {
                'name': contact.get('name'),
                'phone': contact.get('phone'),
                'relationship': contact.get('relationship', ''),
                'created_at': firestore.SERVER_TIMESTAMP
            }
            executor.submit(user_ref.collection('emergency_contacts').add, contact_data)

        return jsonify({"status": "success", "message": "Profile update initiated"})
    except Exception as e:
        app.logger.error(f"Profile completion error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error"}), 500

# --- Chatbot API Routes ---

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    if not (data := request.json) or not (user_message := data.get('message')):
        return jsonify({'error': 'No message provided'}), 400

    with chat_sessions_lock:
        session = chat_sessions.setdefault(current_user, {'history': []})
    
    try:
        user_info = fetch_user_info(current_user)
        username = user_info['name']
        
        # Build messages efficiently
        contents = []
        if not session['history']:
            contents.append({
                'role': 'user',
                'parts': [{'text': SYSTEM_PROMPT_TEMPLATE.format(username)}]
            })
        
        contents.extend({
            'role': msg['role'],
            'parts': [{'text': msg['content']}]
        } for msg in session['history'])
        
        contents.append({
            'role': 'user',
            'parts': [{'text': user_message}]
        })

        config = GenerateContentConfig(
            tools=GEMINI_CONFIG.tools,
            response_modalities=GEMINI_CONFIG.response_modalities
        )

        if not session['history']:
            config.system_instruction = SYSTEM_PROMPT_TEMPLATE.format(username)

        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=config
        )

        # Update session more efficiently
        session['history'].extend([
            {'role': 'user', 'content': user_message},
            {'role': 'model', 'content': response.text}
        ])
        session['history'] = session['history'][-20:]

        # Firestore write in background
        executor.submit(
            USER_COLLECTION.document(current_user)
                .collection('chat_history').add({
                    'user_message': user_message,
                    'ai_response': response.text,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
        )

        return jsonify({
            'response': response.text,
            'grounding_metadata': getattr(response, 'grounding_metadata', None)
        })
    
    except TimeoutError:
        return jsonify({'error': 'Response timed out'}), 504
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500



# --- Travel History API Routes ---

@app.route('/api/travel-history', methods=['GET', 'POST'])
@token_required
def travel_history_manager(current_user: str):
    """Manages travel history entries (GET all, POST new)."""
    try:
        travel_history_ref = USER_COLLECTION.document(current_user).collection('travel_history')
        if request.method == 'GET':
            query = travel_history_ref.order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            history = [{'id': doc.id, **doc.to_dict()} for doc in query if doc.id != 'default']
            return jsonify(history)

        if request.method == 'POST':
            data = request.json
            if not data or not data.get('destination'):
                return jsonify({"error": "Missing destination"}), 400

            entry = {
                'destination': data.get('destination'),
                'start_date': data.get('start_date'),
                'end_date': data.get('end_date'),
                'notes': data.get('notes', ''),
                'created_at': firestore.SERVER_TIMESTAMP
            }
            new_doc_ref = travel_history_ref.document()
            executor.submit(new_doc_ref.set, entry)
            return jsonify({"status": "success", "id": new_doc_ref.id}), 201

    except Exception as e:
        app.logger.error(f"Travel history error on {request.method}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Location Sharing API Routes ---

@app.route('/api/location-sharing/start', methods=['POST'])
@token_required
def start_location_sharing(current_user: str):
    """Starts a new real-time location sharing session."""
    data = request.json
    if not all(k in data for k in ['puv_details', 'emergency_contacts', 'destination']):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        session_data = {
            'user_id': current_user,
            'puv_details': data['puv_details'],
            'emergency_contacts': data['emergency_contacts'],
            'destination': data['destination'],
            'start_time': firestore.SERVER_TIMESTAMP,
            'status': 'active'
        }
        session_ref = LOCATION_SHARING_COLLECTION.document()
        session_ref.set(session_data)

        user_info = fetch_user_info(current_user)
        executor.submit(
            send_travel_notification,
            user_info['name'], data['emergency_contacts'],
            data['puv_details'], data['destination']
        )
        return jsonify({"status": "success", "session_id": session_ref.id}), 201

    except Exception as e:
        app.logger.error(f"Location sharing start error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/location-sharing/update', methods=['POST'])
@token_required
def update_location(current_user: str):
    """Updates the user's location for an active session."""
    data = request.json
    location = data.get('location')
    session_id = data.get('session_id')
    if not session_id or not location or 'lat' not in location or 'lng' not in location:
        return jsonify({"error": "Invalid payload"}), 400

    try:
        update_data = {
            'current_location': firestore.GeoPoint(location['lat'], location['lng']),
            'last_updated': firestore.SERVER_TIMESTAMP
        }
        LOCATION_SHARING_COLLECTION.document(session_id).update(update_data)
        return jsonify({"status": "success"})
    except Exception as e:
        app.logger.error(f"Location update error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/location-sharing/end', methods=['POST'])
@token_required
def end_location_sharing(current_user: str):
    """Marks a location sharing session as completed."""
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
    try:
        update_data = {'status': 'completed', 'end_time': firestore.SERVER_TIMESTAMP}
        LOCATION_SHARING_COLLECTION.document(session_id).update(update_data)
        return jsonify({"status": "success"})
    except Exception as e:
        app.logger.error(f"Location sharing end error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Emergency Contacts API Route ---

@app.route('/api/emergency-contacts', methods=['GET'])
@token_required
def get_emergency_contacts(current_user: str):
    """Retrieves a user's list of emergency contacts."""
    try:
        contacts_ref = USER_COLLECTION.document(current_user).collection('emergency_contacts')
        query = contacts_ref.stream()
        contacts = [{'id': doc.id, **doc.to_dict()} for doc in query if doc.id != 'default']
        return jsonify(contacts)
    except Exception as e:
        app.logger.error(f"Emergency contacts fetch error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Configuration & Test Routes ---

@app.route('/api/maps-config')
@token_required
def get_maps_config(current_user: str):
    """Provides the Google Maps API key to the frontend."""
    return jsonify({
        'key': os.getenv('GOOGLE_MAPS_API_KEY'),
        'libraries': 'places,geocoding,geometry'
    })

@app.route('/api/protected')
@token_required
def protected_route(current_user: str):
    """A sample protected route for testing authentication."""
    user_info = fetch_user_info(current_user)
    return jsonify({'message': f"Hello {user_info.get('name')}, you are authenticated!"})

# ==============================================================================
# VI. APPLICATION RUNNER
# ==============================================================================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=(FLASK_ENV == 'development'),
        threaded=True
    )
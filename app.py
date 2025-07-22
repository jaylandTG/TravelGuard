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
import uuid

# Firebase & Google Cloud
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin import db as realtime_db
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
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", ""),
    'databaseURL': os.getenv("FIREBASE_REALTIME_DB").rstrip('/')
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
firebase_admin.initialize_app(
    FIREBASE_CREDENTIALS,
    {
        'databaseURL': os.getenv("FIREBASE_REALTIME_DB").rstrip('/')
    }
)
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
def get_active_trip_session_id(user_id: str) -> str | None:
    """Checks Firestore for an active trip session for a user."""
    try:
        sessions_ref = LOCATION_SHARING_COLLECTION.where('user_id', '==', user_id).where('status', 'in', ['active', 'sos']).limit(1)
        active_sessions = list(sessions_ref.stream())
        if active_sessions:
            return active_sessions[0].id
    except Exception as e:
        app.logger.error(f"Failed to check for active trip for {user_id}: {e}", exc_info=True)
    return None

def token_required(f):
    """Decorator to protect routes that require JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('jwt_token') or request.headers.get('Authorization')
        if not token:
            return redirect(url_for('home'))
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={'verify_exp': True})
            current_user_id = data['user_id']
            
            # If user has an active trip, lock them out of dashboard/logout
            active_session_id = get_active_trip_session_id(current_user_id)
            if active_session_id and request.path in [url_for('dashboard'), url_for('logout')]:
                return redirect(url_for('traveling_page', id=active_session_id))
                
            return f(current_user_id, *args, **kwargs)
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

def send_travel_notification(user_name: str, emergency_contacts: list, puv_details: dict, destination: dict, shareable_link: str):
    """Sends SMS notifications to emergency contacts in the background."""
    for contact in emergency_contacts:
        message = (
            f"🚨 Travel Update from Travel Guard 🚨\n\n"
            f"Your {contact.get('relationship', 'contact')} {user_name} has started a trip to {destination['address']}.\n\n"
            f"🚌 PUV Info:\n"
            f"Plate: {puv_details.get('plate_number', 'N/A')}\n"
            f"Type: {puv_details.get('type', 'N/A')}\n"
            f"Departed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"You can follow the trip live here: {shareable_link}\n\n"
            f"Travel Guard is monitoring this journey. #TravelSafeWithTravelGuard"
        )
        try:
            app.logger.info(f"SMS notification would be sent to {contact.get('phone')}")
            # In production, uncomment to send actual SMS
            # TWILIO.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=contact.get('phone'))
        except Exception as e:
            app.logger.error(f"Failed to send SMS to {contact.get('phone')}: {e}", exc_info=True)


def _send_sos_notifications_task(user_name: str, contacts: list, session: dict):
    """Background task to send SOS notifications via Twilio."""
    app.logger.info(f"Initiating SOS for {user_name}. Notifying {len(contacts)} contacts.")
    for contact in contacts:
        message = (
            f"🚨 EMERGENCY ALERT from Travel Guard 🚨\n\n"
            f"{user_name} has triggered an SOS alert!\n\n"
            f"Last known location: {session.get('current_location', {}).get('lat')}, "
            f"{session.get('current_location', {}).get('lng')}\n"
            f"Destination: {session.get('destination', {}).get('address')}\n"
            f"Vehicle: {session.get('puv_details', {}).get('type')} "
            f"({session.get('puv_details', {}).get('plate_number')})\n\n"
            f"View live trip: {session.get('shareable_link')}"
        )
        try:
            # In production, uncomment to send actual SMS
            # TWILIO.messages.create(
            #     to=contact.get('phone'),
            #     from_=TWILIO_FROM_NUMBER,
            #     body=message
            # )
            app.logger.info(f"SOS SMS would be sent to {contact.get('phone')}")
        except Exception as e:
            app.logger.error(f"Failed to send SOS SMS to {contact.get('phone')}: {e}", exc_info=True)


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
            # Quick check if token is decodable. Full validation is in decorator.
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={'verify_signature': False})
            
            # Check for active trip even before rendering dashboard
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False, "verify_signature": False})
            active_session_id = get_active_trip_session_id(payload['user_id'])
            if active_session_id:
                return redirect(url_for('traveling_page', id=active_session_id))

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
@token_required # This will redirect to travelling page if a trip is active
def logout(current_user: str):
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
    data = request.json
    if not all(k in data for k in ['puv_details', 'emergency_contacts', 'destination']):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        session_id = str(uuid.uuid4())
        # Use url_for for robust link generation
        shareable_link = url_for('shared_traveling_page', session_id=session_id, _external=True)
        
        session_data = {
            'user_id': current_user,
            'puv_details': data['puv_details'],
            'emergency_contacts': data['emergency_contacts'],
            'destination': data['destination'],
            'current_location': data.get('current_location'), # Include starting location
            'status': 'active',
            'start_time': {'.sv': 'timestamp'},
            'end_time': None,
            'shareable_link': shareable_link
        }
        
        # Write to Realtime Database
        ref = realtime_db.reference(f'location_sessions/{session_id}')
        ref.set(session_data)
        
        # Store persistent data in Firestore
        firestore_data = {
            'user_id': current_user,
            'session_id': session_id,
            'start_time': firestore.SERVER_TIMESTAMP,
            'status': 'active',
            'destination_address': data.get('destination', {}).get('address')
        }
        LOCATION_SHARING_COLLECTION.document(session_id).set(firestore_data)
        
        user_info = fetch_user_info(current_user)
        # Submit notification task to background
        executor.submit(
            send_travel_notification,
            user_info['name'], data['emergency_contacts'],
            data['puv_details'], data['destination'],
            shareable_link
        )
        
        return jsonify({
            "status": "success", 
            "session_id": session_id,
            "shareable_link": shareable_link
        }), 201

    except Exception as e:
        app.logger.error(f"Location sharing start error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/location-sharing/update', methods=['POST'])
@token_required
def update_location(current_user: str):
    data = request.json
    location = data.get('location')
    session_id = data.get('session_id')
    
    if not session_id or not location or 'lat' not in location or 'lng' not in location:
        return jsonify({"error": "Invalid payload"}), 400

    try:
        ref = realtime_db.reference(f'location_sessions/{session_id}/current_location')
        ref.set({
            'lat': location['lat'],
            'lng': location['lng'],
            'timestamp': {'.sv': 'timestamp'}
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        app.logger.error(f"Location update error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/location-sharing/end', methods=['POST'])
@token_required
def end_location_sharing(current_user: str):
    session_id = request.json.get('session_id')
    status = request.json.get('status', 'completed')  # 'completed' or 'cancelled'
    
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
        
    try:
        update_data = {
            'status': status,
            'end_time': {'.sv': 'timestamp'}
        }
        
        # Update Realtime Database
        ref = realtime_db.reference(f'location_sessions/{session_id}')
        # Verify user owns the session before updating
        session_data = ref.get()
        if not session_data or session_data.get('user_id') != current_user:
            return jsonify({"error": "Permission denied"}), 403

        ref.update(update_data)
        
        # Update Firestore
        LOCATION_SHARING_COLLECTION.document(session_id).update({
            'status': status,
            'end_time': firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({"status": "success"})
    except Exception as e:
        app.logger.error(f"Location sharing end error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Travelling Routes ---

@app.route('/travelling')
@token_required
def traveling_page(current_user: str):
    """Renders the active traveling page."""
    session_id = request.args.get('id')
    if not session_id:
        # If user lands here without an ID, check if they have an active session
        active_session_id = get_active_trip_session_id(current_user)
        if active_session_id:
            return redirect(url_for('traveling_page', id=active_session_id))
        return redirect('/dashboard') # No active session, send to dashboard
    
    return render_template(
        'travelling.html',
        firebase_config=FIREBASE_CONFIG_FRONTEND
    )


@app.route('/shared/<session_id>')
@app.route('/travelling/shared/<session_id>')
def shared_traveling_page(session_id: str):
    """Renders the shared view of a trip after it ends."""
    try:
        ref = realtime_db.reference(f'location_sessions/{session_id}')
        session = ref.get()

        if not session:
            return render_template(
                'shared_trip_expired.html',
                message="This trip could not be loaded."
            )

        status = session.get('status')
        user_name = session.get('user_name', 'The traveller')
        destination = session.get('destination', {}).get('address', 'your destination')
        current_loc = session.get('current_location', {})

        if status == 'completed':
            return render_template(
                'shared_trip_final.html',
                title="Arrived Safely",
                emoji="✅",
                headline=f"{user_name} has successfully arrived at {destination}.",
                subtext="Thank you for tracking the journey. All notifications have been sent."
            )

        if status == 'cancelled':
            return render_template(
                'shared_trip_final.html',
                title="Trip Cancelled",
                emoji="❌",
                headline=f"{user_name} cancelled the trip to {destination}.",
                subtext="No further location updates will be shared."
            )

        # if status == 'sos':
        #     return render_template(
        #         'shared_trip_final.html',
        #         title="SOS Alert",
        #         emoji="🚨",
        #         headline="The traveller triggered an emergency SOS.",
        #         subtext=f"Last known location: {current_loc.get('lat')}, {current_loc.get('lng')}. "
        #               "Please contact emergency contacts or the nearest authorities immediately."
        #     )

        # Still active
        return render_template(
            'travelling.html',
            session_data=session,
            firebase_config=FIREBASE_CONFIG_FRONTEND
        )

    except Exception as e:
        print(f"DEBUGGER: {e}")
        return render_template(
            'shared_trip_expired.html',
            message="This trip could not be loaded."
        )


@app.route('/api/location-sharing/sos', methods=['POST'])
@token_required
def send_sos_alert(current_user: str):
    """Sends SOS alerts to emergency contacts."""
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400
        
    try:
        ref = realtime_db.reference(f'location_sessions/{session_id}')
        session = ref.get()
        
        if not session or session.get('user_id') != current_user:
            return jsonify({"error": "Permission denied"}), 403
            
        user_info = fetch_user_info(current_user)
        contacts = session.get('emergency_contacts', [])
        
        if not contacts:
             return jsonify({"status": "success", "message": "SOS triggered, but no contacts to notify."})

        # Run notification sending in the background
        executor.submit(_send_sos_notifications_task, user_info['name'], contacts, session)
        
        # Immediately update trip status to 'sos'
        ref.child('status').set('sos')
        LOCATION_SHARING_COLLECTION.document(session_id).update({'status': 'sos'})
        
        return jsonify({"status": "success", "message": "SOS alert is being sent."})

    except Exception as e:
        app.logger.error(f"SOS alert error: {e}", exc_info=True)
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
def get_maps_config():
    """Provides the Google Maps API key to the frontend."""
    return jsonify({
        'key': os.getenv('GOOGLE_MAPS_API_KEY'),
        'libraries': 'places,marker',
        'mapId': os.getenv('MAP_ID')
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
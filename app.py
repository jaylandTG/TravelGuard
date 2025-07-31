# ==============================================================================
# I. IMPORTS
# ==============================================================================

# --- Standard Library Imports ---
import os
import threading
from datetime import datetime, timedelta, timezone
import uuid
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# --- Third-Party Imports ---
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for
from flask_caching import Cache
from flask_compress import Compress
import jwt
import bcrypt
from dotenv import load_dotenv

# --- Firebase & Google Cloud Imports ---
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin import db as realtime_db
from google.cloud.firestore_v1.client import Client
from google.auth.credentials import Credentials
from google.cloud.firestore import FieldFilter

# --- Google AI & Twilio Imports ---
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from twilio.rest import Client as TwilioClient


# ==============================================================================
# II. CONFIGURATION
# ==============================================================================

# --- Environment Variable Loading ---
# Loads environment variables from a .env file into the script's environment.
load_dotenv()

# --- Flask Application Configuration ---
APP_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
FLASK_ENV = os.getenv('FLASK_ENV', 'production')

# --- JWT (JSON Web Token) Configuration ---
# Secret key for signing JWTs, algorithm, and expiration time.
JWT_SECRET = os.getenv('JWT_SECRET')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXP_DELTA_SECONDS = int(os.getenv('JWT_EXPIRATION_HOURS', 24)) * 3600

# --- Gemini API Configuration ---
# API key for Google Gemini, system prompt template for the AI assistant,
# and configuration for content generation including tools.
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
# Account SID, Auth Token, and Twilio phone number for sending SMS messages.
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# --- Firebase Admin SDK Configuration ---
# Credentials for initializing the Firebase Admin SDK, loaded from environment variables.
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
# Configuration details for Firebase to be exposed to the client-side (frontend).
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
# Various constants used throughout the application, such as chat history limit,
# thread pool size, and cache timeout.
MAX_CHAT_HISTORY = 20
EXECUTOR_MAX_WORKERS = 10
CACHE_TIMEOUT_SECONDS = 300


# ==============================================================================
# III. INITIALIZATION
# ==============================================================================

# --- Flask App Initialization ---
# Sets up the Flask application instance, secret key, compression, and caching.
app = Flask(__name__, template_folder='templates')
app.secret_key = APP_SECRET_KEY
Compress(app)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# --- Service Clients Initialization ---
# Initializes Firebase Admin SDK, Firestore client, Gemini AI client, and Twilio client.
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
# Defines references to top-level Firestore collections for easier access.
USER_COLLECTION = db.collection('users')
LOCATION_SHARING_COLLECTION = db.collection('location_sharing_sessions')
PRESET_DESTINATIONS_COLLECTION = db.collection('preset_destinations') # This variable is not used anywhere else though, maybe remove it?

# --- Thread Pool & Chat Session Management ---
# Sets up a thread pool for background tasks and structures for managing chat sessions.
executor = ThreadPoolExecutor(max_workers=EXECUTOR_MAX_WORKERS)
chat_sessions = {}
chat_sessions_lock = threading.Lock()


# ==============================================================================
# IV. DECORATORS & HELPERS
# ==============================================================================

def get_active_trip_session_id(user_id: str) -> str | None:
    """
    Checks Firestore for an active trip session for a given user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        str | None: The ID of the active session if found, otherwise None.
    """
    try:
        sessions_ref = (
            LOCATION_SHARING_COLLECTION
            .where(filter=FieldFilter("user_id", "==", user_id))
            .where(filter=FieldFilter("status", "in", ["active", "sos"]))
            .limit(1)
        )
        active_sessions = list(sessions_ref.stream())
        if active_sessions:
            return active_sessions[0].id
    except Exception as e:
        app.logger.error(f"Failed to check for active trip for {user_id}: {e}", exc_info=True)
    return None

def token_required(f):
    """
    Decorator to protect Flask routes, ensuring that a valid JWT token
    is present in cookies or headers. It also redirects users with an
    active trip session away from dashboard/logout routes.

    Args:
        f (function): The Flask route function to be decorated.

    Returns:
        function: The wrapped function with authentication logic.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('jwt_token') or request.headers.get('Authorization')
        if not token:
            return redirect(url_for('home'))
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={'verify_exp': True})
            current_user_id = data['user_id']
            
            active_session_id = get_active_trip_session_id(current_user_id)
            if active_session_id and request.path in [url_for('dashboard'), url_for('logout')]:
                return redirect(url_for('traveling_page', id=active_session_id))
                
            return f(current_user_id, *args, **kwargs)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            response = redirect(url_for('home'))
            response.delete_cookie('jwt_token')
            return response
    return decorated

def clean_phone_number(input_number: str) -> str:
    """
    Cleans and formats a phone number for Twilio, converting it to
    the international format with a '+63' prefix if it starts with '0'.

    Args:
        input_number (str): The phone number to clean.

    Returns:
        str: The Twilio-formatted phone number (e.g., '+63917xxxxxxx').
    """
    num_str = str(input_number)
    if num_str.startswith('0'):
        cleaned_num = num_str[1:]
    else:
        cleaned_num = num_str
    twilio_formatted_number = f"+63{cleaned_num}"
    return twilio_formatted_number

@cache.memoize(timeout=CACHE_TIMEOUT_SECONDS)
def fetch_user_info(uid: str) -> dict:
    """
    Fetches and caches user information from Firebase Auth.

    Args:
        uid (str): The user ID (UID) from Firebase Authentication.

    Returns:
        dict: A dictionary containing user information (id, email, name).
    """
    user_record = auth.get_user(uid)
    return {
        'id': user_record.uid,
        'email': user_record.email,
        'name': user_record.display_name or user_record.email.split('@')[0]
    }

def send_travel_notification(user_name: str, emergency_contacts: list, puv_details: dict, destination: dict, shareable_link: str):
    """
    Sends SMS notifications to emergency contacts in the background using Twilio.
    This function is submitted to the thread pool executor.

    Args:
        user_name (str): The name of the user who started the trip.
        emergency_contacts (list): A list of dictionaries, each containing contact 'name', 'phone', and 'relationship'.
        puv_details (dict): Details about the Public Utility Vehicle (e.g., 'plate_number', 'type', 'note').
        destination (dict): Details about the destination (e.g., 'address').
        shareable_link (str): The URL to share the live trip tracking.
    """
    for contact in emergency_contacts:
        message = (
            f"Travel Update from Travel Guard \n\n"
            f"Your {contact.get('relationship', 'contact')} {user_name} has started a trip to {destination['address']}.\n\n"
            f"PUV Info:\n"
            f"Plate: {puv_details.get('plate_number', 'N/A')}\n"
            f"Type: {puv_details.get('type', 'N/A')}\n"
            f"Note: {puv_details.get('note', 'N/A')}\n"
            f"Departed: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"You can follow the trip live here: {shareable_link}\n\n"
            f"Travel Guard is monitoring this journey. #TravelSafeWithTravelGuard"
        )
        try:
            # The original code has a hardcoded Twilio 'to' number.
            # I will keep the original code's logic of using a dev number for safety,
            # but note that `cleaned_phone` would be used in a production environment.
            cleaned_phone = clean_phone_number(contact['phone'])
            TWILIO.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to='+18777804236' # Original code used this dev number for 'to'
            )
            app.logger.info(f"SMS notification would be sent to {contact.get('phone')}")
            
        except Exception as e:
            app.logger.error(f"Failed to send SMS to {contact.get('phone')}: {e}", exc_info=True)

def send_trip_cancelled_sms(user_name: str, contacts: list):
    """
    Sends a short SMS notification to emergency contacts when a trip is cancelled.

    Args:
        user_name (str): The name of the user who cancelled the trip.
        contacts (list): A list of dictionaries, each containing contact 'phone'.
    """
    for c in contacts:
        try:
            TWILIO.messages.create(
                body=f"{user_name} cancelled the trip. #TravelGuardUpdate",
                from_=TWILIO_FROM_NUMBER,
                to="+18777804236" # Original code used this dev number for 'to'
            )
            app.logger.info(f"Trip-cancel SMS would be sent to {c.get('phone')}")
        except Exception as e:
            app.logger.error(f"Failed to send cancel SMS to {c.get('phone')}: {e}", exc_info=True)

def send_arrived_sms(user_name: str, destination: str, contacts: list):
    """
    Sends a short SMS notification to emergency contacts confirming safe arrival.

    Args:
        user_name (str): The name of the user who arrived.
        destination (str): The address of the destination.
        contacts (list): A list of dictionaries, each containing contact 'phone'.
    """
    arrived_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    for c in contacts:
        try:
            msg = (f"{user_name} has arrived safely at {destination} "
                   f"as of {arrived_at}. #TravelGuardUpdate")
            TWILIO.messages.create(
                body=msg,
                from_=TWILIO_FROM_NUMBER,
                to="+18777804236" # Original code used this dev number for 'to'
            )
            app.logger.info(f"Arrival SMS would be sent to {c.get('phone')}")
        except Exception as e:
            app.logger.error(f"Failed to send arrival SMS to {c.get('phone')}: {e}", exc_info=True)

def _send_sos_notifications_task(user_name: str, contacts: list, session: dict):
    """
    Background task to send SOS notifications via Twilio to emergency contacts.
    This function is intended to be run in a thread pool.

    Args:
        user_name (str): The name of the user who triggered the SOS.
        contacts (list): A list of dictionaries, each containing contact 'phone'.
        session (dict): The current session data containing location, destination, and vehicle details.
    """
    app.logger.info(f"Initiating SOS for {user_name}. Notifying {len(contacts)} contacts.")
    for contact in contacts:
        message = (
            f"EMERGENCY ALERT from Travel Guard\n\n"
            f"{user_name} has triggered an SOS alert!\n\n"
            f"Last known location: {session.get('current_location', {}).get('lat')}, "
            f"{session.get('current_location', {}).get('lng')}\n"
            f"Destination: {session.get('destination', {}).get('address')}\n"
            f"Vehicle: {session.get('puv_details', {}).get('type')} "
            f"({session.get('puv_details', {}).get('plate_number')})\n\n"
            f"View live trip: {session.get('shareable_link')}"
        )
        try:
            # The original code has a hardcoded Twilio 'to' number.
            # I will keep the original code's logic of using a dev number for safety,
            # but note that `cleaned_phone` would be used in a production environment.
            cleaned_phone = clean_phone_number(contact['phone'])
            TWILIO.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to='+18777804236' # Original code used this dev number for 'to'
            )
            app.logger.info(f"SOS SMS would be sent to {contact.get('phone')}")
        except Exception as e:
            app.logger.error(f"Failed to send SOS SMS to {contact.get('phone')}: {e}", exc_info=True)


# ==============================================================================
# V. ROUTE DEFINITIONS
# ==============================================================================

# --- Page Rendering Routes ---

@app.route('/')
def home():
    """
    Renders the login page. If a valid JWT token is found, it attempts to
    redirect to the dashboard or an active traveling page if a trip is ongoing.

    Returns:
        Response: The rendered login page or a redirect response.
    """
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
    """
    Renders the main application dashboard. Requires a valid JWT token.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: The rendered dashboard page.
    """
    user_ref = USER_COLLECTION.document(current_user)
    user_doc = user_ref.get()

    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_data['is_new_user'] = not user_data.get('profile_complete', False)
    else:
        user_data = fetch_user_info(current_user)
        user_data['is_new_user'] = True

    return render_template('dashboard.html', user=user_data)

@app.route('/travelling')
@token_required
def traveling_page(current_user: str):
    """
    Renders the active traveling page. If no session ID is provided, it attempts
    to find an active session for the current user and redirects.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: The rendered traveling page or a redirect.
    """
    session_id = request.args.get('id')
    if not session_id:
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
    """
    Renders the publicly shareable view of a trip. Displays live updates for active trips
    or a final status page for completed, cancelled, or SOS trips.

    Args:
        session_id (str): The ID of the location sharing session.

    Returns:
        Response: The rendered shared trip page (live, completed, cancelled, or error).
    """
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

        # The commented out 'sos' block is preserved as in the original code.
        # if status == 'sos':
        #     return render_template(
        #         'shared_trip_final.html',
        #         title="SOS Alert",
        #         emoji="🚨",
        #         headline="The traveller triggered an emergency SOS.",
        #         subtext=f"Last known location: {current_loc.get('lat')}, {current_loc.get('lng')}. "
        #               "Please contact emergency contacts or the nearest authorities immediately."
        #     )

        # Still active or other status that should show the live view
        return render_template(
            'travelling.html',
            session_data=session,
            firebase_config=FIREBASE_CONFIG_FRONTEND
        )

    except Exception as e:
        print(f"DEBUGGER: {e}") # Original code had a print here
        return render_template(
            'shared_trip_expired.html',
            message="This trip could not be loaded."
        )


# --- Authentication & User Management API Routes ---

@app.route('/logout')
@token_required
def logout(current_user: str):
    """
    Logs the user out by clearing the JWT cookie. This route is protected
    and will redirect to the travelling page if a trip is active.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A redirect response to the home page with the JWT cookie cleared.
    """
    response = redirect(url_for('home'))
    response.delete_cookie('jwt_token')
    return response

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """
    Authenticates a user via a Google ID token received from the frontend.
    If successful, it creates or updates a user record in Firestore and
    issues a new JWT token to the user.

    Returns:
        Response: A JSON response containing the JWT token and user details,
                  or an error message.
    """
    id_token = request.json.get('id_token')
    if not id_token:
        return jsonify({'error': 'Missing token'}), 400

    try:
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token['uid']
        user_ref = USER_COLLECTION.document(user_id)
        user_doc = user_ref.get()
        is_new_user = not user_doc.exists

        if is_new_user:
            # For new users, create the document with profile_complete set to False.
            user_data = {
                'email': decoded_token['email'],
                'name': decoded_token.get('name', ''),
                'last_login': firestore.SERVER_TIMESTAMP,
                'profile_complete': False
            }
            user_ref.set(user_data)
            # Schedule background tasks to create placeholder subcollections.
            executor.submit(user_ref.collection('emergency_contacts').document('default').set({'placeholder': True}))
            executor.submit(user_ref.collection('travel_history').document('default').set({'placeholder': True}))
        else:
            # For returning users, only update non-critical info.
            # DO NOT change 'profile_complete' status here.
            user_data = {
                'email': decoded_token['email'],
                'name': decoded_token.get('name', ''),
                'last_login': firestore.SERVER_TIMESTAMP
            }
            user_ref.update(user_data)

        payload = {
            'user_id': user_id,
            'email': decoded_token['email'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
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
    """
    Retrieves the complete user profile from Firestore. Requires authentication.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response containing the user's profile data or an error message.
    """
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
    data = request.json
    print(f"DEBUGGER {data}")
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    try:
        user_ref = USER_COLLECTION.document(current_user)

        # Save 6-digit passcode if provided
        if passcode := data.get('passcode'):
            hashed = bcrypt.hashpw(passcode.encode(), bcrypt.gensalt())
            executor.submit(user_ref.update, {'passcode_hash': hashed.decode()})

        # Update basic profile
        update_data = {
            'phone': data.get('phone'),
            'profile_complete': True,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        executor.submit(user_ref.update, update_data)

        # Add emergency contact if provided
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

@app.route('/api/login-passcode', methods=['POST'])
def login_passcode():
    try:
        email = request.form.get('email', '').strip()
        code  = request.form.get('passcode', '').strip()

        if not email or not code:
            return redirect(url_for('home', error='missing_credentials'))

        user_docs = USER_COLLECTION.where('email', '==', email).limit(1).get()
        if not user_docs:
            return redirect(url_for('home', error='not_found'))

        user_doc = user_docs[0]
        user_dict = user_doc.to_dict()
        hashed = user_dict.get('passcode_hash', '')

        if not hashed or not bcrypt.checkpw(code.encode(), hashed.encode()):
            return redirect(url_for('home', error='invalid_passcode'))

        payload = {
            'user_id': user_doc.id,
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        response = make_response(redirect(url_for('dashboard')))
        
        response.set_cookie(
            'jwt_token',
            token,
            httponly=True,
            secure=(FLASK_ENV == 'production'), 
            samesite='Lax',
            max_age=24 * 3600
        )
        return response

    except Exception as e:
        app.logger.error(f"Passcode login error: {e}", exc_info=True)
        return redirect(url_for('home', error='server_error'))

# --- Chatbot API Routes ---

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    """
    Handles user messages for the Gemini AI chatbot. Manages chat history
    and logs interactions to Firestore in the background. Requires authentication.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response containing the AI's reply and any grounding metadata.
    """
    if not (data := request.json) or not (user_message := data.get('message')):
        return jsonify({'error': 'No message provided'}), 400

    with chat_sessions_lock:
        session = chat_sessions.setdefault(current_user, {'history': []})
    
    try:
        user_info = fetch_user_info(current_user)
        username = user_info['name']
        
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

        session['history'].extend([
            {'role': 'user', 'content': user_message},
            {'role': 'model', 'content': response.text}
        ])
        session['history'] = session['history'][-20:]

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
    """
    Manages user's travel history entries.
    GET: Retrieves all travel history records for the current user.
    POST: Adds a new travel history record for the current user.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response containing travel history data or status/error.
    """
    try:
        travel_history_ref = USER_COLLECTION.document(current_user).collection('travel_history')
        if request.method == 'GET':
            query = travel_history_ref.order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            history = []
            for doc in query:
                if doc.id == 'default':
                    continue
                record = doc.to_dict()
                if record.get('sos_triggered') is True:
                    record['status'] = 'sos'
                history.append({'id': doc.id, **record})
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
                'sos_triggered': data.get('sos_triggered', False),
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
    """
    Initiates a new location sharing session. Stores session data in both
    Firestore (for persistent records) and Realtime Database (for live updates).
    Triggers SMS notifications to emergency contacts in the background.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response with the session ID and shareable link.
    """
    data = request.json
    if not all(k in data for k in ['puv_details', 'emergency_contacts', 'destination']):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        session_id = str(uuid.uuid4())
        shareable_link = url_for('shared_traveling_page', session_id=session_id, _external=True)
        
        session_data = {
            'user_id': current_user,
            'puv_details': data['puv_details'],
            'emergency_contacts': data['emergency_contacts'],
            'destination': data['destination'],
            'current_location': data.get('current_location'),
            'status': 'active',
            'sos_triggered': False,
            'start_time': {'.sv': 'timestamp'},
            'end_time': None,
            'shareable_link': shareable_link
        }
        
        # Write to Realtime Database for live updates
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
    """
    Updates the current location of an ongoing trip session in Realtime Database.

    Args:
        current_user (str): The ID of the authenticated user. (Used by decorator, not directly in function body).

    Returns:
        Response: A JSON response indicating success or failure.
    """
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
    """
    Ends a location sharing session, updating its status in Firestore and
    Realtime Database. Also records the trip in travel history and sends
    final notifications (cancelled or arrived).

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response indicating success or failure.
    """
    session_id = request.json.get('session_id')
    status = request.json.get('status', 'completed')

    if not session_id:
        return jsonify({"error": "Missing session ID"}), 400

    try:
        rt_db_ref = realtime_db.reference(f'location_sessions/{session_id}')
        session_data = rt_db_ref.get()

        if not session_data or session_data.get('user_id') != current_user:
            return jsonify({"error": "Permission denied"}), 403

        sos_triggered = session_data.get('sos_triggered', False)

        history_data = {
            'destination': session_data.get('destination', {}).get('address', 'Unknown'),
            'start_time': session_data.get('start_time'),
            'end_time': firestore.SERVER_TIMESTAMP,
            'status': status,
            'sos_triggered': sos_triggered,
            'puv_type': session_data.get('puv_details', {}).get('type', 'Unknown'),
            'vehicle_plate': session_data.get('puv_details', {}).get('plate_number', 'N/A'),
            'notes': session_data.get('puv_details', {}).get('notes', ''),
            'created_at': firestore.SERVER_TIMESTAMP
        }

        batch = db.batch()

        session_ref = LOCATION_SHARING_COLLECTION.document(session_id)
        batch.update(session_ref, {
            'status': status,
            'end_time': firestore.SERVER_TIMESTAMP
        })

        history_ref = USER_COLLECTION.document(current_user) \
            .collection('travel_history').document(session_id)
        batch.set(history_ref, history_data)

        batch.commit()

        user_info = fetch_user_info(current_user)
        contacts = session_data.get('emergency_contacts', [])
        if status == 'cancelled':
            executor.submit(send_trip_cancelled_sms, user_info['name'], contacts)
        elif status == 'completed':
            dest = session_data.get('destination', {}).get('address', 'your destination')
            executor.submit(send_arrived_sms, user_info['name'], dest, contacts)

        executor.submit(
            rt_db_ref.update,
            {'status': status, 'end_time': {'.sv': 'timestamp'}}
        )

        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.error(f"Location sharing end error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/location-sharing/sos', methods=['POST'])
@token_required
def send_sos_alert(current_user: str):
    """
    Triggers an SOS alert for an active location sharing session.
    Updates the session status, logs to travel history, and sends
    emergency notifications to contacts in the background.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response indicating success or failure.
    """
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

        history_data = {
            'destination': session.get('destination', {}).get('address', 'Unknown'),
            'start_time': session.get('start_time'),
            'end_time': None,
            'status': 'sos',
            'sos_triggered': True,
            'puv_type': session.get('puv_details', {}).get('type', 'Unknown'),
            'vehicle_plate': session.get('puv_details', {}).get('plate_number', 'N/A'),
            'notes': session.get('puv_details', {}).get('notes', ''),
            'created_at': firestore.SERVER_TIMESTAMP,
        }

        batch = db.batch()
        session_ref = LOCATION_SHARING_COLLECTION.document(session_id)
        batch.update(session_ref, {
            'status': 'sos',
            'sos_triggered': True,
            'sos_triggered_at': firestore.SERVER_TIMESTAMP
        })

        history_ref = USER_COLLECTION.document(current_user).collection('travel_history').document(session_id)
        batch.set(history_ref, history_data)

        batch.commit()

        ref.child('status').set('sos')
        ref.child('sos_triggered').set(True)

        executor.submit(_send_sos_notifications_task, user_info['name'], contacts, session)

        return jsonify({"status": "success", "message": "SOS alert is being sent."})

    except Exception as e:
        app.logger.error(f"SOS alert error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# --- Emergency Contacts API Routes ---

@app.route('/api/emergency-contacts', methods=['GET'])
@app.route('/api/push-emergency-contacts', methods=['GET', 'POST']) # Original code had two routes for this
@token_required
def emergency_contacts_manager(current_user: str):
    """
    Manages a user's emergency contacts.
    GET: Retrieves all emergency contacts for the current user.
    POST: Adds a new emergency contact for the current user.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response containing contacts data or status/error.
    """
    try:
        # print("DEBUGGER: INLINE CHECKPOINT") # Original code had a print here
        contacts_ref = USER_COLLECTION.document(current_user).collection('emergency_contacts')
        
        if request.method == 'GET':
            query = contacts_ref.order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            contacts = []
            for doc in query:
                if doc.id == 'default':  
                    continue
                contacts.append({'id': doc.id, **doc.to_dict()})
            return jsonify(contacts)

        if request.method == 'POST':
            data = request.json
            if not data or not data.get('name') or not data.get('phone'):
                return jsonify({"error": "Name and phone are required"}), 400

            count_query = contacts_ref.count()
            count = count_query.get()[0][0].value
            if count >= 5:
                return jsonify({"error": "Maximum of 5 emergency contacts allowed"}), 400

            contact_data = {
                'name': data['name'],
                'phone': data['phone'],
                'relationship': data.get('relationship', ''),
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            new_doc_ref = contacts_ref.document()
            executor.submit(new_doc_ref.set, contact_data)
            return jsonify({"status": "success", "id": new_doc_ref.id}), 201

    except Exception as e:
        app.logger.error(f"Emergency contacts error on {request.method}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/emergency-contacts/<contact_id>', methods=['PUT', 'DELETE'])
@token_required
def manage_emergency_contact(current_user: str, contact_id: str):
    """
    Updates or deletes a specific emergency contact for the current user.

    Args:
        current_user (str): The ID of the authenticated user.
        contact_id (str): The ID of the contact to manage.

    Returns:
        Response: A JSON response indicating success or failure.
    """
    try:
        contact_ref = USER_COLLECTION.document(current_user).collection('emergency_contacts').document(contact_id)
        
        if not contact_ref.get().exists:
            return jsonify({"error": "Contact not found"}), 404

        if request.method == 'PUT':
            data = request.json
            update_data = {
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            if 'name' in data:
                update_data['name'] = data['name']
            if 'phone' in data:
                update_data['phone'] = data['phone']
            if 'relationship' in data:
                update_data['relationship'] = data['relationship']
            
            executor.submit(contact_ref.update, update_data)
            return jsonify({"status": "success"})

        if request.method == 'DELETE':
            contacts_ref = USER_COLLECTION.document(current_user).collection('emergency_contacts')
            count_query = contacts_ref.count()
            count = count_query.get()[0][0].value
            
            if count <= 1:
                return jsonify({"error": "Cannot delete last emergency contact"}), 400
            
            executor.submit(contact_ref.delete)
            return jsonify({"status": "success"})

    except Exception as e:
        app.logger.error(f"Emergency contact management error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# --- Preset Destinations API Routes ---

@app.route('/api/preset-destinations', methods=['GET', 'POST'])
@token_required
def preset_destinations(current_user: str):
    """
    Manages a user's preset destinations.
    GET: Retrieves all preset destinations for the current user.
    POST: Adds a new preset destination for the current user (limited to 5).

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response containing preset destinations data or status/error.
    """
    try:
        if request.method == 'GET':
            query = USER_COLLECTION.document(current_user) \
                .collection('preset_destinations') \
                .order_by('created_at', direction=firestore.Query.DESCENDING) \
                .stream()
            
            presets = []
            for doc in query:
                preset = doc.to_dict()
                preset['id'] = doc.id
                presets.append(preset)
            
            return jsonify(presets)

        if request.method == 'POST':
            data = request.json
            if not data or not data.get('address'):
                return jsonify({"error": "Missing address"}), 400

            count_query = USER_COLLECTION.document(current_user) \
                .collection('preset_destinations') \
                .count()
            
            count = count_query.get()[0][0].value
            if count >= 5:
                return jsonify({"error": "Maximum of 5 preset destinations allowed"}), 400

            preset_data = {
                'name': data.get('name', 'My Destination'),
                'address': data['address'],
                'coordinates': data.get('coordinates'),
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            new_doc_ref = USER_COLLECTION.document(current_user) \
                .collection('preset_destinations') \
                .document()
            
            executor.submit(new_doc_ref.set, preset_data)
            return jsonify({"status": "success", "id": new_doc_ref.id}), 201

    except Exception as e:
        app.logger.error(f"Preset destinations error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/preset-destinations/<preset_id>', methods=['PUT', 'DELETE'])
@token_required
def manage_preset_destination(current_user: str, preset_id: str):
    """
    Updates or deletes a specific preset destination for the current user.

    Args:
        current_user (str): The ID of the authenticated user.
        preset_id (str): The ID of the preset destination to manage.

    Returns:
        Response: A JSON response indicating success or failure.
    """
    try:
        preset_ref = USER_COLLECTION.document(current_user) \
            .collection('preset_destinations') \
            .document(preset_id)
        
        if not preset_ref.get().exists:
            return jsonify({"error": "Preset not found"}), 404

        if request.method == 'PUT':
            data = request.json
            update_data = {
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            if 'name' in data:
                update_data['name'] = data['name']
            if 'address' in data:
                update_data['address'] = data['address']
            if 'coordinates' in data:
                update_data['coordinates'] = data['coordinates']
            
            executor.submit(preset_ref.update, update_data)
            return jsonify({"status": "success"})

        if request.method == 'DELETE':
            executor.submit(preset_ref.delete)
            return jsonify({"status": "success"})

    except Exception as e:
        app.logger.error(f"Preset destination management error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# --- Configuration & Test Routes ---

@app.route('/api/maps-config')
def get_maps_config():
    """
    Provides the Google Maps API key and related configuration to the frontend.

    Returns:
        Response: A JSON response containing Google Maps configuration.
    """
    return jsonify({
        'key': os.getenv('GOOGLE_MAPS_API_KEY'),
        'libraries': 'places,marker',
        'mapId': os.getenv('MAP_ID')
    })

@app.route('/api/protected')
@token_required
def protected_route(current_user: str):
    """
    A sample protected route for testing JWT authentication.

    Args:
        current_user (str): The ID of the authenticated user.

    Returns:
        Response: A JSON response with a welcome message for the authenticated user.
    """
    user_info = fetch_user_info(current_user)
    return jsonify({'message': f"Hello {user_info.get('name')}, you are authenticated!"})

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json')


# ==============================================================================
# VI. APPLICATION RUNNER
# ==============================================================================

if __name__ == '__main__':
    # Runs the Flask application.
    # Host is set to 0.0.0.0 to make it accessible externally (e.g., in Docker).
    # Port is configurable via environment variable, defaults to 5000.
    # Debug mode is enabled only in 'development' FLASK_ENV.
    # Threaded mode allows handling multiple requests concurrently.
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=(FLASK_ENV == 'development'),
        threaded=True
    )
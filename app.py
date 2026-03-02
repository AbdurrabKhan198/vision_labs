import os
import io
import datetime
import torch
import torch.nn as nn
import bcrypt
import boto3
import requests
import googlemaps  # Make sure to pip install googlemaps
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from bson.objectid import ObjectId
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'vision_labs_2026')

# --- DATABASE SETUP ---
try:
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client['VisionLabs_DermAI']
    users_collection = db['users']
    history_collection = db['scans']
    print("✅ MongoDB Atlas Connected")
except Exception as e:
    print(f"❌ DB Connection Error: {e}")

# --- CLOUD STORAGE SETUP ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

# --- GOOGLE MAPS SETUP ---
# Make sure GOOGLE_MAPS_API_KEY is in your .env
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

def upload_to_s3(file_obj, filename):
    try:
        bucket = os.getenv('S3_BUCKET_NAME')
        file_obj.seek(0)
        s3_client.upload_fileobj(
            file_obj, bucket, filename,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/jpeg"}
        )
        return f"https://{bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{filename}"
    except Exception as e:
        print(f"❌ S3 Upload Error: {e}")
        return None

# --- AQI API HELPER ---
def get_live_aqi(city):
    token = os.getenv('AQI_API_TOKEN')
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        response = requests.get(url, timeout=5).json()
        if response['status'] == 'ok':
            return response['data']['aqi']
        return "N/A"
    except Exception as e:
        print(f"❌ AQI Fetch Error: {e}")
        return "N/A"

# --- GOOGLE MAPS HELPER ---
def get_nearby_doctors(city):
    try:
        # Searching for top dermatologists in the user's registered city
        query = f"best dermatologist in {city}"
        places_result = gmaps.places(query=query)
        
        doctors = []
        # Taking top 4 results for the UI
        for place in places_result.get('results', [])[:4]:
            doctors.append({
                'name': place.get('name'),
                'address': place.get('formatted_address'),
                'rating': place.get('rating', 'N/A'),
                'place_id': place.get('place_id')
            })
        return doctors
    except Exception as e:
        print(f"❌ Google Maps Error: {e}")
        return []

# --- SMART DIAGNOSTIC LOGIC (DYNAMIC) ---
def get_severity_logic(prediction, confidence, aqi, city_name):
    conf_val = float(confidence.replace('%', ''))
    aqi_val = 0 if aqi == "N/A" else int(aqi)
    
    severity = "Normal"
    doctor_visit = f"Not required currently in {city_name}. Monitor for 3 days."

    if conf_val > 85 or (conf_val > 70 and aqi_val > 150):
        severity = "Severe"
        doctor_visit = f"URGENT: Please consult a Dermatologist in {city_name} soon."
    elif conf_val > 60 or aqi_val > 100:
        severity = "Moderate"
        doctor_visit = f"Recommended: Consult a professional in {city_name} if symptoms persist."

    care_data = {
        'Acne': {
            'steps': ['Salicylic acid cleanser use karein.', 'Oil-free moisturizer lagayein.'],
            'diet': 'Sugar aur dairy products kam karein.',
            'aqi_advice': f"{city_name} mein pollution se pores block ho sakte hain."
        },
        'Eczema': {
            'steps': ['Skin hydrate rakhein.', 'Hard soaps avoid karein.'],
            'diet': 'Anti-inflammatory food lein.',
            'aqi_advice': f"{city_name} ki dry air itching trigger kar sakti hai."
        },
        'Pigmentation': {
            'steps': ['SPF 50+ Sunscreen hamesha lagayein.', 'Vitamin C serum use karein.'],
            'diet': 'Antioxidants rich food khayein.',
            'aqi_advice': f"{city_name} ke pollutants melanin trigger kar sakte hain."
        }
    }

    current_care = care_data.get(prediction, {})
    return {
        "severity": severity,
        "doctor_visit": doctor_visit,
        "steps": current_care.get('steps', []),
        "diet": current_care.get('diet', ''),
        "aqi_advice": current_care.get('aqi_advice', '')
    }

# --- AUTH SETUP ---
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.city = user_data.get('city', 'Lucknow')

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(user_data) if user_data else None

# --- AI MODEL SETUP ---
MODEL_PATH = 'best_skin_model.pt'
CLASS_NAMES = ['Acne', 'Eczema', 'Pigmentation']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Vision Labs: AI Model Loaded!")
    return model.to(device)

model = load_model()

def transform_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0).to(device)

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email, pw, city = request.form.get('email'), request.form.get('password'), request.form.get('city')
        if users_collection.find_one({"email": email}):
            flash("Email already exists!")
            return redirect(url_for('signup'))
        
        hashed_pw = bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({"email": email, "password": hashed_pw, "city": city, "date": datetime.datetime.now()})
        flash("Account created! Please login.")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_data = users_collection.find_one({"email": request.form.get('email')})
        if user_data and bcrypt.checkpw(request.form.get('password').encode('utf-8'), user_data['password']):
            login_user(User(user_data))
            return redirect(url_for('dashboard'))
        flash("Invalid credentials!")
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    history = list(history_collection.find({"user_id": current_user.id}).sort("date", -1))
    return render_template('dashboard.html', history=history)
# ... (purane imports wahi rahenge)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            file_bytes = file.read()
            img_buffer = io.BytesIO(file_bytes)
            
            # 1. AI Inference
            img_buffer.seek(0)
            img = Image.open(img_buffer).convert('RGB')
            with torch.no_grad():
                outputs = model(transform_image(img))
                _, pred = torch.max(outputs, 1)
                conf = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item() * 100
            res_text = CLASS_NAMES[pred.item()]

            user_city = current_user.city
            aqi_value = get_live_aqi(user_city)

            # 2. Severity Logic
            analysis = get_severity_logic(res_text, f"{conf:.2f}%", aqi_value, user_city)

            # --- SMART TRIAGE LOGIC ---
            # Agar severity Normal hai, toh API hit nahi hogi
            clinics = []
            show_doctors = False
            
            if analysis['severity'] in ['Moderate', 'Severe']:
                clinics = get_nearby_doctors(user_city)
                show_doctors = True
            # --------------------------

            filename = f"scan_{current_user.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            img_buffer.seek(0)
            s3_url = upload_to_s3(img_buffer, filename)
            
            if s3_url:
                history_collection.insert_one({
                    "user_id": current_user.id,
                    "prediction": res_text,
                    "confidence": f"{conf:.2f}%",
                    "image_url": s3_url,
                    "aqi": aqi_value,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "city": user_city,
                    "severity": analysis['severity']
                })
                
                return render_template('predict.html', 
                                     prediction=res_text, 
                                     confidence=f"{conf:.2f}%", 
                                     user_image=s3_url, 
                                     aqi=aqi_value,
                                     analysis=analysis,
                                     city=user_city,
                                     clinics=clinics,
                                     show_doctors=show_doctors, # Template ko batayenge ki doctors dikhane hain ya nahi
                                     datetime=datetime)
    
    return render_template('predict.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
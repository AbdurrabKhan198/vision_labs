import os
import datetime
import torch
import torch.nn as nn
import bcrypt
import boto3
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

# --- DATABASE SETUP (MongoDB Atlas) ---
try:
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client['VisionLabs_DermAI']
    users_collection = db['users']
    history_collection = db['scans']
    print("✅ MongoDB Atlas Connected")
except Exception as e:
    print(f" DB Connection Error: {e}")

# --- CLOUD STORAGE SETUP (AWS S3) ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'ap-south-1')
)

def upload_to_s3(file, filename):
    try:
        bucket = os.getenv('S3_BUCKET_NAME')
        file.seek(0)
        s3_client.upload_fileobj(
            file, bucket, filename,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/jpeg"}
        )
        return f"https://{bucket}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{filename}"
    except Exception as e:
        print(f" S3 Upload Error: {e}")
        return None

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

# --- AI MODEL SETUP (EfficientNet-B0) ---
MODEL_PATH = 'best_skin_model.pt'
CLASS_NAMES = ['Acne', 'Eczema', 'Pigmentation']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
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

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email, pw, city = request.form.get('email'), request.form.get('password'), request.form.get('city')
        if users_collection.find_one({"email": email}):
            flash("Email already exists!")
            return redirect(url_for('signup'))
        
        hashed_pw = bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({"email": email, "password": hashed_pw, "city": city, "date": datetime.datetime.now()})
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

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Generate unique name and upload to S3
            filename = f"scan_{current_user.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            s3_url = upload_to_s3(file, filename)
            
            if s3_url:
                # Inference
                file.seek(0)
                img = Image.open(file).convert('RGB')
                with torch.no_grad():
                    outputs = model(transform_image(img))
                    _, pred = torch.max(outputs, 1)
                    conf = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item() * 100
                
                res_text = CLASS_NAMES[pred.item()]

                # Save to Atlas
                history_collection.insert_one({
                    "user_id": current_user.id,
                    "prediction": res_text,
                    "confidence": f"{conf:.2f}%",
                    "image_url": s3_url,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "city": current_user.city
                })

                return render_template('predict.html', prediction=res_text, confidence=f"{conf:.2f}%", user_image=s3_url)
    return render_template('predict.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
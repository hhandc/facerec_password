import os
import json
import numpy as np
import face_recognition
import hashlib
import base64
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from cryptography.fernet import Fernet
import os

# Initialize the FastAPI app
app = FastAPI()

# Directories for saved data
IMAGE_DIR = "website_images"
FEATURES_FILE = "features.json"


# Ensure the directory for user photos exists
USER_PHOTO_DIR = "user_photos"
os.makedirs(USER_PHOTO_DIR, exist_ok=True)

@app.post("/encrypt-external-data")
async def encrypt_external_data(data: str = Form(...), website: str = Form(...), photo: UploadFile = File(...)):
    """
    Encrypt data sent by another business using the encryption key of a specific website.
    Also processes and stores the captured photo.
    """
    try:
        # Save the captured photo
        photo_path = f"{USER_PHOTO_DIR}/{website}_user_photo.jpg"
        with open(photo_path, "wb") as f:
            f.write(await photo.read())

        # Load existing features and random strings
        if os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, "r") as f:
                saved_features = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="No saved features or encryption keys found.")

        if website not in saved_features:
            raise HTTPException(status_code=404, detail="Encryption key not found for the given website.")

        # Retrieve the random string and features for the website
        random_str = saved_features[website]["random_str"]
        features = saved_features[website]["features"]

        # Generate the encryption key using the saved features and random string
        encryption_key = generate_encryption_key(features, random_str)

        # Encrypt the provided data
        encrypted_data = encrypt_data(data, encryption_key)

        return {"encrypted_data": encrypted_data.decode()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def extract_face_distances(image_path):
    """Extract facial features (eye distance and forehead-to-chin ratio) from an image."""
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)

    if not face_landmarks_list:
        return None, None, "No faces detected in the image."

    face_landmarks = face_landmarks_list[0]

    left_eye = face_landmarks['left_eye']
    right_eye = face_landmarks['right_eye']

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_distance = calculate_distance(left_eye_center, right_eye_center)

    chin = face_landmarks['chin']
    nose_bridge = face_landmarks['nose_bridge']

    forehead_point = nose_bridge[0]
    chin_point = chin[-1]

    forehead_to_chin_distance = calculate_distance(forehead_point, chin_point)

    normalized_forehead_to_chin = forehead_to_chin_distance / eye_distance

    return eye_distance, normalized_forehead_to_chin, None

def is_within_tolerance(new_features, saved_features, tolerance=0.1):
    """Check if the new features are within the acceptable range of saved features."""
    new_eye_distance, new_forehead_to_chin = new_features
    saved_eye_distance, saved_forehead_to_chin = saved_features

    return (
        abs(new_eye_distance - saved_eye_distance) <= tolerance
        and abs(new_forehead_to_chin - saved_forehead_to_chin) <= tolerance
    )

def generate_encryption_key(features, random_str):
    """Generate encryption key based on facial features and random string."""
    eye_distance, forehead_to_chin_distance = features

    eye_distance_str = f"{round(eye_distance, 1):.1f}"
    forehead_to_chin_distance_str = f"{round(forehead_to_chin_distance, 2):.2f}"

    combined_str = eye_distance_str + forehead_to_chin_distance_str + random_str
    hashed_key = hashlib.sha256(combined_str.encode()).digest()
    encryption_key = base64.urlsafe_b64encode(hashed_key[:32])
    return encryption_key

def encrypt_data(data, encryption_key):
    """Encrypt data using the provided encryption key."""
    fernet = Fernet(encryption_key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

@app.post("/process")
async def process(image: UploadFile = File(...), website: str = Form(...), password: str = Form(...)):
    """
    Process uploaded image, generate encryption key, and encrypt password.
    """
    try:
        # Save uploaded image temporarily
        temp_image_path = f"{IMAGE_DIR}/temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(await image.read())

        # Extract facial features from the provided image
        new_features = extract_face_distances(temp_image_path)
        if new_features[2]:
            return JSONResponse(content={"error": new_features[2]}, status_code=400)

        eye_distance, forehead_to_chin_distance = new_features[:2]

        # Load existing features
        if os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, "r") as f:
                saved_features = json.load(f)
        else:
            saved_features = {}

        # Check if the website already has a saved image and features
        if website in saved_features:
            saved_image_path = f"{IMAGE_DIR}/{website}.jpg"
            saved_feature_values = saved_features[website]["features"]

            # Compare new features with saved features
            if is_within_tolerance(new_features, saved_feature_values):
                # Use saved features and random string for key generation
                random_str = saved_features[website]["random_str"]
                encryption_key = generate_encryption_key(saved_feature_values, random_str)
            else:
                return JSONResponse(content={"error": "Facial features do not match the saved image."}, status_code=400)
        else:
            # Save the image and features for the new website
            saved_image_path = f"{IMAGE_DIR}/{website}.jpg"
            os.rename(temp_image_path, saved_image_path)

            random_str = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
            saved_features[website] = {
                "features": [eye_distance, forehead_to_chin_distance],
                "random_str": random_str,
            }

            with open(FEATURES_FILE, "w") as f:
                json.dump(saved_features, f)

            # Generate the encryption key
            encryption_key = generate_encryption_key([eye_distance, forehead_to_chin_distance], random_str)

        # Encrypt the password
        encrypted_password = encrypt_data(password, encryption_key)

        return {
            "encryption_key": encryption_key.decode(),
            "encrypted_password": encrypted_password.decode(),
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/encrypt-external-data")
async def encrypt_external_data(data: str = Form(...), website: str = Form(...)):
    """
    Encrypt data sent by another business using the encryption key of a specific website.
    """
    try:
        # Load existing features and random strings
        if os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, "r") as f:
                saved_features = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="No saved features or encryption keys found.")

        if website not in saved_features:
            raise HTTPException(status_code=404, detail="Encryption key not found for the given website.")

        # Retrieve the random string and features for the website
        random_str = saved_features[website]["random_str"]
        features = saved_features[website]["features"]

        # Generate the encryption key using the saved features and random string
        encryption_key = generate_encryption_key(features, random_str)

        # Encrypt the provided data
        encrypted_data = encrypt_data(data, encryption_key)

        return {"encrypted_data": encrypted_data.decode()}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the introduction page
@app.get("/")
def serve_main_page():
    return FileResponse("static/index.html")

# Serve the face encryption system page
@app.get("/face-encryption")
def serve_face_encryption_page():
    return FileResponse("static/index_face_encryption.html")

# Serve the business data encryption page
@app.get("/encrypt-business-data")
def serve_encrypt_business_page():
    return FileResponse("static/encrypt_business_data.html")

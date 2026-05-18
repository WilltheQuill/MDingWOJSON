import os
import glob
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier

# Point directly to your desktop project folder where your recorded JSONs live
PROJECT_FOLDER = r"C:\Users\Admin\Desktop\MathsDancing\recordings"
MODEL_SAVE_PATH = r"C:\Users\Admin\Desktop\MathsDancing\ai_model.pkl"

GESTURE_MAP = {
    "IDLE_00":0,
    "FOOT_BackCrossStepReturn_01": 1,
    "FOOT_FrontKickReturn_02": 2,
    "FOOT_FrontCrossKickReturn_03": 3,
    
    #"FOOT_FrontStepCrossReturn_04": 4,
    #"FOOT_FrontStepReturn_05": 5,
    #"FOOT_SideStepReturn_06": 6
}

def extract_floats_from_json(json_data):
    """Recursively pulls all numerical coordinates out of a JSON frame."""
    floats = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            floats.extend(extract_floats_from_json(value))
    elif isinstance(json_data, list):
        for item in json_data:
            floats.extend(extract_floats_from_json(item))
    elif isinstance(json_data, (float, int)):
        floats.append(float(json_data))
    return floats

def extract_features_from_json_file(filepath):
    """Loads a recorded JSON file and flattens it into an AI feature array."""
    with open(filepath, 'r') as f:
        frames_list = json.load(f)
        
    matrix = []
    for frame in frames_list:
        numeric_data = extract_floats_from_json(frame)
        if numeric_data:
            matrix.append(numeric_data)
            
    matrix = np.array(matrix)
    
    # Instead of averaging the whole matrix, split it into 3 equal time chunks
    # (Beginning, Middle, and End)
    time_chunks = np.array_split(matrix, 3)

    temporal_features = []
    for chunk in time_chunks:
        if len(chunk) > 0:
         # Calculate the mean and variance for THIS specific slice of time
            chunk_mean = np.mean(chunk, axis=0)
            chunk_std = np.std(chunk, axis=0)
            temporal_features.extend(chunk_mean)
            temporal_features.extend(chunk_std)

    # Combine all time chunks into one massive, highly detailed timeline array
    final_features = np.array(temporal_features)
    
    return final_features

def train_model():
    print(f"Scanning {PROJECT_FOLDER} for recorded JSON files...")
    search_pattern = os.path.join(PROJECT_FOLDER, "*.json")
    files = glob.glob(search_pattern)
    
    X_data = []
    y_labels = []
    
    for file in files:
        filename = os.path.basename(file)
        
        # Skip the layout map file if it's named symbol_map.json
        if "symbol_map" in filename:
            continue
            
        for base_gesture, number_id in GESTURE_MAP.items():
            if base_gesture in filename:
                print(f"Training on layout: {filename} -> Mapped to {number_id}")
                features = extract_features_from_json_file(file)
                X_data.append(features)
                y_labels.append(number_id)
                break
                
    if not X_data:
        print("No matching gesture JSON files found on your Desktop folder yet.")
        return

    print(f"Training dynamic model on matching features...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_data, y_labels)
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Success! New aligned model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
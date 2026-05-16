import os
import glob
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Define your folder path and mapping logic
ROKOKO_FOLDER = r"C:\Users\Admin\AppData\LocalLow\Rokoko Electronics\Rokoko Studio\Rokoko_Data" 
MODEL_SAVE_PATH = "ai_model.pkl"

# Your specific MathsDancing dictionary mapping
GESTURE_MAP = {
     "SideStepReturn_01": 1,
   "FrontKickReturn_02": 2,
   "FrontKickCrossReturn_03": 3,
   "BackStepCrossReturn_04": 4,
   "FrontStepReturn_05": 5,
   "FrontStepCrossReturn_06": 6
}

def extract_features_from_bvh(filepath):
    """Reads a BVH file and extracts statistical features from the motion data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find where the MOTION section starts
    motion_start = 0
    for i, line in enumerate(lines):
        if "MOTION" in line:
            motion_start = i + 3 # Skip 'MOTION', 'Frames:', 'Frame Time:'
            break
            
    # Extract all frames as a matrix of floats
    frames = []
    for line in lines[motion_start:]:
        # Convert the line of text into a list of numbers
        values = [float(x) for x in line.strip().split()]
        frames.append(values)
        
    frames = np.array(frames)
    
    # AI Feature Engineering: We don't care how long the move took. 
    # We care about the average position and the variance (how much things moved).
    mean_features = np.mean(frames, axis=0)
    std_features = np.std(frames, axis=0)
    
    # Combine them into a single 1D array representing the whole file
    return np.concatenate((mean_features, std_features))

def train_model():
    print(f"Scanning {ROKOKO_FOLDER} for .bvh files...")
    search_pattern = os.path.join(ROKOKO_FOLDER, "*.bvh")
    files = glob.glob(search_pattern)
    
    X_data = [] # The mathematical features
    y_labels = [] # The target numbers (1-6)
    
    for file in files:
        filename = os.path.basename(file)
        
        # Check if the file matches any of our known base gestures
        for base_gesture, number_id in GESTURE_MAP.items():
            if base_gesture in filename:
                print(f"Processing: {filename} -> Mapping to {number_id}")
                features = extract_features_from_bvh(file)
                X_data.append(features)
                y_labels.append(number_id)
                break
                
    if not X_data:
        print("No matching files found. Check your file paths and names.")
        return

    # Train the Machine Learning Model
    print("Training AI model on 5 iterations...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_data, y_labels)
    
    # Save the model to your hard drive
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"Training Complete! Model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()

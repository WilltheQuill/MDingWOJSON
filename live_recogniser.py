import socket
import json
import numpy as np
import pickle
import time

# Network settings for Rokoko Studio Forward Data
UDP_IP = "127.0.0.1"
UDP_PORT = 14043
MODEL_PATH = "ai_model.pkl"

# How many frames to capture before making a guess (Assuming ~60fps)
# A 2-second move = 120 frames.
WINDOW_SIZE = 120 

def extract_floats_from_json(json_data):
    """Recursively digs through the JSON stream and pulls out all numerical data."""
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

def start_listening():
    # Load the trained AI
    try:
        with open(MODEL_PATH, 'rb') as f:
            ai_model = pickle.load(f)
        print("AI Model loaded successfully.")
    except FileNotFoundError:
        print("Error: ai_model.pkl not found. Run train_ai.py first.")
        return

    # Open the network port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for Rokoko JSON stream on {UDP_IP}:{UDP_PORT}...")

    frame_buffer = []
    cooldown = 0 # Prevents the AI from spamming the same move multiple times per second

    while True:
        # Wait for a packet from the suit
        data, addr = sock.recvfrom(65536) # 64KB buffer size
        
        try:
            # Decode the JSON
            message = json.loads(data.decode('utf-8'))
            
            # Navigate to the body data (varies slightly by JSON v4 export settings)
            # We pass the whole message into our extractor to grab all skeletal angles/positions
            current_frame_data = extract_floats_from_json(message)
            
            if current_frame_data:
                frame_buffer.append(current_frame_data)
                
            # Once we have enough frames to constitute a complete "move"
            if len(frame_buffer) >= WINDOW_SIZE:
                
                # Check if we are allowed to guess yet
                if time.time() > cooldown:
                    
                    # Convert buffer to numpy array to match our training format
                    buffer_array = np.array(frame_buffer)
                    
                    # Calculate mean and standard deviation just like in training
                    mean_features = np.mean(buffer_array, axis=0)
                    std_features = np.std(buffer_array, axis=0)
                    live_features = np.concatenate((mean_features, std_features))
                    
                    # NOTE: If the length of live_features doesn't match the BVH features perfectly,
                    # the AI will throw an error here. This is the Euler vs Quaternion issue mentioned above.
                    
                    # Ask the AI to guess the move
                    prediction = ai_model.predict([live_features])[0]
                    
                    print(f"\n============================")
                    print(f" DETECTED MOVE NUMBER: {prediction} ")
                    print(f"============================\n")
                    
                    # Set a 2-second cooldown so it doesn't trigger 60 times in a row
                    cooldown = time.time() + 2.0 
                    
                # Slide the window forward (remove the oldest frame)
                frame_buffer.pop(0)
                
        except json.JSONDecodeError:
            pass # Ignore malformed packets

if __name__ == "__main__":
    start_listening()
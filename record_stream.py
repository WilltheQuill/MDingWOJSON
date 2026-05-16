import socket
import json
import time

# Network settings
UDP_IP = "127.0.0.1"
UDP_PORT = 14043

def record_move(move_name, duration=3.0):
    """
    Listens to the Rokoko stream and saves the JSON data to a file.
    move_name: What to call the saved file.
    duration: How many seconds to record.
    """
    filename = f"{move_name}.json"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"Get ready to perform '{move_name}' in 3 seconds...")
    time.sleep(3)
    
    print(f"RECORDING NOW for {duration} seconds!")
    
    end_time = time.time() + duration
    recorded_frames = []

    while time.time() < end_time:
        data, addr = sock.recvfrom(65536)
        try:
            # Decode the live frame and add it to our list
            frame_data = json.loads(data.decode('utf-8'))
            recorded_frames.append(frame_data)
        except json.JSONDecodeError:
            pass

    # Save the captured frames to your hard drive
    with open(filename, 'w') as f:
        json.dump(recorded_frames, f, indent=4)
        
    print(f"Recording complete. Saved as {filename}")

if __name__ == "__main__":
    # Change these names when recording your 5 iterations
    # Example: "BackStepCrossReturn_04_iter1"
    
    target_move = input("Enter the name for this recorded move: ")
    record_move(target_move)
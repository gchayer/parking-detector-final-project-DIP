import cv2
import time
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "best.pt"
HOST_IP = '0.0.0.0'
HOST_PORT = 5001
STATS_UPDATE_INTERVAL = 1.0  # Numerical board updates every 1 second
# ---------------------

app = Flask(__name__)

# Load Model
try:
    model = YOLO(MODEL_PATH)
    print(f"Parking Model Loaded: {MODEL_PATH}")
except Exception as e:
    print(f"Loading Error: {e}")
    exit()

# Global state to keep the "Board" data persistent
parking_stats = {"total": 0, "occupied": 0, "vacant": 0}

def generate_frames():
    global parking_stats
    last_stats_time = 0
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. RUN DETECTION (Fastest possible for fluid boxes)
        # We use a higher iou to help merge those double-box issues
        results = model(frame, verbose=False, conf=0.4, iou=0.45)[0]
        
        # 2. UPDATE BOARD STATS (Only every 1 second)
        current_time = time.time()
        if current_time - last_stats_time >= STATS_UPDATE_INTERVAL:
            classes = results.boxes.cls.tolist()
            
            v_count = classes.count(0) 
            o_count = classes.count(1)
            
            parking_stats["vacant"] = v_count
            parking_stats["occupied"] = o_count
            parking_stats["total"] = v_count + o_count
            last_stats_time = current_time

        # 3. ANNOTATE FRAME (Fluid visual feedback)
        annotated_frame = results.plot()

        # 4. ENCODE AND STREAM
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    # Styled like a digital parking lot sign
    html_page = """
    <html>
    <head>
        <title>Florida Poly Parking Monitor</title>
        <script>
            // Polls the server for stats every 1 second to update the board
            function updateBoard() {
                fetch('/get_stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total').innerText = data.total;
                        document.getElementById('vacant').innerText = data.vacant;
                        document.getElementById('occupied').innerText = data.occupied;
                    });
            }
            setInterval(updateBoard, 1000);
        </script>
        <style>
            body { font-family: 'Courier New', Courier, monospace; background-color: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
            .sign-board { 
                background: #222; border: 4px solid #444; border-radius: 10px; 
                display: inline-block; padding: 20px; margin-bottom: 20px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
            }
            .header { color: #ffcc00; font-size: 1.8rem; margin-bottom: 15px; text-transform: uppercase; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .stat-row { display: flex; justify-content: space-between; width: 300px; font-size: 1.4rem; padding: 5px 0; }
            .val { font-weight: bold; color: #00ff00; }
            .occupied-val { color: #ff4444; }
            .total-val { color: #00ccff; }
            .stream-container { border: 2px solid #333; border-radius: 5px; overflow: hidden; display: inline-block; }
        </style>
    </head>
    <body>
        <div class="sign-board">
            <div class="header">Parking Status: Lot 10</div>
            <div class="stat-row">TOTAL SPOTS: <span id="total" class="total-val">0</span></div>
            <div class="stat-row">AVAILABLE: <span id="vacant" class="val">0</span></div>
            <div class="stat-row">OCCUPIED: <span id="occupied" class="occupied-val">0</span></div>
        </div>
        <br>
        <div class="stream-container">
            <img src="{{ url_for('video_feed') }}" width="800">
        </div>
    </body>
    </html>
    """
    return render_template_string(html_page)

@app.route('/get_stats')
def get_stats():
    # Endpoint to provide the 1-second-delayed data to the UI
    return jsonify(parking_stats)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host=HOST_IP, port=HOST_PORT, debug=False, threaded=True)
    

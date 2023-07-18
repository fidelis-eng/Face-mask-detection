import argparse
import io
from io import BytesIO
import os
from PIL import Image
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, Response
from datetime import datetime
import csv

app = Flask(__name__)


model = torch.hub.load(
        "ultralytics/yolov5", "custom", path = "model/best.pt", force_reload=True
        )

model.eval()
model.conf = 0.6  
model.iou = 0.45  

global chosen_camera, res

def get_cameras():
    i=0
    caplist = []
    while 1:
        if cv2.VideoCapture(i).isOpened():
            caplist.append(i)
        else: 
            break
        print('camera')
        i+=1
    return {idx:f'Camera {idx+1}' for idx,val in enumerate(caplist)}

def gen(chosen_camera_index):
    cap=cv2.VideoCapture(chosen_camera_index)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            log = generate_log(results)
            print(log)
            save_log(log)
            # results.print()
            img = np.squeeze(results.render())
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_log(results):
    global res
    res = []
    resultlist = str(results).split()

    current_time = datetime.now()
    time_stamp = current_time.timestamp()
    date_time = datetime.fromtimestamp(time_stamp)
    str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    list_date_time = str_date_time.split(',')
    
    res = resultlist[3:resultlist.index("Speed:")]
    res.extend(list_date_time)

    return res

def save_log(log):
    file = 'logs/detection log.csv'
    with open(file, 'a',  encoding='UTF8', newline='') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(log)
        csvfile.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo menu.html', cameralist=get_cameras())

@app.route('/display camera', methods=["POST"])
def display_camera():
    global chosen_camera
    """Video streaming route. Put this in the src attribute of an img tag."""
    chosen_camera = int(request.form['chosen_camera'])
    return render_template('demo project.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(chosen_camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 
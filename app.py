from flask import Flask, render_template, request, send_from_directory
import cv2
from ultralytics import YOLO
import os

model = YOLO('license_plate_detector.pt')


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']
    if not img:
        return jsonify({"error": "No file provided or file name is empty"}), 400

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    
    results = model(img_arr,save=True,show_labels=True)[0]

    COUNT += 1
    return render_template('detection.html')



@app.route('/load_img')


def load_img():
    global COUNT

    return send_from_directory('runs\detect\predict', "image0.jpg".format(COUNT-1))
@app.route('/img')
def img():
    global COUNT

    return send_from_directory('static',"{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
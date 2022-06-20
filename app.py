# -*- coding: utf-8 -*-

import os.path
import cv2

import os
from mask_detection import mask_detect_image,mask_detect
from flask import Flask, request, render_template, send_from_directory,redirect,url_for

__author__ = 'Rakhi Sehra'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Homepage template
@app.route('/')
def index():
    return render_template("index.html")

# Webcam feed template
@app.route('/upload', methods=["POST"])
def upload():
    mask_detect()
    return redirect("/", code=302)
   

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)




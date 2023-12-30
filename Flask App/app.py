import os
import sys

import cv2
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)



MODEL_PATH = 'models/tankdetector.h5'
SIZE = 250
# Load your own trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img, model):
    img = np.array(img, dtype="uint8")
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.reshape((1, 250, 250, 3))
    preds = model.predict(img)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        if preds > 0.50:
            result = "Tank"
        else:
            result = "No Tank"
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

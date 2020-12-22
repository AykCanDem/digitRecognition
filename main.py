from flask import Flask
from flask import request
from flask import render_template
import os
import sys
import re
import base64
import cv2 # for resize
import json
#for image processing

import numpy as np
# for importing the trained keras model

from keras.models import load_model
from keras.models import model_from_json


def init():

    json_file = open('mnist_cnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("mnist_cnn_weights.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'], run_eagerly=False)
    #graph = tf.compat.v1.get_default_graph()
    return loaded_model





def convertImage(imgData):
    #find the pattern starts with base64.
    # Remaining part will be regular expression group
    imgstr = str(imgData)
    imgstr = re.search(r'base64,(.*)', imgstr).group(1)

    # convert str to byte array
    byte_img = bytearray()
    byte_img.extend(map(ord, imgstr))
    # save it as png
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(byte_img))

global model
model = init()


app = Flask(__name__)

# this URL will trigger below function
@app.route('/')
def index():
    return render_template('index.html')  


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print('predicted')
    # get raw data of the image
    imgData = request.get_data()

    #saves image as output.png
    convertImage(imgData)

    # Using cv2.imread() method
    # Using 0 to read image in grayscale mode
    x = cv2.imread('output.png', 0)
    x = np.invert(x)
    x = cv2.resize(x, (28, 28))

    x = x.reshape(1, 28, 28, 1)
    x = x.astype('float32') / 255
    out = model.predict(x)
    for index, probability in enumerate(out[0]):
        print(f'{index}: {probability:.10%}')

    print(np.argmax(out, axis=1))
    prediction = np.array_str(np.argmax(out, axis=1))
    # out includes prediction probabilities of 1 instance (1 image)
    probabilities = list(out[0])
    # convert elements to string
    probabilities = [str(i) for i in probabilities]
    # create the response
    response = dict({'probabilities' : probabilities, 'prediction': prediction})
    response = json.dumps(response)
    print(response)
    return response


"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
"""
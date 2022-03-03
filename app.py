# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:05:54 2022

@author: Baseet
"""

import PIL
from keras.models import load_model 
from helper import *
from flask import Flask, request, render_template , jsonify


app = Flask(__name__)


@app.route('/disease', methods=['POST'])
def disease():
   disease =  load_model('m.h5')
   imagefile = request.files.get('imagefile', '')
   img = PIL.Image.open(imagefile)
   Image_to_pred = Load_Images(img)
   value = disease.predict(Image_to_pred)
   prediction = np.argmax(value)
   if (prediction == 0):
       return jsonify({'prediction':'Glioma'})
   elif (prediction == 1):
       return jsonify({'prediction':'No Tumor'})
   elif (prediction == 2):
       return jsonify({'prediction':'Meningioma Tumor'})
   elif (prediction == 3):
       return jsonify({'prediction':'Pituitary Tumor'})
  

if __name__ == '__main__':
	
    app.run()
	
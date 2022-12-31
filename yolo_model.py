# -*- coding: utf-8 -*-

import glob

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import load_model

class Сlassifier:
    name_json = glob.glob('model_yolo/*.json')[0]
    name_weights = glob.glob('model_yolo/*.h5')[0]
    
    def _init_(self, name_json, name_weights):
        self.name_json = name_json
        self.name_weights = name_weights
    
    def loading(self):
        with open(self.name_json, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.name_weights)
        
    def predicted(self, roi, dict_emotions):
        pred = self.model.predict(roi)[0]
        label = dict_emotions[pred.argmax()]
        return label

model = Сlassifier()
model.loading()
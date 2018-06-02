# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 13:01:28 2018

@author: Milestone7
"""
import matplotlib.pyplot as plt
import json
import pandas as pd
import bwr
import time
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

def conv(books):
    elevations = json.dumps(books)
    #print(type(elevations))
    data=pd.read_json(elevations)
    data.columns=['Time','Record']
    yy=min(data.Record)
    yy = yy*-1 if np.sign(yy) == -1 else yy
    data.Record=[x+yy for x in pd.to_numeric(data.Record)]
    (baseline, ecg_out) = bwr.bwr(data.Record)
    data.Record=ecg_out
    #kk=data.to_json(orient='records')
    plt.figure(figsize=(10,2))
    #plt.axis('off')
    plt.plot(data.Time,data.Record)
    plt.savefig('1234.png')
    prediction_key = "45183f01c6a5411aa812727e157ff00f"
    project_id = "40428416-afc2-4201-819e-27ce80646b0f"
    iteration_id = "21d2b2bc-bc4b-4e7f-903e-4d961df831b7"
    print (time.strftime("%H:%M:%S"))
    predictor = prediction_endpoint.PredictionEndpoint(prediction_key)
    with open("1234.png", mode="rb") as test_data:
        results = predictor.predict_image(project_id, test_data, iteration_id)

#print(pd.DataFrame(results.predictions))
    appended_data = []
    for prediction in results.predictions:
        appended_data.append(prediction.tag_name+ ": {0:.2f}%".format(prediction.probability * 100))
        #print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))
    k1=pd.DataFrame(appended_data).to_json(orient='records')
    #print(k1)
    #print (time.strftime("%H:%M:%S"))
    return k1

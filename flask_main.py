from flask import Flask
from flask import request, jsonify
import json
import pandas as pd
import numpy as np
import Test as Dm

app = Flask(__name__)



@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print (request.is_json)
    content = request.get_json()
    data = pd.io.json.json_normalize(content)
    data.columns=['Record']#['Time','Record']
    # Peak points
    data=data.Record
    aDict=Dm.Get_PQRS(data)
    #print (type(content))
    return json.dumps(aDict)#jsonify(content)

if __name__ == '__main__':
  app.run()
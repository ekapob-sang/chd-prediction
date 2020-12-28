# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:25:42 2020

@author: admin-1309
"""

import numpy as np
import flask 
import pickle
from flask import render_template , request



# app
app = flask.Flask(__name__, static_url_path='/static')

# load model
chd = pickle.load(open("E:\Practice deploy\practice1\logisticchd2.pkl","rb"))
scale = pickle.load(open("E:\Practice deploy\practice1\scaler.pkl","rb"))

@app.route("/")
def home():
   return render_template('/main.html')
@app.route("/page")    
def page():
    return render_template('/main.html')


@app.route("/predict", methods=["GET", "POST"])
def result():
    """Gets prediction using the HTML form"""
    if flask.request.method == "POST":
        inputs = flask.request.form
        age = inputs["age"]
        cigs = inputs["cig"]
        sysbp = inputs["sysbp"]
        diabp = inputs["diabp"]
        male = inputs["sex"]
        dm = inputs["dm"]
    fmap = {"female": [1, 0],
            "male": [0, 1],
            "nodm": [1, 0],
            "dm": [0, 1]}
    show = {"female": "หญิง",
            "male": "ชาย",
            "nodm": "ไม่เป็น",
            "dm": "เป็น"}
    X_raw = np.array([int(age)] + [int(cigs)] + [int(sysbp)] + [int(diabp)]).reshape(1, -1)
    X_rawscale = scale.transform(X_raw)
    X_cat = np.array(fmap[male]+fmap[dm]).reshape(1, -1)
    X_new = np.concatenate((X_rawscale,X_cat),axis=None)
    X_new2=X_new.reshape(1, -1)
    age = 20
    cigs = 0
    sysbp = 120
    diabp = 80
    male = "male"
    dm = "dm" 
    a = np.array([int(age)] + [int(cigs)] + [int(sysbp)] + [int(diabp)]).reshape(1, -1)
    b = np.array([str(show[male]),str(show[dm])]).reshape(1, -1)
    c = np.concatenate((a,b),axis=None)
    d=  c.reshape(1, -1)
    value= d.flatten().tolist()
    item = ('age','cigs','sysbp','diabp','gender','diab')
    data=dict(zip(item,value))
    prob = chd.predict_proba(X_new2)
    prob_per = (round(prob[0][1],2)) * 100
    return render_template('/result2.html',data=data,prob_per=prob_per)
    
    


@app.route("/result", methods=["GET" ,"POST"])
def show():
    if request.method == 'POST':
      result = request.form
      return render_template('/result2.html',result=result)



if __name__ == '__main__':
    """Connect to Server"""
    HOST = "127.0.0.1"
    PORT = "4000"
    app.run(HOST, PORT)

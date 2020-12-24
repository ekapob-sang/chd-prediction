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
app = flask.Flask(__name__)

# load model
chd = pickle.load(open("logisticchd.pkl","rb"))

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
    X_new = np.array([int(age)] + [int(cigs)] + [int(sysbp)] + [int(diabp)] + [int(male)] + [int(dm)]).reshape(1, -1)
    yhat = chd.predict(X_new)
    # if yhat[0] == 1:
    #     outcome = "Coronay heart disease"
    # else:
    #     outcome = "normal"
    if male == "1":
        gender = "ชาย"
    else:
        gender = "หญิง"
    if dm == "1":
        diab = "เป็น"
    else:
        diab = "ไม่เป็น"
    prob = chd.predict_proba(X_new)
    prob_per = (round(prob[0][1],2)) * 100
    return render_template('/result.html',gender=gender,age=age,cigs=cigs,sysbp=sysbp,diabp=diabp,diab=diab,prob_per=prob_per)
    # results = """
    #           <body>
    #           <h3> Coronary Heart Disease prediction <h3>
    #           <p><h4> Your profile </h4></p>
              # <table>
              #   <tr>
              #     <td>1.เพศ:</td>
              #     <td>""" + gender + """</td>
              # </tr>
              # <tr>
              #     <td>2.อายุ: </td>
              #     <td>""" + age + """</td>
              # </tr>
              # <tr>
              #     <td>3.จำนวนบุหรี่ที่สูบต่อวัน:</td>
              #     <td>""" + cigs + """</td>
              # </tr>
              # <tr>
              #     <td>4.ความดันโลหิตตัวบน( Systolic BP: </td>
              #     <td>""" + sysbp + """</td>
              # </tr>
              # <tr>
              #     <td>5.ความดันโลหิตตัวล่าง( Diastolic BP): </td>
              #     <td>""" + diabp + """</td>
              # </tr>
              # <tr>
              #     <td>6.คุณเป็นเบาหวานหรือไม่:</td>
              #     <td>""" + diab + """</td>
              # </tr>
              # </table>
    #           <p> คุณจะมีโอกาสเป็นโรคหัวใจขาดเลือดภายใน 10 ปีด้วยโอกาส  """  + str(prob_per) + """ % """""".
    #           </body>"""          
    # return results

@app.route("/result", methods=["GET" ,"POST"])
def show():
    if request.method == 'POST':
      result = request.form
      return render_template('/result.html',result=result)



if __name__ == '__main__':
    app.run(debug=True)

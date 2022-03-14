# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:51:33 2021

@author: sumis
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features=[float(x) for x in request.form.values()]
    final_features=[np.array(init_features)]
    print(final_features)
    
    prediction=model.predict(final_features)
    if(prediction[0]<=50000):
        output="Predicted salary <= 50 k"
    else:
        output="Predicted salary <= 50 k"
    
        
    return render_template('result.html',predict_text='The person  {}'.format(output))

    
    
    return render_template('result.html',predict_text=prediction)

if __name__ == '__main__':
      app.run(port=8000)

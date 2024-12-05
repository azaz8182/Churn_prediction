from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = keras.models.load_model('churn_3.h5')

app=Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    Credit = [a for a in request.form.values()]
    Country = [a for a in request.form.values()]
    Gender = [a for a in request.form.values()]
    Age = [a for a in request.form.values()]
    TEnure = [a for a in request.form.values()]
    BAlance = [a for a in request.form.values()]
    Card = [a for a in request.form.values()]
    Member = [a for a in request.form.values()]
    Salary = [a for a in request.form.values()]
    
    
    final_features = pd.DataFrame([Credit,Country,Gender,Age,TEnure,BAlance,Card,Member,Salary],columns=['CreditScore','Geography','Gender','Age','Tenure','Balance','HasCrCard','IsActiveMember','EstimatedSalary'])
    final_features.replace(to_replace={'no':1,'yes':2,'No':1,'Yes':2,'NO':1,'YES':2},inplace=True)
    final_features["Gender"].replace(to_replace={'MALE':2,'FEMALE':1,'Male':2,'Female':1,'male':2,'female':1},inplace=True)
    final_features['Geography'].replace(to_replace={'FRANCE':0,'France':0,'france':0,'Germany':1,'GERMANY':1,'germany':1,'Spain':2,'SPAIN':2,'spain':2},inplace=True)
    ss = StandardScaler()
    final = ss.fit_transform(final_features)
    
    
    # Make prediction
    prediction =model.predict(final)
    #prediction.replace(to_replace ={0:'Not Have Heart Disease',1:'Have Heart Disease'},inplace=True)
    output = prediction[0]
    return render_template('index.html', prediction_text='This Customer Not Exited' if output ==0 else 'This Customer Exited')

    #return render_template('index.html', prediction_text='You {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
# Importing essential libraries
from pydoc import render_doc
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

STATIC_FOLDER = 'templates/'
app = Flask(__name__,
            static_folder=STATIC_FOLDER)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/')
def predict_diabetes():
    return render_template('prediction.html')

@app.route('/predict', methods=["GET","POST"])
def predict():
    
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
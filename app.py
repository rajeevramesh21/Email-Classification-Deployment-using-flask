import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('emailmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dash1', methods=['POST'])
def dash1() :
    return render_template('dash1.html')

@app.route('/dash2', methods=['POST'])
def dash2() :
    return render_template('dash2.html')

@app.route('/dash3', methods=['POST'])
def dash3() :
    return render_template('dash1.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    #final_features = [int_features]
    prediction = model.predict(int_features)

    output = prediction

    return render_template('index.html', prediction_text='The content is {}'.format(output))




if __name__ == "__main__":
    app.run(debug=True)

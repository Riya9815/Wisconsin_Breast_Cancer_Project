import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import scipy
app = Flask(__name__)
model = pickle.load(open('logistic_cancer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    a = int(request.form['a'])
    b = int(request.form['b'])
    c = int(request.form['c'])
    d = int(request.form['d'])
    e = int(request.form['e'])
    f = int(request.form['f'])
    g = int(request.form['g'])
    h = int(request.form['h'])
    i = int(request.form['i'])
    j = int(request.form['j'])
    k = int(request.form['k'])
    l = int(request.form['l'])
    m = int(request.form['m'])




    data = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
    output = model.predict(data)
    #print(my_prediction)
    if output > 0.5:
        return render_template('index.html',pred=f'You have chance of having Breast Cancer.\nProbability of having Diabetes is {output}')
    else:
        return render_template('index.html', pred=f'You are safe.\n Probability of having Breast Cancer is {output}')
if __name__ == "__main__":
    app.run(debug=True)
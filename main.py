from flask import Flask,jsonify,request
import numpy as np
import pickle
import sklearn

model = pickle.load(open('model.pkl','rb'))

main = Flask(__name__)

@main.route('/')
def home():
    return "Hello World"

@main.route('/predict',methods = ['POST'])
def predict():
    spx = request.form.get('spx')
    uso = request.form.get('uso')
    slv = request.form.get('slv')
    c_pair = request.form.get('c_pair')

    m = {'spx':spx,'uso':uso,'slv':slv,'c_pair':c_pair}

    input_query = np.array([[spx,uso,slv,c_pair]])
    res = model.predict(input_query)[0]


    return jsonify({'Price':str(res)})



if __name__ == '__main__':
    main.run(debug=True)




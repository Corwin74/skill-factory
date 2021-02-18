from flask import Flask, request, jsonify
import datetime, pickle, numpy as np

with open('hw1.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/hello')
def hello_func():
    name = request.args.get('name')
    return f'hello {name}!'

@app.route('/time') 
def current_time():     
    return {'time': datetime.datetime.now()}

@app.route('/predict', methods=['POST'])
def predict_func():
    num = request.json
    num = np.array(num).reshape(1,-1)
    y = model.predict(num)
    return  jsonify({
        'prediction': y[0]
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
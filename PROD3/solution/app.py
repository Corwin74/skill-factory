from flask import Flask, request, jsonify
import datetime, pickle

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
    num = request.json.get('num')
    print(request.json)
    return  jsonify({
        'prediction': 0.25
        })

if __name__ == '__main__':
    app.run('localhost', 5000)
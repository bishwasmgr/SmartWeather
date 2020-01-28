from flask import Flask, render_template, request
from ml import *
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
  y = Engine()
  y.trainModel()
  print ('saved')
  return render_template('index.html')

@app.route('/train', methods=['POST', 'GET'])
def train():
  y = Engine()
  accuracy, correlation, actual_vs_predicted = y.trainModel()

  return json.dumps({'accuracy':round(accuracy*100, 2), 'correlation' : correlation, 'actual_vs_predicted': actual_vs_predicted})

@app.route('/predict', methods=['POST'])
def predict():
  data=json.loads(request.data)
  min_temp = data['min_temp']
  precip = data['precip']
  return str(Engine().predict(min_temp, precip))

  pass
if __name__ == '__main__':
  app.run(debug=True)
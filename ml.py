import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

class DataClean():

  def __init__(self, x_columns, y_column):
    self.dataset = pd.read_csv('Weather.csv')[x_columns + y_column]
    self.x_attributes = x_columns
    self.y_attributes = y_column

  def cleanData(self):
    self.dataset['Precip'] = self.dataset['Precip'].replace('T','0')
    columns_to_be_in_float = self.x_attributes+self.y_attributes
    self.dataset[columns_to_be_in_float] = self.dataset[columns_to_be_in_float].astype(float)

  def data(self):
    self.cleanData()
    return self.dataset


class Engine():
  def __init__(self):
    self.x_attributes = ['MinTemp', 'Precip']
    self.y_attribute = ['MaxTemp']
    self.dataset = DataClean(self.x_attributes,self.y_attribute)
    self.dataset = self.dataset.data()
    self.trained_model = ''

  def trainModel(self):
    predictor = Predictor(self.x_attributes, self.y_attribute, self.dataset)
    accuracy, correlation, actual_vs_predicted =predictor.train()
    predictor.saveModel()
    #self.trained_model = predictor.get_model()
    return accuracy, correlation, actual_vs_predicted

  @staticmethod
  def predict(min_temp, precip):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    min_temp = float(min_temp)
    precip = float(precip)
    x_values = np.array([[precip, min_temp]])

    y_pred = list(loaded_model.predict(x_values))
    return round(y_pred[0][0],2)


class Predictor():
  def __init__(self, x_attributes, y_attribute, dataset):
    self.x_attributes = x_attributes
    self.y_attribute = y_attribute
    self.dataset = dataset


  def train(self):
    X = self.dataset[self.x_attributes].values
    y = self.dataset[self.y_attribute].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    self.regressor = LinearRegression()
    self.regressor.fit(X_train, y_train)
    y_pred = self.regressor.predict(X_test)
    return self.regressor.score(X_test, y_test), self.dataset.corr().to_html(), self.actualVsPredicted(y_test, y_pred)


  def saveModel(self):
    filename = 'finalized_model.sav'
    pickle.dump(self.regressor, open(filename, 'wb'))

  def actualVsPredicted(self,actual, predicted):
    df = pd.DataFrame({'Actual': actual.flatten(), 'Predicted': predicted.flatten()})
    return df.head(10).to_html()
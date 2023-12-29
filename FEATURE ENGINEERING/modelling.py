import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class ClassificationModel():
    
    def __init__(self, test_size = 0.2) -> None:
        self.model = RandomForestClassifier() 
        self.test_size = test_size
        self.hyperparameters = None
        pass

    def train_test_split(self, X,y, test_size):

        ''' Implementing train-test split logic '''
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= 0)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

        
        return train_set, test_set
    
    def preprocess(self, X):

        '''Scaling data'''
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X
    
    def fit(self, train_set): 

        ''' Performs training logic for any model. First pre-processing, 
        and then training. '''
        
        X_train, y_train = train_set
        X_train = self.preprocess(X_train)     
        self.model.fit(X_train, y_train)

        return self.model
        
    def predict(self, X):
        ''' Makes predictions on new data '''
        X = self.preprocess(X)
        return self.model.predict(X)
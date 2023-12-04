# backend.py
from PyQt5.QtCore import QObject, pyqtSignal
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class EEGModel(QObject):
    resultReady = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def process_eeg_signal(self, eeg_data):
            
            eeg_data = eeg_data.drop(columns = ["Unnamed"])

            # Separate Data into Features and Target
            X = eeg_data.drop(columns = ['y'])
            y = eeg_data['y']
            
            # Get Columns
            cols = list(X.columns)
            
            
            print(X.describe())
            
            labels = ['no seizure','seizure']

            seizure = y[y.values == 1].shape[0]
            no_seizure = y[y.values > 1].shape[0]

            colors = ['blue', 'red']
            
            sn.barplot(x=labels, y=[no_seizure,seizure], palette=colors)
            plt.show()
            
            # Make the dataset into a binary classification problem
            y = y.values
            y[y>1] = 0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            
            print(y)
            print(X_train.shape, X_test.shape)
            print(y_train.shape, y_test.shape)

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


            X_train = pd.DataFrame(X_train, columns=[cols])
            X_test = pd.DataFrame(X_test, columns=[cols])
            
            print(X_train.describe())
            
            svc=SVC() 
            
            # Fit classifier to training set
            svc.fit(X_train,y_train)
            
            # Make predictions on test set - unseen daa
            y_pred = svc.predict(X_test)
            
            print(f'y predict = {y_pred}')
            print(f'y test = {y_test}')
            
            print(f'Accuracy score: {accuracy_score(y_test, y_pred):0.4f}')
            
            # Confusion Matrix
            cf_matrix = confusion_matrix(y_test,y_pred)
            print(cf_matrix)
            
            plt.figure(figsize=(4,3))
            plt.title('Confusion Matrix for Seizure Dataset')
            sn.heatmap(cf_matrix, annot=True, fmt=' ', cmap='Reds')
            plt.show()
            
            print(classification_report(y_test, y_pred))
            
            joblib.dump(svc, 'svm_model.pkl')
            
            X_test['y'] = y_test
            return X_test
            

    def detect_epilepsy(self, svc, data):
        
        y_predict = svc.predict(data)
        print(f'y_predict{y_predict}')
        
        if y_predict == 1:
            result = 'YES'
        else:
            result = 'NO'
            
        return result

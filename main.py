# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import joblib
from sklearn.discriminant_analysis import StandardScaler
from EEG_model import EEGModel
from PyQt5.uic import loadUiType
import pandas as pd
from os import path
import firebase_admin
from firebase_admin import credentials, db


FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "eegdesign.ui"))

class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        # Firebase configuration
        cred = credentials.Certificate('path/to/your/firebase/credentials.json')
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://your-firebase-database-url.firebaseio.com'})

        
        # Initialize the EEG model
        self.eeg_model = EEGModel()
        self.input = None
        self.eeg_data = None

        # Connect signals and slots
        self.importButton.clicked.connect(self.run_detection)
        self.importButton_2.clicked.connect(self.update_result)
        
        

    def upload_file(self):

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        filters = "CSV and DAT Files (*.csv *.dat)"
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileNames()", "", filters, options=options)

        if file_path:
            # Store file name
            file_name = file_path.split('/')[-1]
            self.file_label.setText(file_name)
            
            data = pd.read_csv(file_path)
            
            return data
    
    def run_detection(self):
        # Get EEG data from the user input (you need to implement this)
        self.eeg_data = self.upload_file()

        # Process EEG data using the model
        self.input = self.eeg_model.process_eeg_signal(self.eeg_data)
        
        self.importButton_2.setEnabled(True)

    def update_result(self):
   
        # Random Row for Input
        random_row = self.input.sample(n=1)
        random_row = random_row.sort_index(axis=1)
        X = random_row.drop(columns = ['y'])
        y = random_row['y']
        print(y)
        
        loaded_svm_model = joblib.load('svm_model.pkl')
        # Update the GUI with the result
        
        result = self.eeg_model.detect_epilepsy(loaded_svm_model, X)
        self.detection_label.setText(result)

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()

import csv
from sklearn.model_selection import train_test_split

class Reader:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        
    def read(self):
        # Read data in from file
        with open(self.filename) as f:
            self.reader = csv.reader(f)
            next(self.reader)

            #self.data = []
            for row in self.reader:
                self.data.append({
                    "features": [float(cell) for cell in row[:4]],
                    "target": 1 if row[4] == "0" else 0
                })
                
    def get_split_data(self):
        # Separate data into training and testing groups
        features = [row["features"] for row in self.data]
        targets = [row["target"] for row in self.data]

        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.4)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)
        
        return X_train, X_test, X_valid, y_train, y_test, y_valid
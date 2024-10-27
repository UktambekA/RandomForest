import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

    def prepare_data(self):
        X = self.train_df.drop(['smoking', 'id'], axis=1)
        y = self.train_df['smoking']
        X_test_final = self.test_df.drop(['id'], axis=1)
        test_ids = self.test_df['id']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_val, y_train, y_val, X_test_final, test_ids

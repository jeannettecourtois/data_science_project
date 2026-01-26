from dataclasses import dataclass, field 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def get_data_with_pandas()->pd.DataFrame:
    df = pd.read_table("data/prostate.txt", sep=" ")
    return df

@dataclass 

class LinearRegression: 
    target : str = "lpsa"
    data_variable : pd.DataFrame = field(default_factory = get_data_with_pandas)
    
    def most_correlated_variable(self):
        for col in self.data_variable.columns:
            if col != "age": 
                new_col = "l" + col
                self.data_variable[new_col] = np.log(self.data_variable[col])
        correlations = self.data_variable.corr()[self.target].sort_values(ascending=False)
        return f"Correlations with {self.target}:\n {correlations}"
    
    def first_graphic(self):
        self.data_variable.plot(kind="scatter", x="lvol", y = "lpsa", figsize=(9, 9), color="black")
        regression_model = linear_model.LinearRegression()
        # Train the model using the actual data
        regression_model.fit(X = pd.DataFrame(self.data_variable["lvol"]), 
        y = self.data_variable["lpsa"])
        train_prediction = regression_model.predict(X = pd.DataFrame(self.data_variable["lvol"]))
        # Actual - prediction = residuals
        residuals = self.data_variable["lvol"] - train_prediction
        residuals.describe()
        plt.plot(self.data_variable["lvol"],      # Explanitory variable
        train_prediction,color="blue")
        plt.show()



    
        
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass, field
from typing import Type
import seaborn as sns


def get_data_with_pandas()->pd.DataFrame:
    df = pd.read_table("data/prostate.txt", sep=" ")
    return df


@dataclass
class Variable:
    variable_name : str = "age"
    data_variable : pd.DataFrame = field(default_factory = get_data_with_pandas)
    
    def __str__(self):
        return f"{self.get_mean()}\n {self.get_median()} \n {self.standard_deviation()} \n {self.variance()} \n {self.range()} \n {self.skewness()}"
    
    def get_mean(self) -> str:
        return f"Mean of {self.variable_name}: {self.data_variable[self.variable_name].mean()}"
    
    def get_median(self)->str:
        return f"Median of {self.variable_name}: {self.data_variable[self.variable_name].median()}"
    
    def get_mode(self)->str:
        return f"Mode of {self.variable_name}: {self.data_variable[self.variable_name].mode()[0]}"
    
    def standard_deviation(self)->str:
        return f"Standard Deviation of {self.variable_name}: {self.data_variable[self.variable_name].std()}"
    
    def variance(self)->str:
        return f"Variance of {self.variable_name}: {self.data_variable[self.variable_name].var()}"
    
    def range(self)->str:
        return f"Range of {self.variable_name}: {self.data_variable[self.variable_name].max() - self.data_variable[self.variable_name].min() }"
    
    def skewness(self)->str:
        return f"Skewness of {self.variable_name}: {self.data_variable[self.variable_name].skew()}"
    

    def all_figures(self):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        figure = Figure(self.variable_name, self.data_variable)
        figure.histogram(ax=axes[0])
        figure.box_plot(ax=axes[1])
        figure.kde_density_plot(ax=axes[2])
        plt.tight_layout()
        plt.show()
        fig.savefig(f"{self.data_variable} figure")
        
        
        
        
        
    
    

@dataclass
class Figure:
    data_name : str
    data_info : pd.DataFrame

    
    def histogram(self, ax:int):
        sns.histplot(self.data_info[self.data_name].dropna(), kde=False, bins=20, ax=ax)
        ax.set_title(f"{self.data_name} distribution")
        return ax
    def box_plot(self, ax):
        sns.boxplot(x = self.data_info[self.data_name], ax=ax)
        ax.set_title(f"{self.data_name} box plot")
        return ax
    def kde_density_plot(self, ax):
        sns.kdeplot(self.data_info[self.data_name].dropna(), fill=True, ax=ax)
        ax.set_title(f"{self.data_name} Density Plot")
        return ax
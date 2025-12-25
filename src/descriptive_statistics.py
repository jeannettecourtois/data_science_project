import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass, field
from typing import Type


def get_data_with_pandas()->pd.DataFrame:
    df = pd.read_table("data/prostate.txt", sep=" ")
    return df


@dataclass
class Variable:
    variable_name : str = "age"
    data_variable : pd.DataFrame = field(default_factory = get_data_with_pandas)
    
    def get_mean(self) -> str:
        return f"Mean {self.variable_name}: {self.data_variable[self.variable_name].mean()}"


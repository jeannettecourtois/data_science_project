import pandas as pd 
from src.descriptive_statistics import Variable 









def main():
    Age_Variable = Variable()
    print(Age_Variable.get_mean())

if __name__ == "__main__":
    main()
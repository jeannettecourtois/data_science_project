import pandas as pd 
from src.descriptive_statistics import Variable 





def main():
    try:
        
        Age_Variable = Variable()
        print(Age_Variable)
        print(Age_Variable.all_figures())

    except OSError as err:
        print("OS error:", err)
        
    

if __name__ == "__main__":
    main()
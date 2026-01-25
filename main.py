import pandas as pd 
from src.descriptive_statistics import Variable 
from src.PCA import PCATD




def main():
    try:
        
        # Age_Variable = Variable()
        # print(Age_Variable)
        # print(Age_Variable.all_figures())
        
        # Bh_Variable = Variable("bh")
        # print(Bh_Variable)
        # print(Bh_Variable.all_figures())
        
        # Wht_Variable = Variable("wht")
        # print(Wht_Variable)
        # print(Wht_Variable.all_figures())
        
        # Volume_Variable = Variable("vol")
        # print(Volume_Variable)
        # print(Volume_Variable.all_figures())
        
        # Pc_Variable = Variable("pc")
        # print(Pc_Variable)
        # print(Pc_Variable.all_figures())
        
        # PSA_Variable = Variable("psa")
        # print(PSA_Variable)
        # print(PSA_Variable.all_figures())
        
        td = PCATD()
        print(td.variances())           # Q1
        td.fit()                        # Q2/Q3
        td.print_pve()                  # Q3
        td.scree_plot()                 # Q3
        td.cumulative_plot()            # Q3
        print(td.loadings.iloc[:, :2])  # Q2 (first two loading vectors)
        td.correlation_circle(1, 2)     # Q4
        td.scores_plot(1, 2)   
          
        
        

    except OSError as err:
        print("OS error:", err)
        
    

if __name__ == "__main__":
    main()
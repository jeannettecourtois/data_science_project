import pandas as pd 
from src.descriptive_statistics import Variable 
from src.PCA import PCATD
from src.linear_regression import LinearRegression




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
        
        # td = PCATD(target_col='psa', log_transform=True)  # log sauf age
        # print(td.variances())
        # td.fit()
        # td.print_pve()
        # td.scree_plot()
        # td.cumulative_plot()
        # td.scores_plot(1, 2)
        # td.correlation_circle(1, 2)
        # print(td.loadings)
        li = LinearRegression()
        print(li.most_correlated_variable())
        li.first_graphic()
        
        
        
        
 
          
        
        

    except OSError as err:
        print("OS error:", err)
        
    

if __name__ == "__main__":
    main()
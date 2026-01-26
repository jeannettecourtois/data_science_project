import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field #creating datas simpler, field indicates the field will be calculated later
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA  # éviter conflit de nom
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns 


def get_data_with_pandas() -> pd.DataFrame:
    # Lecture du fichier avec espaces variables entre colonnes
    return pd.read_table("data/prostate.txt", delim_whitespace=True)


@dataclass
class PCATD:
   

    df: pd.DataFrame = field(default_factory=get_data_with_pandas)

   
    target_col: str | int | None = None

    
    standardize: bool = True

    
    log_transform: bool = True
    age_col_name: str = "age"      
    eps: float = 1e-8              

    # Champs calculés après init
    X: pd.DataFrame = field(init=False)               
    X_std: np.ndarray = field(init=False, default=None)
    pca: SkPCA = field(init=False, default=None)
    scores: pd.DataFrame = field(init=False, default=None)
    loadings: pd.DataFrame = field(init=False, default=None)
    per_var: np.ndarray = field(init=False, default=None)
    cum_var: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        
        data = self.df.select_dtypes(include="number").copy()

        
        if self.target_col is not None:
            if isinstance(self.target_col, int):
                target_name = data.columns[self.target_col]
            else:
                target_name = self.target_col
            data = data.drop(columns=[target_name])

       
        if self.log_transform:
            for col in data.columns:
                if col != self.age_col_name:
                    
                    data[col] = np.log(data[col] + self.eps)

        
        self.X = data

    def variances(self) -> pd.Series:
        
        return self.X.var(ddof=1)

    def fit(self, n_components: int | None = None):
        
       
        X_values = self.X.values

        
        if self.standardize:
            self.X_std = StandardScaler().fit_transform(X_values)
            X_for_pca = self.X_std
        else:
            X_for_pca = X_values

        #PCA
        self.pca = SkPCA(n_components=n_components)
        self.pca.fit(X_for_pca)

        # PVE (% variance explained)
        self.per_var = np.round(self.pca.explained_variance_ratio_ * 100, 1)
        self.cum_var = np.round(np.cumsum(self.per_var), 1)

        # 4) Scores 
        Phi = self.pca.transform(X_for_pca)
        labels = [f"PC{i+1}" for i in range(Phi.shape[1])]
        self.scores = pd.DataFrame(Phi, columns=labels, index=self.X.index)

        #Loadings 
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.X.columns,
            columns=labels
        )

    def print_pve(self):
        
        print("PVE (%):", self.per_var)
        print("Cumulative PVE (%):", self.cum_var)

    def scree_plot(self):
        
        labels = [f"PC{i+1}" for i in range(len(self.per_var))]
        x = np.arange(1, len(self.per_var) + 1)

        plt.figure()
        plt.plot(x, self.per_var, marker="o")
        plt.xticks(x, labels)
        plt.ylabel("Percentage of variance explained (PVE)")
        plt.xlabel("Principal component")
        plt.title("Scree plot")
        plt.show()

    def cumulative_plot(self):
        
        labels = [f"PC{i+1}" for i in range(len(self.cum_var))]
        x = np.arange(1, len(self.cum_var) + 1)

        plt.figure()
        plt.plot(x, self.cum_var, marker="o")
        plt.xticks(x, labels)
        plt.ylabel("Cumulative PVE (%)")
        plt.xlabel("Principal component")
        plt.title("Cumulative explained variance")
        plt.show()

    def scores_plot(self, pcx: int = 1, pcy: int = 2):
        
        if self.scores is None:
            raise RuntimeError("Call fit() first.")

        xlab = f"PC{pcx}"
        ylab = f"PC{pcy}"

        plt.figure(figsize=(7, 7))
        plt.scatter(self.scores[xlab], self.scores[ylab])
        plt.axhline(0, color="silver", linewidth=1)
        plt.axvline(0, color="silver", linewidth=1)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f"Scores plot: {xlab} vs {ylab}")
        plt.show()

    def correlation_circle(self, pcx: int = 1, pcy: int = 2):
        
        if self.loadings is None:
            raise RuntimeError("Call fit() first.")

        i = pcx - 1
        j = pcy - 1

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # axes
        ax.axhline(0, color="silver", linewidth=1)
        ax.axvline(0, color="silver", linewidth=1)

        # circle 
        circle = plt.Circle((0, 0), 1, fill=False)
        ax.add_artist(circle)

        # arrows for each variable 
        for var in self.loadings.index:
            x = self.loadings.loc[var].iloc[i]
            y = self.loadings.loc[var].iloc[j]
            ax.arrow(0, 0, x, y, head_width=0.03, width=0.002)
            ax.text(x * 1.05, y * 1.05, var)

        ax.set_title(f"Correlation circle (PC{pcx} vs PC{pcy})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()



    
    
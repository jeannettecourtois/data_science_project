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
    """
    Cette classe fait le PCA comme dans le TD :
    1) (optionnel) transforme les variables par ln (log naturel) sauf 'age'
    2) calcule variances
    3) standardise (centrer-réduire)
    4) PCA
    5) scree plot, cumul, scores plot, correlation circle
    """

    df: pd.DataFrame = field(default_factory=get_data_with_pandas)

    # si tu as une variable cible à exclure du PCA (ex: "lpsa"), mets son nom ici
    target_col: str | int | None = None

    # standardiser avant PCA (souvent obligatoire si unités différentes)
    standardize: bool = True

    # appliquer ln sur toutes les variables sauf age
    log_transform: bool = True
    age_col_name: str = "age"      # nom de la colonne à ne pas log-transformer
    eps: float = 1e-8              # petit nombre pour éviter log(0)

    # Champs calculés après init
    X: pd.DataFrame = field(init=False)               # variables finales utilisées pour PCA
    X_std: np.ndarray = field(init=False, default=None)
    pca: SkPCA = field(init=False, default=None)
    scores: pd.DataFrame = field(init=False, default=None)
    loadings: pd.DataFrame = field(init=False, default=None)
    per_var: np.ndarray = field(init=False, default=None)
    cum_var: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        """
        Cette fonction s’exécute automatiquement après la création de l’objet PCATD().
        Elle prépare la matrice X utilisée par le PCA.
        """
        # 1) On ne garde que les colonnes numériques (PCA nécessite du numérique)
        data = self.df.select_dtypes(include="number").copy()

        # 2) On enlève la variable cible du PCA si elle est spécifiée
        #    Exemple : si 'lpsa' est Y, on ne veut pas qu’elle soit dans X.
        if self.target_col is not None:
            if isinstance(self.target_col, int):
                target_name = data.columns[self.target_col]
            else:
                target_name = self.target_col
            data = data.drop(columns=[target_name])

        # 3) Transformation log (ln) : ln(x) pour toutes les colonnes sauf "age"
        #    Pourquoi ? Pour réduire l’asymétrie / rendre les relations plus linéaires.
        if self.log_transform:
            for col in data.columns:
                if col != self.age_col_name:
                    # Sécurité : éviter log(0) ou log(négatif)
                    # Si tu es sûr que c’est >0, tu peux faire juste np.log(data[col])
                    data[col] = np.log(data[col] + self.eps)

        # 4) On stocke la matrice finale X (DataFrame) utilisée par PCA
        self.X = data

    def variances(self) -> pd.Series:
        """
        Question 1 du TD :
        On calcule la variance de chaque variable.
        Si les variances sont très différentes => standardiser avant PCA.
        """
        return self.X.var(ddof=1)

    def fit(self, n_components: int | None = None):
        """
        Question 2/3 :
        - Standardisation (centrer-réduire) si standardize=True
        - Fit PCA
        - Calcule PVE et cumul
        - Calcule scores (projection des individus)
        - Calcule loadings (vecteurs propres / contributions des variables)
        """
        # Matrice numpy des variables
        X_values = self.X.values

        # 1) Standardisation
        if self.standardize:
            self.X_std = StandardScaler().fit_transform(X_values)
            X_for_pca = self.X_std
        else:
            X_for_pca = X_values

        # 2) PCA
        self.pca = SkPCA(n_components=n_components)
        self.pca.fit(X_for_pca)

        # 3) PVE (% variance expliquée)
        self.per_var = np.round(self.pca.explained_variance_ratio_ * 100, 1)
        self.cum_var = np.round(np.cumsum(self.per_var), 1)

        # 4) Scores = coordonnées des individus dans l’espace PC
        Phi = self.pca.transform(X_for_pca)
        labels = [f"PC{i+1}" for i in range(Phi.shape[1])]
        self.scores = pd.DataFrame(Phi, columns=labels, index=self.X.index)

        # 5) Loadings = coefficients des variables pour construire PC1, PC2, ...
        #    pca.components_ a la forme (n_components, p)
        #    donc on transpose pour obtenir (p, n_components)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.X.columns,
            columns=labels
        )

    def print_pve(self):
        """Affiche PVE et PVE cumulée (Question 3)."""
        print("PVE (%):", self.per_var)
        print("Cumulative PVE (%):", self.cum_var)

    def scree_plot(self):
        """Scree plot (Question 3)."""
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
        """Plot variance cumulée (Question 3)."""
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
        """
        Question 4 : scores plot = projection des individus sur PC1-PC2.
        """
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
        """
        Question 4 : cercle de corrélation.
        Ici on trace les loadings sur PC1-PC2.
        Interprétation :
        - même direction => corrélation positive
        - direction opposée => corrélation négative
        - 90° => quasi pas corrélées
        - flèche longue => bien représentée par PC1-PC2
        """
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

        # cercle unité
        circle = plt.Circle((0, 0), 1, fill=False)
        ax.add_artist(circle)

        # flèches pour chaque variable
        for var in self.loadings.index:
            x = self.loadings.loc[var].iloc[i]
            y = self.loadings.loc[var].iloc[j]
            ax.arrow(0, 0, x, y, head_width=0.03, width=0.002)
            ax.text(x * 1.05, y * 1.05, var)

        ax.set_title(f"Correlation circle (PC{pcx} vs PC{pcy})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()



    
    
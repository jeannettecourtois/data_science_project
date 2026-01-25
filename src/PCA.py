import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA  # avoid name clash


def get_data_with_pandas() -> pd.DataFrame:
    # Better than sep=" " because it handles multiple spaces
    return pd.read_table("data/prostate.txt", delim_whitespace=True)


@dataclass
class PCATD:
    df: pd.DataFrame = field(default_factory=get_data_with_pandas)
    target_col: str | int | None = None      # set if you have a target to exclude from PCA
    standardize: bool = True                 # TD says standardize if variances differ a lot

    X: pd.DataFrame = field(init=False)      # numeric variables used in PCA
    X_std: np.ndarray = field(init=False, default=None)
    pca: SkPCA = field(init=False, default=None)
    scores: pd.DataFrame = field(init=False, default=None)
    loadings: pd.DataFrame = field(init=False, default=None)
    per_var: np.ndarray = field(init=False, default=None)
    cum_var: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # Keep only numeric columns
        data = self.df.select_dtypes(include="number").copy()

        # If target_col is provided, remove it from PCA variables
        if self.target_col is not None:
            if isinstance(self.target_col, int):
                target_name = data.columns[self.target_col]
            else:
                target_name = self.target_col
            data = data.drop(columns=[target_name])

        self.X = data  # dataframe of PCA variables

    # ---------- Q1: variances + interpretation ----------
    def variances(self) -> pd.Series:
        """
        Empirical variances of each variable (unbiased, ddof=1).
        Use this to decide if standardization is needed (TD Q1/Q3 logic). :contentReference[oaicite:1]{index=1}
        """
        return self.X.var(ddof=1)

    # ---------- Q2/Q3: fit PCA + PVE ----------
    def fit(self, n_components: int | None = None):
        """
        Standardize (if standardize=True), fit PCA, compute PVE, scores, loadings. :contentReference[oaicite:2]{index=2}
        """
        X_values = self.X.values

        if self.standardize:
            self.X_std = StandardScaler().fit_transform(X_values)
            X_for_pca = self.X_std
        else:
            # PCA on raw data (covariance scale)
            X_for_pca = X_values

        self.pca = SkPCA(n_components=n_components)
        self.pca.fit(X_for_pca)

        # PVE (%)
        self.per_var = np.round(self.pca.explained_variance_ratio_ * 100, 1)
        self.cum_var = np.round(np.cumsum(self.per_var), 1)

        # Scores (coordinates in PC space)
        Phi = self.pca.transform(X_for_pca)
        labels = [f"PC{i+1}" for i in range(Phi.shape[1])]
        self.scores = pd.DataFrame(Phi, columns=labels, index=self.X.index)

        # Loadings (features x components)
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            index=self.X.columns,
            columns=labels
        )

    def print_pve(self):
        """Print PVE and cumulative PVE like the TD. :contentReference[oaicite:3]{index=3}"""
        print("PVE (%):", self.per_var)
        print("Cumulative PVE (%):", self.cum_var)

    def scree_plot(self):
        """Scree plot (PVE by component) like the TD. :contentReference[oaicite:4]{index=4}"""
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
        """Cumulative PVE plot. :contentReference[oaicite:5]{index=5}"""
        labels = [f"PC{i+1}" for i in range(len(self.cum_var))]
        x = np.arange(1, len(self.cum_var) + 1)

        plt.figure()
        plt.plot(x, self.cum_var, marker="o")
        plt.xticks(x, labels)
        plt.ylabel("Cumulative PVE (%)")
        plt.xlabel("Principal component")
        plt.title("Cumulative explained variance")
        plt.show()

    # ---------- Q2/Q4: loadings + correlation circle ----------
    def correlation_circle(self, pcx: int = 1, pcy: int = 2):
        """
        Correlation circle plot (uses loadings on PCs pcx/pcy).
        Works best when standardize=True (PCA on correlation matrix). :contentReference[oaicite:6]{index=6}
        """
        if self.loadings is None:
            raise RuntimeError("Call fit() first.")

        i = pcx - 1
        j = pcy - 1

        fig, axis = plt.subplots(figsize=(5, 5))
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)

        plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
        plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)

        for var in self.loadings.index:
            x = self.loadings.loc[var].iloc[i]
            y = self.loadings.loc[var].iloc[j]
            plt.arrow(0, 0, x, y, head_width=0.02, width=0.001)
            plt.annotate(var, (x, y))

        circle = plt.Circle((0, 0), 1, fill=False)
        axis.add_artist(circle)

        plt.title(f"Correlation circle (PC{pcx} vs PC{pcy})")
        plt.show()

    # ---------- Q4: scores plot ----------
    def scores_plot(self, pcx: int = 1, pcy: int = 2, annotate: bool = False):
        """
        Plot observations in PC space like the TD. :contentReference[oaicite:7]{index=7}
        """
        if self.scores is None:
            raise RuntimeError("Call fit() first.")

        xlab = f"PC{pcx}"
        ylab = f"PC{pcy}"

        plt.figure(figsize=(7, 7))
        plt.scatter(self.scores[xlab], self.scores[ylab])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(f"Scores plot: {xlab} vs {ylab}")

        plt.plot([self.scores[xlab].min(), self.scores[xlab].max()], [0, 0],
                 color="silver", linestyle="-", linewidth=1)
        plt.plot([0, 0], [self.scores[ylab].min(), self.scores[ylab].max()],
                 color="silver", linestyle="-", linewidth=1)

        if annotate:
            for obs in self.scores.index:
                plt.annotate(obs, (self.scores.loc[obs, xlab], self.scores.loc[obs, ylab]))

        plt.show()

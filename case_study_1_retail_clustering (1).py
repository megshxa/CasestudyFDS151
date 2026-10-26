"""Case Study 1: Transaction Analysis and Clustering of an Online Retail Store
Megha Annavarapu | Roll No 160124733151 | CSE 3 | Fundamentals of Data Science
Script version of Case_Study_1_Retail_Clustering.ipynb (figures are saved to ./figures/).
"""
import os
os.makedirs("figures", exist_ok=True)

# ---- imports ----
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", font_scale=1.0)
BLUE, ORANGE = "#2a6fb0", "#e08a1e"
print("Libraries loaded.")


# 1. Understanding the dataset
# ---- load ----
raw = pd.read_csv("Online retail_new.csv", header=None)
print("Shape (transactions x max basket width):", raw.shape)
raw.head()

# ---- structure ----
print("Data types:")
print(raw.dtypes.value_counts(), "\n")
print("Total cells:", raw.size, "| filled cells:", int(raw.notna().sum().sum()),
      "| empty cells:", int(raw.isna().sum().sum()))
print(f"Share of empty cells: {raw.isna().mean().mean():.1%}\n")
print("Non-null count per column position (how many baskets reach that many items):")
print(raw.notna().sum().to_string())


# 2. Cleaning and preprocessing with Pandas
# ---- clean ----
def clean_row(row):
    items = [str(x).strip().lower() for x in row if pd.notna(x) and str(x).strip() != ""]
    return list(dict.fromkeys(items))          # drop repeats, keep order

transactions = [clean_row(r) for r in raw.values.tolist()]

raw_unique = pd.unique(raw.stack().astype(str))
clean_unique = sorted({i for t in transactions for i in t})
print("Unique product labels before cleaning:", len(raw_unique))
print("Unique products after cleaning       :", len(clean_unique))
print("Baskets containing a repeated item (before dedup):",
      sum(len([x for x in r if pd.notna(x)]) != len(clean_row(r)) for r in raw.values.tolist()))
print("Empty baskets:", sum(len(t) == 0 for t in transactions))

df = pd.DataFrame({"items": transactions})
df["basket_size"] = df["items"].str.len()
df["basket_key"] = df["items"].apply(lambda t: tuple(sorted(t)))
n_dup = df["basket_key"].duplicated().sum()
print(f"Transactions: {len(df)} | exact duplicate baskets (same item set): {n_dup} "
      f"({n_dup/len(df):.1%}); distinct baskets: {df['basket_key'].nunique()}")


# 3. Descriptive statistics and frequent products
# ---- desc ----
print("Basket-size statistics (items per transaction):")
print(df["basket_size"].describe().round(2).to_string())
print("\nMode:", df["basket_size"].mode().tolist(),
      "| Skewness:", round(df["basket_size"].skew(), 2),
      "| Kurtosis:", round(df["basket_size"].kurt(), 2))
print("Baskets with 1 item:", f"{(df.basket_size==1).mean():.1%}",
      "| with >= 10 items:", f"{(df.basket_size>=10).mean():.1%}")

freq = Counter(i for t in transactions for i in t)
freq = pd.Series(freq).sort_values(ascending=False)
freq_df = pd.DataFrame({"count": freq, "support_%": (freq / len(df) * 100).round(2)})
print("\nTotal item purchases:", int(freq.sum()), "| distinct products:", len(freq))
print(f"Top 10 products account for {freq.head(10).sum()/freq.sum():.1%} of all purchases")
print(f"Median product appears in {freq.median():.0f} baskets; the rarest appears in {freq.min()}")
freq_df.head(15)


# 4. Visualisation
# ---- plot_top ----
top = freq_df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top.index, top["count"], color=BLUE)
for y, (c, s) in enumerate(zip(top["count"], top["support_%"])):
    ax.text(c + 15, y, f"{s:.1f}%", va="center", fontsize=9)
ax.set_title("Top 15 Most Frequently Purchased Products")
ax.set_xlabel("Number of transactions containing the product")
ax.set_ylabel("Product")
ax.set_xlim(0, top["count"].max() * 1.12)
plt.tight_layout(); plt.savefig("figures/fig1_top_products.png", dpi=150); plt.show()

# ---- plot_size ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
sizes = df["basket_size"]
axes[0].hist(sizes, bins=np.arange(0.5, sizes.max() + 1.5, 1), color=BLUE, edgecolor="white")
axes[0].axvline(sizes.mean(), color=ORANGE, ls="--", label=f"Mean = {sizes.mean():.2f}")
axes[0].axvline(sizes.median(), color="black", ls=":", label=f"Median = {sizes.median():.0f}")
axes[0].set_title("Distribution of Transaction (Basket) Size")
axes[0].set_xlabel("Number of distinct items in the basket")
axes[0].set_ylabel("Number of transactions")
axes[0].legend()
axes[1].boxplot(sizes, vert=True, patch_artist=True, boxprops=dict(facecolor="#bcd4ec"),
                medianprops=dict(color=ORANGE))
axes[1].set_title("Box Plot of Basket Size")
axes[1].set_ylabel("Items per basket"); axes[1].set_xticks([])
plt.tight_layout(); plt.savefig("figures/fig2_basket_size.png", dpi=150); plt.show()
q1, q3 = sizes.quantile([.25, .75]); upper = q3 + 1.5 * (q3 - q1)
print(f"IQR upper fence = {upper:.1f}; baskets above it: {(sizes > upper).sum()} ({(sizes > upper).mean():.1%})")

# ---- plot_long ----
fig, ax = plt.subplots(figsize=(8, 4))
rank = np.arange(1, len(freq) + 1)
ax.plot(rank, freq.values, color=BLUE)
ax.set_yscale("log")
ax.set_title("Product Popularity (Rank vs Purchase Count, log scale)")
ax.set_xlabel("Product rank (1 = most purchased)")
ax.set_ylabel("Number of transactions (log scale)")
plt.tight_layout(); plt.savefig("figures/fig3_rank_frequency.png", dpi=150); plt.show()


# 5. Representing transactions as numerical feature vectors
# ---- vectors ----
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["items"]).astype(float)
products = np.array(mlb.classes_)
print("Feature matrix shape (transactions x products):", X.shape)
print(f"Sparsity (share of zeros): {1 - X.mean():.2%}")
print("Average ones per row:", X.sum(1).mean().round(2))
pd.DataFrame(X[:5, :8], columns=products[:8]).astype(int)


# 6. Principal Component Analysis (PCA)
# ---- pca_full ----
pca_full = PCA(random_state=RANDOM_STATE).fit(X)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n50, n70, n90 = [int(np.argmax(cum >= t)) + 1 for t in (0.5, 0.7, 0.9)]
print("Variance explained by PC1..PC5:", np.round(evr[:5] * 100, 2), "%")
print(f"Components needed -> 50%: {n50} | 70%: {n70} | 90%: {n90} (of {X.shape[1]})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(1, 21), evr[:20] * 100, color=BLUE)
axes[0].set_title("Scree Plot: Variance Explained by Each Component")
axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("Explained variance (%)")
axes[1].plot(range(1, len(cum) + 1), cum * 100, color=BLUE)
for t in (50, 70, 90):
    axes[1].axhline(t, color="grey", ls=":", lw=0.8)
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of components"); axes[1].set_ylabel("Cumulative variance (%)")
plt.tight_layout(); plt.savefig("figures/fig4_pca_variance.png", dpi=150); plt.show()

# ---- pca2 ----
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X2 = pca2.fit_transform(X)
print("Variance explained by PC1 + PC2:", f"{pca2.explained_variance_ratio_.sum():.1%}")
load = pd.DataFrame(pca2.components_.T, index=products, columns=["PC1", "PC2"])
print("\nTop positive loadings on PC1:", load["PC1"].nlargest(5).round(2).to_dict())
print("Top positive loadings on PC2:", load["PC2"].nlargest(5).round(2).to_dict())

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(X2[:, 0], X2[:, 1], c=df["basket_size"], cmap="viridis", s=6, alpha=0.5, vmax=10)
plt.colorbar(sc, label="Basket size (capped at 10 for colour)")
ax.set_title("Transactions Projected onto the First Two Principal Components")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} of variance)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} of variance)")
plt.tight_layout(); plt.savefig("figures/fig5_pca_scatter.png", dpi=150); plt.show()


# 7. Clustering
# ---- zspace ----
pca_c = PCA(n_components=n50, random_state=RANDOM_STATE)
Z = pca_c.fit_transform(X)
print("Clustering space:", Z.shape, f"(variance retained: {pca_c.explained_variance_ratio_.sum():.1%})")

def evaluate(Z, labels):
    """Silhouette (higher better) and Davies-Bouldin (lower better); noise (-1) excluded."""
    m = labels != -1
    if len(set(labels[m])) < 2:
        return np.nan, np.nan
    return silhouette_score(Z[m], labels[m]), davies_bouldin_score(Z[m], labels[m])

# ---- kmeans_k ----
ks = range(2, 11)
rows = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    s, d = evaluate(Z, km.labels_)
    rows.append((k, km.inertia_, s, d))
kdf = pd.DataFrame(rows, columns=["k", "inertia", "silhouette", "davies_bouldin"]).set_index("k")
print(kdf.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col, title, yl in zip(axes, kdf.columns,
        ["Elbow Plot (Inertia)", "Silhouette vs k", "Davies-Bouldin vs k"],
        ["Within-cluster sum of squares", "Silhouette coefficient", "Davies-Bouldin index"]):
    ax.plot(kdf.index, kdf[col], marker="o", color=BLUE)
    ax.set_title(title); ax.set_xlabel("Number of clusters k"); ax.set_ylabel(yl)
plt.tight_layout(); plt.savefig("figures/fig6_kmeans_selection.png", dpi=150); plt.show()

# ---- kmeans_final ----
k_sil = int(kdf["silhouette"].idxmax())
print("k with maximum silhouette:", k_sil, "| silhouette range over k=2..10:",
      round(kdf.silhouette.min(), 3), "-", round(kdf.silhouette.max(), 3))
# Silhouette is almost flat across k, and k=2 is just "small vs large baskets".
# k = 5 is used for all methods: Davies-Bouldin has improved to ~2.0 by k=5, the elbow
# in inertia flattens after it, and five groups are still few enough to act on.
K = 5
print("k used for K-means, hierarchical and spectral clustering:", K)
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Z)
lab_km = km.labels_
print("Cluster sizes:", dict(zip(*np.unique(lab_km, return_counts=True))))

def profile(labels, top_n=5, min_support=0.05):
    """Describe each cluster: size, mean basket size and items with highest lift."""
    base = X.mean(0)
    out = []
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        sup = X[m].mean(0)
        lift = sup / base
        ok = sup >= min_support
        idx = np.argsort(-(lift * ok))[:top_n]
        out.append({"cluster": c, "size": int(m.sum()), "share_%": round(m.mean() * 100, 1),
                    "avg_basket": round(df.loc[m, "basket_size"].mean(), 2),
                    "characteristic items (lift)": ", ".join(f"{products[i]} ({lift[i]:.1f}x)" for i in idx if ok[i])})
    return pd.DataFrame(out).set_index("cluster")

pd.set_option("display.max_colwidth", 140)
prof_km = profile(lab_km)
prof_km

# ---- hier ----
rng = np.random.RandomState(RANDOM_STATE)
samp = rng.choice(len(Z), 800, replace=False)
Lk = linkage(Z[samp], method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
dendrogram(Lk, truncate_mode="lastp", p=30, ax=ax, color_threshold=0, above_threshold_color=BLUE)
ax.set_title("Ward Dendrogram (800 sampled transactions, last 30 merges)")
ax.set_xlabel("Merged cluster (number of baskets in brackets)"); ax.set_ylabel("Ward merge distance")
plt.tight_layout(); plt.savefig("figures/fig7_dendrogram.png", dpi=150); plt.show()

res = []
for link in ["ward", "average", "complete"]:
    l = AgglomerativeClustering(n_clusters=K, linkage=link).fit_predict(Z)
    s, d = evaluate(Z, l)
    res.append((link, np.bincount(l).tolist(), round(s, 3), round(d, 3)))
print(pd.DataFrame(res, columns=["linkage", "cluster sizes", "silhouette", "davies_bouldin"]).to_string(index=False))

lab_hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z)
profile(lab_hc)

# ---- dbscan ----
min_samples = 2 * Z.shape[1]
nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
kdist = np.sort(nn.kneighbors(Z)[0][:, -1])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(kdist, color=BLUE)
ax.set_title(f"k-distance Plot (k = {min_samples}) for Choosing DBSCAN eps")
ax.set_xlabel("Transactions sorted by distance"); ax.set_ylabel(f"Distance to {min_samples}th nearest neighbour")
plt.tight_layout(); plt.savefig("figures/fig8_kdistance.png", dpi=150); plt.show()

grid = []
for eps in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    l = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    ncl = len(set(l)) - (1 if -1 in l else 0)
    s, d = evaluate(Z, l)
    grid.append((eps, ncl, round((l == -1).mean() * 100, 1), round(s, 3) if ncl > 1 else np.nan, round(d, 3) if ncl > 1 else np.nan))
gdf = pd.DataFrame(grid, columns=["eps", "clusters", "noise_%", "silhouette", "davies_bouldin"])
print(gdf.to_string(index=False))

# ---- dbscan_final ----
# Selection rule: among settings that still find >= 2 clusters, take the one with the LOWEST noise share.
# (Best silhouette would pick the smallest eps, which labels over half of all baskets as noise and
#  scores well only because the hard-to-cluster points are removed before scoring.)
cand = gdf[gdf.clusters >= 2]
eps_best = float(cand.loc[cand["noise_%"].idxmin(), "eps"])
print("eps selected:", eps_best, "| min_samples:", min_samples)
lab_db = DBSCAN(eps=eps_best, min_samples=min_samples).fit_predict(Z)
vals, cnts = np.unique(lab_db, return_counts=True)
print("Label counts (-1 = noise):", dict(zip(vals.tolist(), cnts.tolist())))
profile(lab_db)

# ---- spectral ----
res = []
store = {}
for nn_ in [10, 15, 30, 50, 100]:
    sp = SpectralClustering(n_clusters=K, affinity="nearest_neighbors", n_neighbors=nn_,
                            assign_labels="kmeans", random_state=RANDOM_STATE)
    l = sp.fit_predict(Z)
    s, d = evaluate(Z, l)
    store[nn_] = l
    res.append((nn_, np.bincount(l).tolist(), round(s, 3), round(d, 3)))
sdf = pd.DataFrame(res, columns=["n_neighbors", "cluster sizes", "silhouette", "davies_bouldin"])
print(sdf.to_string(index=False))
best_nn = int(sdf.loc[sdf["silhouette"].idxmax(), "n_neighbors"])
print("n_neighbors selected (max silhouette):", best_nn)
lab_sp = store[best_nn]
profile(lab_sp)


# 8. Comparing the algorithms
# ---- compare ----
labels = {"K-means": lab_km, "Hierarchical (Ward)": lab_hc,
          "DBSCAN": lab_db, "Spectral": lab_sp}
rows = []
for name, l in labels.items():
    s, d = evaluate(Z, l)
    ncl = len(set(l)) - (1 if -1 in l else 0)
    rows.append({"Algorithm": name, "Clusters": ncl, "Noise_%": round((l == -1).mean() * 100, 1),
                 "Largest cluster_%": round(np.bincount(l[l != -1]).max() / len(l) * 100, 1),
                 "Silhouette": round(s, 3), "Davies-Bouldin": round(d, 3)})
cmp_df = pd.DataFrame(rows).set_index("Algorithm")
print(cmp_df.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(cmp_df.index, cmp_df["Silhouette"], color=BLUE)
axes[0].set_title("Silhouette Coefficient by Algorithm (higher is better)")
axes[0].set_ylabel("Silhouette coefficient")
axes[1].bar(cmp_df.index, cmp_df["Davies-Bouldin"], color=ORANGE)
axes[1].set_title("Davies-Bouldin Index by Algorithm (lower is better)")
axes[1].set_ylabel("Davies-Bouldin index")
for ax in axes:
    ax.set_xlabel("Clustering algorithm"); ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("figures/fig9_comparison.png", dpi=150); plt.show()

# ---- grid_plot ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
for ax, (name, l) in zip(axes.ravel(), labels.items()):
    for c in sorted(set(l)):
        m = l == c
        if c == -1:
            ax.scatter(X2[m, 0], X2[m, 1], s=4, c="lightgrey", label="noise")
        else:
            ax.scatter(X2[m, 0], X2[m, 1], s=5, alpha=0.6, label=f"Cluster {c}")
    ax.set_title(name)
    ax.legend(markerscale=3, fontsize=7, loc="upper right")
for ax in axes[1]: ax.set_xlabel("PC1")
for ax in axes[:, 0]: ax.set_ylabel("PC2")
fig.suptitle("Cluster Assignments Shown on the First Two Principal Components", y=1.0)
plt.tight_layout(); plt.savefig("figures/fig10_cluster_scatter.png", dpi=150); plt.show()

# ---- size_by_cluster ----
fig, ax = plt.subplots(figsize=(8, 4))
bs = pd.DataFrame({"basket_size": df["basket_size"].values, "cluster": lab_km})
sns.boxplot(data=bs, x="cluster", y="basket_size", color="#bcd4ec", ax=ax, showfliers=False)
ax.set_title("Basket Size by K-means Cluster")
ax.set_xlabel("K-means cluster"); ax.set_ylabel("Items per basket")
plt.tight_layout(); plt.savefig("figures/fig11_size_by_cluster.png", dpi=150); plt.show()
print(bs.groupby("cluster")["basket_size"].agg(["count", "mean", "median"]).round(2))


# 9. Limitations

# 10. Business implications

# 11. Conclusion
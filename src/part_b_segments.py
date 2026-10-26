import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

# ---------- Task 1: RFM table ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = done.groupby("customer_id").agg(
    last_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("Customers:", len(rfm))
print("Recency  median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("Frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max())
print("Monetary  mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))

# ---------- Task 2: K-means for K = 2..8 ----------
cols = ["recency", "frequency", "monetary"]
X = StandardScaler().fit_transform(rfm[cols])

print("\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))

# ---------- Task 3: final K, profile, names ----------
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

prof = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)
print("\nSegment profile:\n", prof)

# name segments from their numbers (cluster numbers themselves have no order)
names = {}
premium_id = prof["mean_monetary"].idxmax()
names[premium_id] = "Premium Big Spenders"
rest = prof.drop(premium_id)
loyal_id = rest["mean_frequency"].idxmax()
names[loyal_id] = "Loyal Frequent Buyers"
rest = rest.drop(loyal_id)
sleepy_id = rest["mean_recency"].idxmax()
names[sleepy_id] = "Drifting Low-Activity Buyers"
names[rest.drop(sleepy_id).index[0]] = "Occasional Buyers"

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("\nNames:", names)

pca = PCA(n_components=2).fit(X)
print("\nPCA explained variance:", pca.explained_variance_ratio_.round(3),
      "total", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
     "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("Saved outputs/customer_segments.csv")
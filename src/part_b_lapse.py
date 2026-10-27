import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# ---------- Task 4: cohort, label, features (only orders BEFORE the cutoff) ----------
before = orders[orders["order_ts"] < CUTOFF]
done_b = before[before["status"] == "completed"]

g = done_b.groupby("customer_id").agg(
    last_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
)
g["recency"] = (CUTOFF - g["last_ts"]).dt.days
cancel = (before["status"] == "cancelled").groupby(before["customer_id"]).mean()
fav = (done_b.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])

df = (g.join(cancel.rename("cancel_rate")).join(fav.rename("fav_category"))
        .reset_index()
        .merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id"))
df["tenure"] = (CUTOFF - df["signup_date"]).dt.days

# label: lapsed = no completed order in Sept 2025
sept = orders[(orders["status"] == "completed") &
              (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]
df["lapsed"] = (~df["customer_id"].isin(set(sept["customer_id"]))).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure"]
CAT = ["city", "gender", "fav_category"]
X, y = df[NUM + CAT], df["lapsed"]
assert not X.isna().any().any(), "missing values in features"
print("No missing values in any feature column: OK")

print("Cohort:", len(df), "| lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", len(X_train), "Test:", len(X_test))

def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first"), CAT)])

def metrics(model_or_pred, X_t=None):
    pred = model_or_pred.predict(X_t) if X_t is not None else model_or_pred
    return (confusion_matrix(y_test, pred),
            precision_score(y_test, pred, zero_division=0),
            recall_score(y_test, pred, zero_division=0),
            f1_score(y_test, pred, zero_division=0))

def show(name, res):
    cm, p, r, f = res
    print(f"{name}: precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    print("  confusion matrix [[TN FP],[FN TP]]:", cm.tolist())

# ---------- Task 6a: baseline ----------
base_pred = [0] * len(y_test)
print(f"\nBaseline accuracy {accuracy_score(y_test, base_pred)*100:.1f}%  "
      f"lapsed F1 {f1_score(y_test, base_pred, zero_division=0):.2f}")

# ---------- Task 6b/c: tuning (cross-validation on TRAIN only) ----------
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_gs.fit(X_train, y_train)
print("\nLogReg best:", lr_gs.best_params_, "CV F1", round(lr_gs.best_score_, 3))

rf_pipe = Pipeline([("pre", make_pre()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                               "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_gs.fit(X_train, y_train)
print("RandForest best:", rf_gs.best_params_, "CV F1", round(rf_gs.best_score_, 3))

# ---------- Task 5: scaler learned on training rows only ----------
scaler = lr_gs.best_estimator_.named_steps["pre"].named_transformers_["num"]
print("\nScaler mean of frequency (train only):", round(scaler.mean_[NUM.index("frequency")], 3),
      "| full-cohort mean:", round(df["frequency"].mean(), 3))

# ---------- Task 6d: test-set evaluation ----------
lr_best, rf_best = lr_gs.best_estimator_, rf_gs.best_estimator_
lr_res, rf_res = metrics(lr_best, X_test), metrics(rf_best, X_test)
print()
show("LogReg test", lr_res)
show("RandForest test", rf_res)
best_model = rf_best if rf_res[3] >= lr_res[3] else lr_best
print("Selected model:", "RandomForest" if best_model is rf_best else "LogisticRegression")

# ---------- Task 7: class-balance experiment (train only) ----------
n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
extra_idx = y_train[y_train == 1].sample(n=n_extra, replace=True, random_state=42).index
X_bal = pd.concat([X_train, X_train.loc[extra_idx]])
y_bal = pd.concat([y_train, y_train.loc[extra_idx]])
print("\nBalanced train counts:", y_bal.value_counts().to_dict())
lr_bal = clone(lr_best).fit(X_bal, y_bal)
show("LogReg unbalanced", lr_res)
show("LogReg balanced  ", metrics(lr_bal, X_test))

# ---------- Task 8a: interpretation ----------
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names)
print("\nTop 5 random-forest features:\n", imp.sort_values(ascending=False).head(5).round(3))
coef = pd.Series(lr_best.named_steps["clf"].coef_[0],
                 index=lr_best.named_steps["pre"].get_feature_names_out())
print("\nLogReg coefficients:\n", coef[["num__recency", "num__frequency", "num__avg_discount"]].round(3))

# ---------- Task 8b: save and reload ----------
joblib.dump(best_model, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
same = (loaded.predict(X_test) == best_model.predict(X_test)).all()
print("\nReloaded model reproduces test predictions exactly:", bool(same))

# ---------- Task 8c: retention list ----------
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
ids = df.loc[X_test.index, "customer_id"].values
ret = pd.DataFrame({"customer_id": ids,
                    "lapse_probability": best_model.predict_proba(X_test)[:, 1],
                    "actually_lapsed": y_test.values})
ret = ret[ret["customer_id"].isin(premium_ids)].sort_values(
    "lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print("\nRetention list rows:", len(ret), "| truly lapsed:", int(ret["actually_lapsed"].sum()))
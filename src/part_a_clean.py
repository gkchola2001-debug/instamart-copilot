import os, sqlite3
import pandas as pd

RAW = "data/raw"
os.makedirs("data/clean", exist_ok=True)

customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

# ---------- AUDIT (task 2) ----------
for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n=== {name}: shape {df.shape}")
    print(df.dtypes)
    miss = pd.DataFrame({"missing": df.isna().sum(),
                         "missing_pct": (df.isna().mean() * 100).round(2)})
    print(miss)

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
audit = {
    "raw order rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(deliv.isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((customers["age"] < 10) | (customers["age"] > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct raw gender spellings": int(customers["gender"].nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT TABLE (put this in the README) ===")
for k, v in audit.items():
    print(f"{k}: {v}")

# ---------- CLEAN CUSTOMERS (task 3) ----------
c = customers.copy()
c["city"] = c["city"].str.strip().str.title()
g = c["gender"].fillna("").str.strip().str.lower().str[:1]
c["gender"] = g.map({"m": "male", "f": "female"}).fillna("unknown")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c["age"] = c["age"].where(c["age"].between(10, 100))
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("\nCities:\n", c["city"].value_counts())
print("Genders:\n", c["gender"].value_counts())
print("Missing ages:", c["age"].isna().sum(), "| min/max age:", c["age"].min(), c["age"].max())

# ---------- CLEAN ORDERS (task 4) ----------
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                       # rule 1
n0 = len(o)
o = o.drop_duplicates()                                             # rule 2
print(f"\nRule 2: removed {n0 - len(o)} exact duplicates")
n1 = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")  # rule 3
print(f"Rule 3: removed {n1 - len(o)} older records of repeated order_ids")

o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")   # rule 4
n_missing_amt = int(o["order_amount"].isna().sum())
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(calc)
print(f"Rule 4: filled {n_missing_amt} missing amounts from product price")

neg = int((o["delivery_minutes"] < 0).sum())                            # rule 5
o.loc[o["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")
comp = o["status"] == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")
n_fill = int(o.loc[comp, "delivery_minutes"].isna().sum())
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med)
print(f"Rule 5: {neg} negatives set to missing; filled {n_fill} completed orders with category median")

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("\nRows:", len(o), "| status counts:\n", o["status"].value_counts())
print("Missing amount:", o["order_amount"].isna().sum())
print("Missing delivery (completed):", o.loc[o.status == "completed", "delivery_minutes"].isna().sum())
print("Missing delivery (cancelled):", o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum())
print("Completed revenue:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---------- STORE (task 5) ----------
o["order_ts"] = o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
o.to_csv("data/clean/orders.csv", index=False)

con = sqlite3.connect("data/instamart.db")
c_out.to_sql("customers", con, if_exists="replace", index=False)
products.to_sql("products", con, if_exists="replace", index=False)
o.to_sql("orders", con, if_exists="replace", index=False)
con.close()
print("\nSaved clean CSVs and data/instamart.db")
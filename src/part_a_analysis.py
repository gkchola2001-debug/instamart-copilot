import re, sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()
names = re.findall(r"(?m)^-- (Q\d)", text)
bodies = re.split(r"(?m)^-- Q\d.*$", text)[1:]
res = {}
for n, b in zip(names, bodies):
    res[n] = pd.read_sql_query(b, con)
    print(f"\n===== {n}: {len(res[n])} rows =====")
    print(res[n].head(8).to_string(index=False))

print("\nQ1 customers:", len(res["Q1"]))
print("Q2 total:", round(res["Q2"].revenue.sum(), 2), "| rows:", len(res["Q2"]), "| top:", res["Q2"].iloc[0].tolist())
print("Q3 products:", len(res["Q3"]))
print("Q4 customers:", len(res["Q4"]))
print("Q5 customers:", len(res["Q5"]), "| average:", res["Q5"].avg_total.iloc[0])
q6 = res["Q6"]
print("Q6 top per city:\n", q6[q6.rnk == 1].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print("Q7 rows:", len(res["Q7"]), "| last cumulative:", res["Q7"].cumulative_revenue.iloc[-1])

# ---------- pandas cross-check ----------
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
done = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                  .merge(p[["product_id", "category"]], on="product_id")
pv = pd.pivot_table(done, index="city", columns="category", values="order_amount",
                    aggfunc="sum", margins=True)
print("\nPivot grand total:", round(pv.loc["All", "All"], 2), "(must equal SQL total)")

city_stats = done.groupby("city").agg(total=("order_amount", "sum"),
                                       mean=("order_amount", "mean"),
                                       count=("order_amount", "count")).round(2)
print(city_stats)

# ---------- descriptive statistics ----------
x = done["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f}")
print(f"Q1 {q1:.2f} Q3 {q3:.2f} lower fence {lo:.2f} upper fence {hi:.2f}")
print("outliers below:", int((x < lo).sum()), "above:", int((x > hi).sum()))
print(done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3))
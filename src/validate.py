import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
done = o[o.status == "completed"]

assert len(o) == 12309
assert o["order_id"].nunique() == 12309
assert (o.status == "completed").sum() == 11110
assert (o.status == "cancelled").sum() == 1199
assert not (o.status == "pending").any()
assert o["order_amount"].isna().sum() == 0
assert done["delivery_minutes"].isna().sum() == 0
assert o[o.status == "cancelled"]["delivery_minutes"].isna().sum() == 1199
assert round(done["order_amount"].sum(), 2) == 2998668.80
assert c["city"].nunique() == 6
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c["age"].isna().sum() == 0 and c["age"].between(18, 60).all()
print("ALL CHECKPOINTS PASSED")
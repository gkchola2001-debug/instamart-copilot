# Instamart Customer Intelligence & Support Copilot

## Overview
This project builds a Customer Intelligence and Support Copilot for a quick-commerce company in four parts: (A) cleaning raw order data and storing it in SQLite, (B) customer segmentation and lapse prediction, (C) support ticket triage with privacy guardrails, and (D) a policy question-answering assistant and a refund agent. All data is synthetic and all money is in INR.

## Reproduce
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/part_a_clean.py`
6. `python src/validate.py`
7. `python src/part_a_analysis.py`

## Part A

### Audit of raw data
| Check | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules
- Rule 1: order_ts converted to datetime.
- Rule 2: removed 119 exact duplicate rows.
- Rule 3: for each order_id kept only the latest record, removing the older pending records.
- Rule 4: unparseable order_amount values (120) turned into missing, then filled from price x quantity x (1 - discount/100).
- Rule 5: 106 negative delivery times set to missing; missing delivery times of completed orders filled with the median of the product category; cancelled orders left missing (1,199).

Result: 12,309 rows, 11,110 completed, 1,199 cancelled, completed revenue INR 2,998,668.80. Validation script prints ALL CHECKPOINTS PASSED.

### SQL results
| Query | Result |
|---|---|
| Q1 | 30 customers |
| Q2 | 30 rows, total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | 11 products |
| Q4 | 25 customers |
| Q5 | 550 customers, average INR 2,552.06 |
| Q6 | 8 rows where RANK differs from ROW_NUMBER |
| Q7 | 183 daily rows, last cumulative INR 2,998,668.80 |

Q6 note: RANK() gives tied customers the same rank and then skips numbers, whereas ROW_NUMBER() gives every row a unique number, so they differ whenever customers in a city have exactly the same total spend.

### Exploratory analysis
Completed quick-commerce orders across six Indian cities. Order amounts are right-skewed (mean 269.91 vs median 135.00, skewness 4.61). 897 orders lie above the upper fence of 645.30 and none below; these are mostly high-priced electronics, so they are real, not errors. The largest correlation in absolute value is order_amount with discount_pct at about -0.095, which is very weak, and correlation of this size does not establish causation.

- Mean 269.91, median 135.00, skewness 4.61
- Q1 64.80, Q3 297.00, lower fence -283.50, upper fence 645.30
- Outliers: 0 below, 897 above
- Largest correlation: order_amount vs discount_pct = -0.095

## Part B
(to be added)

## Part C
(to be added)

## Part D
(to be added)

## Limitations and Next Steps

## Part B

### RFM table
(Customers, recency median/min/max, frequency mean/max, monetary mean/max, copied from your output)

### K-means
(Table of K = 2..8 with inertia and silhouette, copied from your output.)
Chosen K = 4: the silhouette score is highest at K=4 and the inertia drops sharply from K=3 to K=4, then flattens (the elbow). Four segments are also easy for the business to act on.
Inertia always falls as K grows because every point gets a closer centre, so K=8 having lower inertia than K=2 does not make it better. K must be chosen using the elbow, silhouette and business meaning.

### Segment profile
Cluster numbers are only labels and have no ordering. Premium = the segment with the highest mean monetary value.

| Segment | Name | Size | Mean recency | Mean frequency | Mean monetary | Action |
|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | Early access to new electronics, priority support |
| 2 | Loyal Frequent Buyers | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards, subscription offers |
| 3 | Drifting Low-Activity Buyers | 182 | 109.04 | 2.20 | 685.99 | Win-back discount coupon |
| 0 | Occasional Buyers | 436 | (from your output) | (from your output) | (from your output) | Grocery bundle offers to raise order frequency |

### PCA
Explained variance ratios 0.634 and 0.218, total 0.852.
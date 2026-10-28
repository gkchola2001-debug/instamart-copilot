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
8. `python src/part_b_segments.py`
9. `python src/part_b_lapse.py`
10. `python src/part_c_triage.py`
11. `python src/part_d_rag_eval.py`

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
### RFM table
Snapshot moment 2025-10-01 00:00:00; only completed orders count. Recency is the whole days since the last completed order, frequency the number of distinct completed orders, and monetary the sum of completed order_amount (INR).

| Measure | Value |
|---|---|
| Customers with at least one completed order | 1,175 |
| Recency (median / min / max) | 19 / 0 / 180 days |
| Frequency (mean / max) | 9.46 / 33 |
| Monetary (mean / max) | INR 2,552.06 / INR 14,766.55 |

### K-means
The three RFM columns were standardised with StandardScaler, and KMeans(n_clusters=K, n_init=10, random_state=42) was fitted for K = 2 to 8.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Chosen K = 4. The inertia drops sharply from K=3 to K=4 (1268 to 743) and then flattens (617 at K=5), so the elbow is at 4, and K=4 also has the highest silhouette score (0.526). Four segments are also few enough for the business to give each one its own campaign.

Inertia is the sum of squared distances from points to their cluster centre, so adding clusters always lowers it: K=8 has lower inertia than K=2 only because every point gets a closer centre (with one cluster per customer it would be zero). That fact alone cannot pick K, so the choice also uses the elbow, the silhouette scores and business interpretability.

### Segment profile
Cluster numbers are only labels and have no ordering. The Premium segment is the one with the highest mean monetary value.

| Segment | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to new electronics and priority support |
| 2 | Loyal Frequent Buyers | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards and subscription offers |
| 3 | Drifting Low-Activity Buyers | 182 | 109.04 | 2.20 | 685.99 | Win-back discount coupon |
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Grocery bundle offers to raise order frequency |

The result is saved in outputs/customer_segments.csv (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### PCA
PCA(n_components=2) on the scaled RFM matrix gives explained variance ratios of 0.634 and 0.218, a total of 0.852.

### Lapse problem
Cutoff 2025-09-01. The cohort is every customer with a completed order before the cutoff: 1,156 customers, of whom 443 lapsed (38.32%). The split is 924 training and 232 test rows (test_size=0.2, stratified, random_state=42). Always predicting "not lapsed" gives accuracy 61.6% and lapsed-class F1 0.00.

### Preprocessing (no leakage)
- No encoder, scaler or resampler saw test rows. No data from 2025-09-01 onwards entered the features. The segment labels from the K-means step were not used as features.
- No feature column contains a missing value (checked in code).
- Scaler mean of frequency (training rows only): 8.115 versus full-cohort mean: 8.029. They differ because the scaler learned its mean from the 924 training rows only, while the cohort mean also includes the 232 test rows.
### Models and tuning
| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Logistic regression | C=100 | 0.651 | 0.644 | 0.652 | 0.648 |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.681 | 0.659 | 0.652 | 0.655 |

Confusion matrices on the test set ([[TN, FP], [FN, TP]]): logistic regression [[111, 32], [31, 58]], random forest [[113, 30], [31, 58]].

Model choice: I chose the random forest, which has the slightly higher lapsed-class F1 (0.655, above the required 0.55). Missing a customer who is about to lapse loses their future revenue, while wrongly targeting a loyal customer costs only a small offer, so missing a lapser is the costlier mistake and recall matters at least as much as precision. Both models have the same recall (0.652), and the forest has the better precision.

### Class-balance experiment
The lapsed training rows were over-sampled with replacement until both classes had 570 rows (training set only), and the best logistic regression was refit.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Unbalanced | 0.644 | 0.652 | 0.648 |
| Balanced | 0.589 | 0.854 | 0.697 |

Recall moved the most (0.652 to 0.854), while precision did not rise (0.644 to 0.589). The retention team would catch many more real lapsers but also contact more loyal customers by mistake, so this helps when campaigns are cheap.

### Interpretation
Top 5 random-forest features (importance):
- frequency 0.280: customers who order more often behave very differently from occasional buyers, so it is the strongest signal of lapsing.
- monetary 0.271: total spend before September is almost as informative as order count.
- recency 0.115: how recently the customer last ordered carries a smaller but real signal.
- avg_delivery 0.061: slower average deliveries contribute a little to lapsing.
- cancel_rate 0.052: customers who cancel more often contribute a little to lapsing.

| Feature | Coefficient | Sign | Business reading |
|---|---|---|---|
| recency | 0.040 | + | The longer since the last order, the slightly more likely the customer lapses (a small effect). |
| frequency | -1.698 | - | Customers who order often are much less likely to lapse; this is the strongest effect. |
| avg_discount | 0.004 | + | Almost no effect, because the coefficient is close to zero. |

(Business reading examples: a positive recency coefficient means the longer since the last order, the more likely the customer lapses; a negative frequency coefficient means frequent buyers are less likely to lapse.)

### Saved model and retention list
The selected model was saved with joblib to outputs/lapse_model.joblib, reloaded, and reproduced the test predictions exactly (True). The retention list (outputs/retention_list.csv) has 10 Premium test customers ranked by predicted lapse probability, and 1 of them truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label.

## Part C
### Guardrails and redaction
- Empty tickets (4) get status skipped_empty and long tickets (4, about 734 estimated tokens against MAX_INPUT_TOKENS = 400) get flagged_long. Neither is ever sent to the model.
- Redaction was applied in the order email, phone, order id before any prompt was built.

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts found 0 matches, and an assertion in the code checks every prompt before each model call.

### Prompts
A zero-shot prompt (no EXAMPLES: line) and a few-shot prompt (EXAMPLES: followed by four labelled examples, including a fraud example). Both contain the word CLASSIFY, list the four categories and require a JSON-only reply.

### Retries
Every model call is wrapped in a retry function: at most 3 retries, wait = 0.05 x 2^n x random.uniform(0.9, 1.15) seconds, one log line per retry. A full run over the 156 eligible tickets had 19 TransientLLMErrors, all recovered by a single retry, so no ticket ended with a permanent error. Example log lines (every wait lies between 0.045 and 0.0575 seconds):
- RETRY ticket=T0002 retry=0 wait=0.0530s
- RETRY ticket=T0116 retry=0 wait=0.0453s
- RETRY ticket=T0153 retry=0 wait=0.0484s

### Output validation
A reply is parsed as JSON and accepted if the category is one of the four allowed values (classified_json). If it is not JSON, lowercase words are extracted with re.findall and the reply is accepted only if exactly one allowed category word is present (classified_keyword). Anything else, including valid JSON with a category outside the list, is flagged_invalid_output and sent to human review. Nothing is guessed.

| Status | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

Rule-validated and routed automatically: 147 tickets. Sent to human review: 17 (9 invalid outputs, 4 empty, 4 too long).

### Zero-shot versus few-shot
| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 10,180 | 69.3 |
| Few-shot | 147 | 100% (147 of 147) | 25,000 | 170.1 |

Zero-shot confusion table (rows = true, columns = predicted):

| True \ Predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Every zero-shot error is a fraud ticket predicted as billing (11) or a technical ticket predicted as billing (10). The few-shot prompt costs about 100 more tokens per ticket (170.1 versus 69.3, roughly 2.5 times as many in total) but removed all 21 errors. The extra tokens are worth paying when a wrong label is costly, such as a fraud ticket landing in the billing queue instead of the security team within 2 hours. For low-risk tickets, where an 85% accurate and much cheaper zero-shot prompt is acceptable, they are not.

### Token budget demo
With the zero-shot prompt and TOKEN_BUDGET = 6000, 64 tickets ended with status budget_exhausted (routed to human review). The result is saved in outputs/triage_budget_demo.csv.

### Routing
The final outputs/triage_results.csv (164 rows) comes from a few-shot run with TOKEN_BUDGET = 100000. Rules: skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted go to human_review with priority NORMAL; fraud goes to security_team with priority HIGH; Premium customers (from the Part B segment file) go to premium_desk with priority HIGH; everyone else goes to support_queue with priority NORMAL. A customer with no row in the segment file is treated as not Premium.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priority: 119 NORMAL and 45 HIGH.
## Part D
### Chunking
Each of the five policy files was split into chunks of 40 words with an overlap of 10 words, and the source file name is stored with every chunk. Total: [15] chunks (refund [4], delivery [3], escalation [3], payment [3], privacy [2]).

### Retrieval (TF-IDF)
Chunks are vectorised with TfidfVectorizer fitted on the chunk texts, the query is transformed with the same vectorizer, and chunks are ranked by cosine similarity (retrieve(query, k=3) returns source, score and text). For the query "refund policy late delivery" the top three chunks all come from refund_policy.txt.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) measures how rare the word is across all chunks, so rare words count more than common ones. A limitation is that TF-IDF only matches words, not meaning: two phrasings of the same question with different words (for example "money back" versus "refund") can retrieve the wrong chunk.

### Augmented answer
answer(question) retrieves three chunks and sends two messages to mock_llm.chat: a system message containing POLICY ASSISTANT that tells the model to answer only from the context and to say it could not find the information otherwise, and a user message of the form CONTEXT: (three chunks, each prefixed by [file name]) QUESTION: (the question). It returns the answer text and the sources used.

### Evaluation (14 questions)
| Setting | Expected doc in top 3 (12 answerable) | Answer contains expected fact (12 answerable) | Fallback "I could not find" (2 unanswerable) |
|---|---|---|---|
| Overlap 10 | [12/12] | [12/12] | [2/2] |
| Overlap 0 | 12/12 | 10/12 | 2/2 |

With zero overlap, two answerable questions fail. "What delivery time makes a completed order eligible for a full refund?" fails because the sentence is cut at a chunk boundary, so the retrieved chunk only holds the second half ("receives a full refund of the order amount") and not the expected fact "more than 45 minutes". "How soon is money returned when a payment fails but is deducted?" fails the same way: the chunk ends at "returned automatically" and "within 48 hours" falls into the next chunk. The 10-word overlap repeats the end of each chunk at the start of the next one, so sentences cut by a boundary stay whole in at least one chunk, and with overlap 10 all 12 answers contain their facts.## Limitations and Next Steps


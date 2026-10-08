# Data

The dataset is not committed to the repo. Download it once:

**Option A: UCI (original)**
1. Go to <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
2. Download and unzip. You get `default of credit card clients.xls`.
3. Save it here as `data/default_of_credit_card_clients.xls`.

**Option B: Kaggle mirror (CSV)**
<https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset>
Save as `data/UCI_Credit_Card.csv`.

`src/features.py::load_raw` accepts `.xls`, `.xlsx` or `.csv` and normalises the
column names (`PAY_0` -> `PAY_1`, target -> `DEFAULT`), so either source works.

## Schema (30,000 rows)
| Column | Meaning |
|---|---|
| LIMIT_BAL | Credit limit (NT$) |
| SEX | 1 = male, 2 = female |
| EDUCATION | 1 grad school, 2 university, 3 high school, 4 other (undocumented: 0, 5, 6) |
| MARRIAGE | 1 married, 2 single, 3 other (undocumented: 0) |
| AGE | Years |
| PAY_1..PAY_6 | Repayment status, Sept (1) back to April (6): -1 paid duly, 1..9 months delay (undocumented: -2, 0) |
| BILL_AMT1..6 | Bill statement amount, Sept back to April |
| PAY_AMT1..6 | Amount paid, Sept back to April |
| DEFAULT | 1 = defaulted next month (target, ~22%) |

Data are from a Taiwanese issuer in 2005. See the README "Limitations" section.

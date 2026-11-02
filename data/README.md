# Data

Download `online_retail_II.csv` (95 MB) from [Kaggle: Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place it here. License: CC0 public domain. Original source: UCI Machine Learning Repository (Chen, 2019).

| Column | Meaning |
|---|---|
| `Invoice` | Invoice number; starts with `C` for a cancellation |
| `StockCode` | Product code (non-product codes such as `POST` or `M` are removed in cleaning) |
| `Description` | Product name |
| `Quantity` | Units per line (negative on cancellations) |
| `InvoiceDate` | Date and time of the invoice |
| `Price` | Unit price in pounds sterling |
| `Customer ID` | Customer number (missing on 23% of lines) |
| `Country` | Customer's country |

`src/build.py` also writes `rfm_backtest.parquet` and `rfm_current.parquet` here; they're regenerated on each run and not committed.

"""Load the raw CSV into DuckDB, clean it with sql/01_clean.sql and compute RFM tables.

Run from the project root:  python src/build.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "online_retail_II.csv"
SQL = ROOT / "sql"
REPORTS = ROOT / "reports"

# Two snapshots, one year apart. Each looks back 365 days.
BACKTEST_SNAPSHOT = "2010-12-10"   # segment here, then check the following 365 days
CURRENT_SNAPSHOT = "2011-12-10"    # the day after the last transaction: the live segmentation


def connect(raw: Path = RAW) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE raw AS SELECT * FROM read_csv(?, header = true, "
        "types = {'Invoice': 'VARCHAR', 'StockCode': 'VARCHAR', 'Customer ID': 'DOUBLE'})",
        [str(raw)],
    )
    con.execute((SQL / "01_clean.sql").read_text())
    return con


def cleaning_log(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    steps = [("Raw rows", "raw"), ("Remove exact duplicates", "s1_dedup"),
             ("Remove lines without a customer ID", "s2_customer"),
             ("Remove non-product lines and zero prices", "s3_products")]
    rows, prev = [], None
    for label, table in steps:
        n, rev = con.execute(f"SELECT count(*), sum(Quantity * Price) FROM {table}").fetchone()
        rows.append({"step": label, "rows": n, "rows_removed": None if prev is None else prev - n,
                     "net_sales_gbp": round(rev)})
        prev = n
    return pd.DataFrame(rows)


def query(con: duckdb.DuckDBPyConnection, file: str, snapshot: str) -> pd.DataFrame:
    return con.execute((SQL / file).read_text(), {"snapshot": snapshot}).df()


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    con = connect()
    log = cleaning_log(con)
    log.to_csv(REPORTS / "cleaning_log.csv", index=False)

    dup_overlap, dup_other = con.execute("""
        SELECT sum(k - 1) FILTER (WHERE InvoiceDate >= '2010-12-01' AND InvoiceDate < '2010-12-10'),
               sum(k - 1) FILTER (WHERE NOT (InvoiceDate >= '2010-12-01' AND InvoiceDate < '2010-12-10'))
        FROM (SELECT *, count(*) AS k FROM raw GROUP BY ALL HAVING count(*) > 1)""").fetchone()
    facts = con.execute("""
        SELECT count(*) lines, count(DISTINCT customer_id) customers, count(DISTINCT invoice) invoices,
               min(invoice_ts)::VARCHAR first_ts, max(invoice_ts)::VARCHAR last_ts,
               sum(is_cancel::INT) cancel_lines, round(sum(amount)) net_sales
        FROM lines""").df().iloc[0].to_dict()
    facts.update(duplicates_in_overlap_week=int(dup_overlap), duplicates_elsewhere=int(dup_other))
    (REPORTS / "data_facts.json").write_text(json.dumps(facts, indent=2, default=int))

    for name, snap in [("backtest", BACKTEST_SNAPSHOT), ("current", CURRENT_SNAPSHOT)]:
        rfm = query(con, "02_rfm.sql", snap)
        if name == "backtest":
            nxt = query(con, "03_next_year.sql", snap)
            rfm = rfm.merge(nxt, on="customer_id", how="left").fillna({"next_orders": 0, "next_revenue": 0})
        rfm.to_parquet(ROOT / "data" / f"rfm_{name}.parquet", index=False)
        print(f"{name}: {len(rfm):,} customers as of {snap}")

    print(log.to_string(index=False))
    print(facts)


if __name__ == "__main__":
    main()

-- Clean the raw Online Retail II export into one row per sales line.
-- Each step is a separate table so src/build.py can log how many rows it removes.

-- Step 1: exact duplicates. The source workbook has two yearly sheets that overlap
-- on 1-9 Dec 2010, so that week appears twice; a smaller number of lines are repeated
-- within the same invoice.
CREATE OR REPLACE TABLE s1_dedup AS
SELECT DISTINCT * FROM raw;

-- Step 2: only lines with a customer ID can be assigned to a customer.
CREATE OR REPLACE TABLE s2_customer AS
SELECT * FROM s1_dedup WHERE "Customer ID" IS NOT NULL;

-- Step 3: drop non-merchandise lines (postage, fees, manual adjustments, test items,
-- gift vouchers) and lines with no price.
CREATE OR REPLACE TABLE s3_products AS
SELECT * FROM s2_customer
WHERE Price > 0
  AND NOT regexp_matches(StockCode,
        '^(POST|DOT|M|m|C2|D|S|B|CRUK|BANK CHARGES|AMAZONFEE|ADJUST[0-9]*|TEST[0-9]*|gift_.*)$');

-- Final table. Cancellations (invoice starting with C) stay in, as negative amounts,
-- so spend is measured net of returns.
CREATE OR REPLACE TABLE lines AS
SELECT
    Invoice                          AS invoice,
    StockCode                        AS stock_code,
    Description                      AS description,
    Quantity                         AS quantity,
    InvoiceDate                      AS invoice_ts,
    Price                            AS price,
    CAST("Customer ID" AS INTEGER)   AS customer_id,
    Country                          AS country,
    Quantity * Price                 AS amount,
    Invoice LIKE 'C%'                AS is_cancel
FROM s3_products;

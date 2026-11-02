-- Recency, frequency and monetary value per customer, for the 365 days before a snapshot.
-- Parameter $snapshot: the first moment AFTER the window (e.g. 2011-12-10 00:00).
WITH window_lines AS (
    SELECT *
    FROM lines
    WHERE invoice_ts >= $snapshot::TIMESTAMP - INTERVAL 365 DAY
      AND invoice_ts <  $snapshot::TIMESTAMP
)
SELECT
    customer_id,
    date_diff('day', max(invoice_ts) FILTER (WHERE NOT is_cancel), $snapshot::TIMESTAMP)
                                                        AS recency_days,
    count(DISTINCT invoice) FILTER (WHERE NOT is_cancel) AS frequency,
    round(sum(amount), 2)                                AS monetary,       -- net of returns
    round(sum(amount) FILTER (WHERE NOT is_cancel), 2)   AS gross_sales,
    round(-coalesce(sum(amount) FILTER (WHERE is_cancel), 0), 2) AS returns,
    arg_max(country, invoice_ts)                         AS country
FROM window_lines
GROUP BY customer_id
-- Customers whose returns cancel out all purchases in the window have no value to segment.
-- Rounded to pence first: parallel float sums can leave e.g. +0.0000000001 on a fully refunded customer.
HAVING count(DISTINCT invoice) FILTER (WHERE NOT is_cancel) > 0
   AND round(sum(amount), 2) > 0
ORDER BY customer_id;

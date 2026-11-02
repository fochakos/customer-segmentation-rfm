-- What each customer did in the 365 days AFTER a snapshot (used to backtest segments).
SELECT
    customer_id,
    count(DISTINCT invoice) FILTER (WHERE NOT is_cancel) AS next_orders,
    round(sum(amount), 2)                                AS next_revenue    -- net of returns
FROM lines
WHERE invoice_ts >= $snapshot::TIMESTAMP
  AND invoice_ts <  $snapshot::TIMESTAMP + INTERVAL 365 DAY
GROUP BY customer_id;

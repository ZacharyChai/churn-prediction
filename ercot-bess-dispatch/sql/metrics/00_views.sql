-- Shared views. Expects tables: daily, params(primary_hub, power_mw), breaks(label, kind, break_date).
-- `paired` keeps only days where all three strategies exist, so capture rates compare like with like
-- (S3 has no forecast for the first day of the window or after a data gap).
CREATE OR REPLACE VIEW paired AS
SELECT hub, scenario, delivery_date,
       max(revenue_usd) FILTER (WHERE strategy = 'S1') AS s1,
       max(revenue_usd) FILTER (WHERE strategy = 'S2') AS s2,
       max(revenue_usd) FILTER (WHERE strategy = 'S3') AS s3
FROM daily
GROUP BY hub, scenario, delivery_date
HAVING count(DISTINCT strategy) = 3;

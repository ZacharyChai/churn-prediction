-- Ten highest-revenue days for the day-ahead strategy at the primary hub, base scenario.
SELECT delivery_date,
       max(revenue_usd) FILTER (WHERE strategy = 'S1') AS s1_usd,
       max(revenue_usd) FILTER (WHERE strategy = 'S2') AS s2_usd,
       max(revenue_usd) FILTER (WHERE strategy = 'S3') AS s3_usd
FROM daily
WHERE hub = (SELECT primary_hub FROM params) AND scenario = 'base'
GROUP BY delivery_date
ORDER BY s2_usd DESC NULLS LAST
LIMIT 10;

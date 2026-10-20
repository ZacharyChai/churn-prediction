-- Primary metric: monthly capture rate = sum(realised revenue) / sum(perfect-foresight revenue),
-- over days where all three strategies exist. NULL when perfect-foresight revenue is zero.
SELECT hub, scenario,
       date_trunc('month', delivery_date)::DATE AS month,
       count(*) AS days,
       sum(s1) AS s1_usd, sum(s2) AS s2_usd, sum(s3) AS s3_usd,
       sum(s2) / NULLIF(sum(s1), 0) AS capture_s2,
       sum(s3) / NULLIF(sum(s1), 0) AS capture_s3
FROM paired
GROUP BY ALL
ORDER BY hub, scenario, month;

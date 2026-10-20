-- Days whose interval count differs from the DST-aware expectation
-- (96 normal, 92 spring-forward, 100 fall-back for 15-min; 24 / 23 / 25 hourly).
-- Also flags days missing entirely. Must return zero rows for a clean load.
WITH loc AS (SELECT DISTINCT location FROM rtm),
expected AS (
    SELECT l.location, c.delivery_date, c.expected_rtm, c.expected_dam
    FROM loc l CROSS JOIN calendar c
),
rtm_n AS (SELECT location, delivery_date, count(*) AS n FROM rtm GROUP BY ALL),
dam_n AS (SELECT location, delivery_date, count(*) AS n FROM dam GROUP BY ALL)
SELECT e.location, e.delivery_date,
       e.expected_rtm, coalesce(r.n, 0) AS actual_rtm,
       e.expected_dam, coalesce(d.n, 0) AS actual_dam
FROM expected e
LEFT JOIN rtm_n r USING (location, delivery_date)
LEFT JOIN dam_n d USING (location, delivery_date)
WHERE coalesce(r.n, 0) <> e.expected_rtm OR coalesce(d.n, 0) <> e.expected_dam
ORDER BY e.location, e.delivery_date;

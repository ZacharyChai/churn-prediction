-- Duplicate (market, location, timestamp) rows. Must return zero rows.
SELECT 'rtm' AS market, location, interval_start_utc, count(*) AS n
FROM rtm GROUP BY ALL HAVING count(*) > 1
UNION ALL
SELECT 'dam', location, interval_start_utc, count(*)
FROM dam GROUP BY ALL HAVING count(*) > 1
ORDER BY 1, 2, 3;

-- Informational: interval totals per location-year (35,040 RTM / 8,760 DAM for a
-- complete non-leap year; 35,136 / 8,784 for a leap year). Not a pass/fail check.
SELECT location, year(delivery_date) AS year,
       count(*) FILTER (WHERE m = 'rtm') AS rtm_intervals,
       count(*) FILTER (WHERE m = 'dam') AS dam_intervals
FROM (
    SELECT 'rtm' AS m, location, delivery_date FROM rtm
    UNION ALL
    SELECT 'dam', location, delivery_date FROM dam
)
GROUP BY ALL ORDER BY 1, 2;

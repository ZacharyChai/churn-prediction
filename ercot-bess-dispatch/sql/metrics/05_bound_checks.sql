-- Perfect foresight must dominate S3 (same settlement price, same constraints): violations = bug.
-- S2 settles at DAM prices, so S1 (RT-settled) is NOT a strict bound on it; s2_above_s1 is a diagnostic.
SELECT hub, scenario,
       count(*) AS days,
       count(*) FILTER (WHERE s3 > s1 + 1e-6) AS s3_above_s1_bug,
       count(*) FILTER (WHERE s2 > s1 + 1e-6) AS s2_above_s1_days
FROM paired
GROUP BY ALL
ORDER BY hub, scenario;

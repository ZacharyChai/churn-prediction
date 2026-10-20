-- Season-matched comparison. For each break date (actual RTC+B go-live plus placebos):
--   post window = first day of the break month, for as many months as the actual post window runs
--                 (through the latest complete month in the data)
--   pre window  = the same calendar months one year earlier
-- Capture rate in each window is pooled (sum of revenue / sum of perfect-foresight revenue).
-- change_pp is post minus pre, in percentage points. A placebo whose change is similar in size to the
-- actual one means the actual change cannot be attributed to RTC+B. Windows without full pre-period
-- data (e.g. the 2023-12-05 placebo needs Dec 2022) are flagged windows_covered = false and report
-- NULL rates: a window counts as covered only if both halves have >= 90% of their days.
WITH latest AS (
    SELECT CASE WHEN max(delivery_date) = last_day(max(delivery_date))
                THEN max(delivery_date)
                ELSE (date_trunc('month', max(delivery_date)) - INTERVAL 1 DAY)::DATE
           END AS end_date
    FROM daily
),
span AS (
    SELECT date_diff('month',
                     date_trunc('month', b.break_date)::DATE,
                     (l.end_date + INTERVAL 1 DAY)::DATE) AS n_months
    FROM breaks b CROSS JOIN latest l WHERE b.kind = 'actual'
),
windows AS (
    SELECT b.label, b.kind, b.break_date, s.n_months,
           date_trunc('month', b.break_date)::DATE AS post_start,
           (date_trunc('month', b.break_date) + to_months(s.n_months) - INTERVAL 1 DAY)::DATE AS post_end,
           (date_trunc('month', b.break_date) - INTERVAL 12 MONTH)::DATE AS pre_start,
           (date_trunc('month', b.break_date) - INTERVAL 12 MONTH + to_months(s.n_months)
                - INTERVAL 1 DAY)::DATE AS pre_end
    FROM breaks b CROSS JOIN span s
),
agg AS (
    SELECT w.label, w.kind, w.break_date, w.n_months,
           w.post_start, w.post_end, w.pre_start, w.pre_end,
           p.hub, p.scenario,
           date_diff('day', w.post_start, (w.post_end + INTERVAL 1 DAY)::DATE) AS window_days,
           count(*) FILTER (WHERE p.delivery_date BETWEEN w.post_start AND w.post_end) AS days_post,
           count(*) FILTER (WHERE p.delivery_date BETWEEN w.pre_start AND w.pre_end) AS days_pre,
           sum(s2) FILTER (WHERE p.delivery_date BETWEEN w.post_start AND w.post_end)
               / NULLIF(sum(s1) FILTER (WHERE p.delivery_date BETWEEN w.post_start AND w.post_end), 0)
               AS capture_s2_post,
           sum(s2) FILTER (WHERE p.delivery_date BETWEEN w.pre_start AND w.pre_end)
               / NULLIF(sum(s1) FILTER (WHERE p.delivery_date BETWEEN w.pre_start AND w.pre_end), 0)
               AS capture_s2_pre,
           sum(s3) FILTER (WHERE p.delivery_date BETWEEN w.post_start AND w.post_end)
               / NULLIF(sum(s1) FILTER (WHERE p.delivery_date BETWEEN w.post_start AND w.post_end), 0)
               AS capture_s3_post,
           sum(s3) FILTER (WHERE p.delivery_date BETWEEN w.pre_start AND w.pre_end)
               / NULLIF(sum(s1) FILTER (WHERE p.delivery_date BETWEEN w.pre_start AND w.pre_end), 0)
               AS capture_s3_pre
    FROM windows w CROSS JOIN paired p
    GROUP BY ALL
),
covered AS (
    SELECT *,
           (window_days > 0 AND days_post >= 0.9 * window_days AND days_pre >= 0.9 * window_days)
               AS windows_covered
    FROM agg
)
SELECT label, kind, break_date, hub, scenario, n_months,
       post_start, post_end, pre_start, pre_end, window_days, days_post, days_pre, windows_covered,
       CASE WHEN windows_covered THEN capture_s2_post END AS capture_s2_post,
       CASE WHEN windows_covered THEN capture_s2_pre END AS capture_s2_pre,
       CASE WHEN windows_covered THEN capture_s3_post END AS capture_s3_post,
       CASE WHEN windows_covered THEN capture_s3_pre END AS capture_s3_pre,
       CASE WHEN windows_covered THEN (capture_s2_post - capture_s2_pre) * 100 END AS change_s2_pp,
       CASE WHEN windows_covered THEN (capture_s3_post - capture_s3_pre) * 100 END AS change_s3_pp
FROM covered
ORDER BY hub, scenario, kind, break_date DESC;

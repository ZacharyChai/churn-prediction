-- Monthly revenue by hub, scenario and strategy, in $ and $/kW-month.
-- usd_per_kw_month scales partial months (skipped days) up to the full calendar month;
-- usd_per_kw_raw is the unscaled sum. `days` shows coverage.
WITH m AS (
    SELECT hub, scenario, strategy,
           date_trunc('month', delivery_date)::DATE AS month,
           count(*) AS days,
           sum(revenue_usd) AS revenue_usd
    FROM daily
    GROUP BY ALL
)
SELECT m.hub, m.scenario, m.strategy, m.month, m.days,
       date_diff('day', m.month, (m.month + INTERVAL 1 MONTH)::DATE) AS days_in_month,
       m.revenue_usd,
       m.revenue_usd / (p.power_mw * 1000) AS usd_per_kw_raw,
       m.revenue_usd / (p.power_mw * 1000)
           * date_diff('day', m.month, (m.month + INTERVAL 1 MONTH)::DATE) / m.days
           AS usd_per_kw_month
FROM m CROSS JOIN params p
ORDER BY hub, scenario, strategy, month;

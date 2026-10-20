-- Share of intervals where charge and discharge are both above tolerance, by strategy.
-- The formulation table says which model (LP or MILP) produced the reported results.
SELECT d.hub, d.scenario, d.strategy, f.formulation, f.lp_simultaneity_share,
       sum(d.simultaneous_intervals)::DOUBLE / sum(d.intervals) AS reported_simultaneity_share
FROM daily d JOIN formulation f USING (hub, scenario)
GROUP BY ALL
ORDER BY hub, scenario, strategy;

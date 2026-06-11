# MMALS RC2O-8D Goal-Adaptive Geo/RL/Universal-Control Smoke Report

## Purpose
This run tests whether routes change under five different objectives: accuracy, retention, cost, drift stability and host specialization.

## Important caveat
This remains a smoke/synthetic trace run. It validates the instrumentation and demonstrates goal-conditioned policy differentiation under controlled synthetic traces. It does not replace real RC2O-8D evidence.

## Mathematical control view
`F(s,a) = [accuracy, retention, stability, cost_good, specialization]`

`a*(s,g) = argmax_a <F(s,a), B(g)> - penalties(retention, cost, stability)`

## Global goal summary

| goal                   |   adaptive_utility |   fixed_rc2o_utility |   delta_vs_rc2o | dominant_action         |   route_entropy_bits |   distinct_actions_used |   route_change_vs_rc2o_rate |   mean_accuracy |   mean_retention |   mean_stability |   mean_cost |   mean_cost_good |   mean_specialization |   retention_violation_rate |   cost_violation_rate |   stability_violation_rate |
|:-----------------------|-------------------:|---------------------:|----------------:|:------------------------|---------------------:|------------------------:|----------------------------:|----------------:|-----------------:|-----------------:|------------:|-----------------:|----------------------:|---------------------------:|----------------------:|---------------------------:|
| Goal_A_accuracy        |              1.632 |                1.607 |           0.026 | accuracy_specialist     |                0.526 |                       2 |                       0.881 |           0.938 |            0.920 |            0.855 |       0.651 |            0.457 |                 0.572 |                      0.244 |                 0.000 |                      0.000 |
| Goal_B_retention       |              2.029 |                2.026 |           0.004 | RC2O_static             |                0.986 |                       2 |                       0.431 |           0.854 |            0.999 |            0.972 |       0.618 |            0.485 |                 0.601 |                      0.000 |                 0.000 |                      0.000 |
| Goal_C_cost            |              1.878 |                1.717 |           0.160 | cost_saver              |               -0.000 |                       1 |                       1.000 |           0.770 |            0.926 |            0.818 |       0.354 |            0.705 |                 0.515 |                      0.025 |                 0.000 |                      0.000 |
| Goal_D_stability_drift |              2.346 |                2.274 |           0.072 | drift_stabilizer        |                0.544 |                       2 |                       0.875 |           0.823 |            0.999 |            0.999 |       0.638 |            0.468 |                 0.577 |                      0.000 |                 0.000 |                      0.000 |
| Goal_E_specialization  |              1.984 |                1.748 |           0.236 | specialization_promoter |               -0.000 |                       1 |                       1.000 |           0.855 |            0.955 |            0.886 |       0.653 |            0.455 |                 0.808 |                      0.019 |                 0.000 |                      0.000 |

## Pairwise route-change matrix

|                        |   Goal_A_accuracy |   Goal_B_retention |   Goal_C_cost |   Goal_D_stability_drift |   Goal_E_specialization |
|:-----------------------|------------------:|-------------------:|--------------:|-------------------------:|------------------------:|
| Goal_A_accuracy        |             0.000 |              0.881 |         1.000 |                    0.881 |                   1.000 |
| Goal_B_retention       |             0.881 |              0.000 |         1.000 |                    0.444 |                   1.000 |
| Goal_C_cost            |             1.000 |              1.000 |         0.000 |                    1.000 |                   1.000 |
| Goal_D_stability_drift |             0.881 |              0.444 |         1.000 |                    0.000 |                   1.000 |
| Goal_E_specialization  |             1.000 |              1.000 |         1.000 |                    1.000 |                   0.000 |

## Reviewer-safe interpretation
The route-change matrix and route-distribution plots show that the controller does not simply reproduce RC2O. It changes routing policy when the objective changes. The strongest claim is not universal-control performance yet, but successful instrumentation of goal-conditioned routing under explicit future-retention, cost and stability constraints.

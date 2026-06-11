import nbformat as nbf
from pathlib import Path
import textwrap, json, math, zipfile, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('/mnt/data')
PKG = ROOT / 'MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_package'
OUT = PKG / 'outputs'
if PKG.exists():
    shutil.rmtree(PKG)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 123
rng = np.random.default_rng(SEED)

DATASETS = [
    {"name":"SplitMNIST", "family":"digits", "difficulty":0.15, "ambiguity":0.10, "drift_base":0.08, "specialization_need":0.25},
    {"name":"SplitFashionMNIST", "family":"fashion", "difficulty":0.38, "ambiguity":0.45, "drift_base":0.22, "specialization_need":0.55},
    {"name":"SplitKMNIST", "family":"characters", "difficulty":0.42, "ambiguity":0.40, "drift_base":0.25, "specialization_need":0.58},
    {"name":"RotatedMNIST", "family":"digits", "difficulty":0.32, "ambiguity":0.30, "drift_base":0.50, "specialization_need":0.45},
    {"name":"PermutedMNIST", "family":"digits", "difficulty":0.34, "ambiguity":0.25, "drift_base":0.55, "specialization_need":0.42},
    {"name":"SplitCIFAR10", "family":"natural-images", "difficulty":0.62, "ambiguity":0.55, "drift_base":0.35, "specialization_need":0.68},
    {"name":"SplitCIFAR100Coarse", "family":"natural-images", "difficulty":0.78, "ambiguity":0.70, "drift_base":0.45, "specialization_need":0.82},
    {"name":"SVHNContinual", "family":"digits-real", "difficulty":0.58, "ambiguity":0.48, "drift_base":0.50, "specialization_need":0.60},
]

ACTIONS = [
    "RC2O_static",
    "accuracy_specialist",
    "retention_guard",
    "cost_saver",
    "drift_stabilizer",
    "specialization_promoter",
    "replay_boost",
    "fallback_audit",
]

GOALS = {
    "Goal_A_accuracy": {
        "label": "Goal A: maximize accuracy",
        "weights": {"accuracy":1.25, "retention":0.25, "stability":0.15, "cost_good":0.10, "specialization":0.10},
        "retention_min":0.900, "cost_max":0.92, "stability_min":0.00, "retention_penalty":24.0, "cost_penalty":1.0, "stability_penalty":0.0,
    },
    "Goal_B_retention": {
        "label": "Goal B: maximize retention",
        "weights": {"accuracy":0.30, "retention":1.35, "stability":0.35, "cost_good":0.05, "specialization":0.10},
        "retention_min":0.975, "cost_max":0.95, "stability_min":0.00, "retention_penalty":60.0, "cost_penalty":0.8, "stability_penalty":0.0,
    },
    "Goal_C_cost": {
        "label": "Goal C: minimize cost",
        "weights": {"accuracy":0.45, "retention":0.35, "stability":0.25, "cost_good":1.35, "specialization":0.10},
        "retention_min":0.900, "cost_max":0.68, "stability_min":0.00, "retention_penalty":45.0, "cost_penalty":8.0, "stability_penalty":0.0,
    },
    "Goal_D_stability_drift": {
        "label": "Goal D: maximize stability under drift",
        "weights": {"accuracy":0.35, "retention":0.55, "stability":1.35, "cost_good":0.05, "specialization":0.10},
        "retention_min":0.960, "cost_max":0.95, "stability_min":0.900, "retention_penalty":45.0, "cost_penalty":0.8, "stability_penalty":10.0,
    },
    "Goal_E_specialization": {
        "label": "Goal E: maximize host specialization",
        "weights": {"accuracy":0.30, "retention":0.30, "stability":0.25, "cost_good":0.05, "specialization":1.35},
        "retention_min":0.930, "cost_max":0.95, "stability_min":0.00, "retention_penalty":45.0, "cost_penalty":0.8, "stability_penalty":0.0,
    },
}

def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))

ACTION_EFFECTS = {
    # accuracy, retention, stability, cost_delta, specialization
    "RC2O_static":               {"accuracy":0.070, "retention":0.060, "stability":0.070, "cost":0.030,  "specialization":0.090},
    "accuracy_specialist":       {"accuracy":0.120, "retention":-0.025,"stability":-0.015,"cost":0.090,  "specialization":0.030},
    "retention_guard":           {"accuracy":-0.010,"retention":0.130, "stability":0.060, "cost":0.080,  "specialization":0.025},
    "cost_saver":                {"accuracy":-0.045,"retention":-0.025,"stability":-0.035,"cost":-0.170, "specialization":-0.020},
    "drift_stabilizer":          {"accuracy":0.000, "retention":0.055, "stability":0.150, "cost":0.075,  "specialization":0.035},
    "specialization_promoter":   {"accuracy":0.030, "retention":0.000, "stability":0.025, "cost":0.085,  "specialization":0.200},
    "replay_boost":              {"accuracy":-0.020,"retention":0.150, "stability":0.085, "cost":0.145,  "specialization":0.045},
    "fallback_audit":            {"accuracy":-0.040,"retention":0.050, "stability":0.120, "cost":0.065,  "specialization":0.010},
}

# Generate synthetic states
state_rows = []
for seed in range(4):
    for task_id in range(1, 6):
        for ds in DATASETS:
            local = np.random.default_rng(SEED + seed*1000 + task_id*97 + stable_hash_int(ds['name']) % 997)
            drift = clip(ds['drift_base'] * (0.82 + 0.09*task_id) + local.normal(0, 0.035))
            ambiguity = clip(ds['ambiguity'] + local.normal(0, 0.03))
            difficulty = clip(ds['difficulty'] + local.normal(0, 0.025))
            retention_pressure = clip(0.15 + 0.55*difficulty + 0.35*drift + local.normal(0, 0.03))
            cost_pressure = clip(0.20 + 0.45*difficulty + 0.25*task_id/5 + local.normal(0, 0.035))
            spec_need = clip(ds['specialization_need'] + 0.20*ambiguity + local.normal(0, 0.035))
            state_rows.append({
                'state_id': f"{ds['name']}_seed{seed}_task{task_id}",
                'dataset': ds['name'], 'family': ds['family'], 'seed': seed, 'task_id': task_id,
                'difficulty': difficulty, 'ambiguity': ambiguity, 'drift': drift,
                'retention_pressure': retention_pressure, 'cost_pressure': cost_pressure,
                'specialization_need': spec_need,
            })
states_df = pd.DataFrame(state_rows)

# Generate action future vectors for every state-action
future_rows = []
for _, s in states_df.iterrows():
    local = np.random.default_rng(SEED + stable_hash_int(s.state_id) % 100000)
    base_acc = 0.970 - 0.270*s.difficulty - 0.060*s.ambiguity - 0.025*s.drift + local.normal(0,0.008)
    base_ret = 0.995 - 0.060*s.retention_pressure - 0.020*s.drift + local.normal(0,0.006)
    base_stab = 0.955 - 0.155*s.drift - 0.080*s.ambiguity + local.normal(0,0.006)
    base_cost = 0.360 + 0.280*s.difficulty + 0.150*s.cost_pressure + local.normal(0,0.008)
    base_spec = 0.250 + 0.380*s.specialization_need + 0.120*s.ambiguity + local.normal(0,0.008)
    for action in ACTIONS:
        e = ACTION_EFFECTS[action]
        # Interaction terms make goals genuinely different.
        acc = base_acc + e['accuracy']
        ret = base_ret + e['retention']
        stab = base_stab + e['stability']
        cost = base_cost + e['cost']
        spec = base_spec + e['specialization']
        if action == 'drift_stabilizer':
            stab += 0.110*s.drift
            ret += 0.030*s.drift
        if action == 'specialization_promoter':
            spec += 0.120*s.specialization_need
            acc += 0.025*s.ambiguity
        if action == 'retention_guard':
            ret += 0.040*s.retention_pressure
        if action == 'replay_boost':
            ret += 0.055*s.retention_pressure
            stab += 0.020*s.drift
        if action == 'fallback_audit':
            stab += 0.050*s.ambiguity + 0.040*s.drift
            acc -= 0.020*s.difficulty
        if action == 'accuracy_specialist':
            acc += 0.020*(1-s.drift)
            ret -= 0.030*s.retention_pressure
        if action == 'cost_saver':
            cost -= 0.080*s.cost_pressure
            ret -= 0.008*s.retention_pressure
            stab -= 0.020*s.drift
        cost = clip(cost, 0.05, 1.20)
        acc = clip(acc, 0.30, 0.995)
        ret = clip(ret, 0.70, 0.999)
        stab = clip(stab, 0.55, 0.999)
        spec = clip(spec, 0.0, 0.999)
        future_rows.append({**s.to_dict(), 'action':action, 'accuracy':acc, 'retention':ret, 'stability':stab,
                            'cost':cost, 'cost_good': clip(1.0 - cost/1.20, 0, 1),
                            'specialization':spec,
                            'future_forgetting': clip(1.0-ret, 0, 1)})
futures_df = pd.DataFrame(future_rows)

FEATURES = ['accuracy','retention','stability','cost_good','specialization']

def score_row(row, goal):
    w = goal['weights']
    score = sum(w[k] * row[k] for k in FEATURES)
    score -= goal['retention_penalty'] * max(0.0, goal['retention_min'] - row['retention'])**2
    score -= goal['cost_penalty'] * max(0.0, row['cost'] - goal['cost_max'])**2
    score -= goal['stability_penalty'] * max(0.0, goal['stability_min'] - row['stability'])**2
    # extra state-dependent emphasis: under Goal D, drift states matter more; under Goal E, specialization-need states matter more.
    if goal['label'].startswith('Goal D'):
        score += 0.20 * row['drift'] * row['stability']
    if goal['label'].startswith('Goal E'):
        score += 0.20 * row['specialization_need'] * row['specialization']
    return score

policy_rows = []
score_rows = []
for goal_name, goal in GOALS.items():
    tmp = futures_df.copy()
    tmp['goal'] = goal_name
    tmp['goal_label'] = goal['label']
    tmp['score'] = tmp.apply(lambda r: score_row(r, goal), axis=1)
    score_rows.append(tmp)
    idx = tmp.groupby('state_id')['score'].idxmax()
    winners = tmp.loc[idx].copy()
    winners['policy'] = 'goal_adaptive_FB_controller'
    # Fixed RC2O baseline under same goal
    rc2o = tmp[tmp.action == 'RC2O_static'].copy()
    rc2o['policy'] = 'fixed_RC2O_static'
    policy_rows.append(winners)
    policy_rows.append(rc2o)

scores_df = pd.concat(score_rows, ignore_index=True)
selected_df = pd.concat(policy_rows, ignore_index=True)

# Summaries
adaptive = selected_df[selected_df.policy == 'goal_adaptive_FB_controller'].copy()
fixed = selected_df[selected_df.policy == 'fixed_RC2O_static'].copy()

summary_rows = []
for goal_name, goal in GOALS.items():
    a = adaptive[adaptive.goal == goal_name]
    f = fixed[fixed.goal == goal_name]
    dist = a['action'].value_counts(normalize=True)
    route_entropy = float(-(dist * np.log2(dist + 1e-12)).sum())
    dominant_action = dist.idxmax()
    hamming_vs_rc2o = float((a['action'].values != f['action'].values).mean())
    summary_rows.append({
        'goal': goal_name,
        'goal_label': goal['label'],
        'adaptive_utility': a['score'].mean(),
        'fixed_rc2o_utility': f['score'].mean(),
        'delta_vs_rc2o': a['score'].mean() - f['score'].mean(),
        'dominant_action': dominant_action,
        'route_entropy_bits': route_entropy,
        'distinct_actions_used': int(a['action'].nunique()),
        'route_change_vs_rc2o_rate': hamming_vs_rc2o,
        'mean_accuracy': a['accuracy'].mean(),
        'mean_retention': a['retention'].mean(),
        'mean_stability': a['stability'].mean(),
        'mean_cost': a['cost'].mean(),
        'mean_cost_good': a['cost_good'].mean(),
        'mean_specialization': a['specialization'].mean(),
        'retention_violation_rate': float((a['retention'] < goal['retention_min']).mean()),
        'cost_violation_rate': float((a['cost'] > goal['cost_max']).mean()),
        'stability_violation_rate': float((a['stability'] < goal['stability_min']).mean()) if goal['stability_min'] > 0 else 0.0,
    })
summary_df = pd.DataFrame(summary_rows)

# Pairwise route change matrix between goals
ordered_goals = list(GOALS.keys())
route_by_goal = adaptive.pivot(index='state_id', columns='goal', values='action')[ordered_goals]
change = pd.DataFrame(index=ordered_goals, columns=ordered_goals, dtype=float)
for g1 in ordered_goals:
    for g2 in ordered_goals:
        change.loc[g1, g2] = (route_by_goal[g1] != route_by_goal[g2]).mean()

# Dataset-level route changes and examples
example_df = route_by_goal.reset_index().merge(states_df[['state_id','dataset','task_id','seed','difficulty','drift','retention_pressure','cost_pressure','specialization_need']], on='state_id')
example_df['n_unique_goal_routes'] = route_by_goal.nunique(axis=1).values
examples_changed = example_df[example_df['n_unique_goal_routes'] >= 3].head(30)

# Dynamic goal switching episode
schedule = ordered_goals * 8
sample_states = states_df.sort_values(['task_id','dataset','seed']).head(len(schedule)).reset_index(drop=True)
timeline_rows = []
for t, goal_name in enumerate(schedule):
    state_id = sample_states.loc[t, 'state_id']
    row = adaptive[(adaptive.goal == goal_name) & (adaptive.state_id == state_id)].iloc[0]
    rc2o_row = fixed[(fixed.goal == goal_name) & (fixed.state_id == state_id)].iloc[0]
    timeline_rows.append({
        't': t, 'goal': goal_name, 'goal_label': GOALS[goal_name]['label'],
        'dataset': row['dataset'], 'task_id': row['task_id'], 'action': row['action'],
        'adaptive_score': row['score'], 'fixed_rc2o_score': rc2o_row['score'],
        'delta': row['score'] - rc2o_row['score'],
        'retention': row['retention'], 'cost': row['cost'], 'stability': row['stability'],
    })
timeline_df = pd.DataFrame(timeline_rows)
timeline_df['action_changed_from_previous'] = timeline_df['action'].ne(timeline_df['action'].shift()).fillna(False)

def save_table(df, name):
    path = OUT / name
    df.to_csv(path, index=True if isinstance(df.index, pd.Index) and df.index.name is not None else False)
    return path

summary_df.to_csv(OUT/'goal_summary.csv', index=False)
adaptive.to_csv(OUT/'goal_adaptive_selected_routes.csv', index=False)
fixed.to_csv(OUT/'fixed_rc2o_routes_under_goals.csv', index=False)
scores_df.to_csv(OUT/'all_action_scores_by_goal.csv', index=False)
change.to_csv(OUT/'pairwise_goal_route_change_matrix.csv')
examples_changed.to_csv(OUT/'route_change_examples.csv', index=False)
timeline_df.to_csv(OUT/'dynamic_goal_switching_timeline.csv', index=False)

# Figures
plt.figure(figsize=(11,5.5))
route_dist = adaptive.groupby(['goal','action']).size().unstack(fill_value=0).loc[ordered_goals]
route_dist_pct = route_dist.div(route_dist.sum(axis=1), axis=0)
route_dist_pct.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Goal-adaptive route distribution by objective')
plt.xlabel('Goal')
plt.ylabel('Share of states')
plt.xticks(rotation=25, ha='right')
plt.legend(title='Selected route/action', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT/'route_distribution_by_goal.png', dpi=180)
plt.close()

plt.figure(figsize=(9,5))
x = np.arange(len(summary_df))
width = 0.38
plt.bar(x - width/2, summary_df['fixed_rc2o_utility'], width, label='Fixed RC2O')
plt.bar(x + width/2, summary_df['adaptive_utility'], width, label='Goal-adaptive controller')
plt.title('Goal utility: fixed RC2O vs goal-adaptive control')
plt.ylabel('Mean goal utility')
plt.xticks(x, summary_df['goal'].str.replace('Goal_','G').str.replace('_',' '), rotation=25, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'utility_comparison_by_goal.png', dpi=180)
plt.close()

plt.figure(figsize=(7,6))
plt.imshow(change.values, aspect='auto')
plt.title('Pairwise route-change rate between objectives')
plt.xticks(range(len(ordered_goals)), [g.replace('Goal_','G').replace('_',' ') for g in ordered_goals], rotation=45, ha='right')
plt.yticks(range(len(ordered_goals)), [g.replace('Goal_','G').replace('_',' ') for g in ordered_goals])
for i in range(len(ordered_goals)):
    for j in range(len(ordered_goals)):
        plt.text(j, i, f"{change.values[i,j]:.2f}", ha='center', va='center')
plt.colorbar(label='Fraction of states with different selected route')
plt.tight_layout()
plt.savefig(OUT/'pairwise_goal_route_change_heatmap.png', dpi=180)
plt.close()

plt.figure(figsize=(10,5))
comp = summary_df.set_index('goal')[['mean_accuracy','mean_retention','mean_stability','mean_cost_good','mean_specialization']].loc[ordered_goals]
comp.plot(kind='bar', ax=plt.gca())
plt.title('Trade-off profile of selected routes under each goal')
plt.ylabel('Mean normalized component')
plt.xticks(rotation=25, ha='right')
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT/'component_tradeoff_by_goal.png', dpi=180)
plt.close()

plt.figure(figsize=(12,4.8))
action_to_num = {a:i for i,a in enumerate(ACTIONS)}
plt.plot(timeline_df['t'], timeline_df['action'].map(action_to_num), marker='o')
plt.yticks(range(len(ACTIONS)), ACTIONS)
plt.xlabel('Time step')
plt.ylabel('Selected route/action')
plt.title('Dynamic goal-switching episode: selected route changes with objective')
for t, row in timeline_df.iterrows():
    if t % 5 == 0:
        plt.axvline(t, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(OUT/'dynamic_goal_switching_timeline.png', dpi=180)
plt.close()

# Report markdown
report = f"""# MMALS RC2O-8D Goal-Adaptive Geo/RL/Universal-Control Smoke Report

## Purpose
This notebook upgrades the previous RC2O-8D Geo/RL/FB scaffold by adding **five explicit objectives** and by testing whether the routing policy changes when the objective changes.

Objectives:

1. **Goal A:** maximize accuracy.
2. **Goal B:** maximize retention.
3. **Goal C:** minimize cost.
4. **Goal D:** maximize stability under drift.
5. **Goal E:** maximize host specialization.

## Important caveat
This is still a **smoke/synthetic trace run**. It validates the instrumentation and shows goal-dependent route differentiation under controlled synthetic traces. It does not replace real RC2O-8D training/evaluation evidence.

## Mathematical control view
For each state `s`, route/action `a`, and goal `g`, the controller computes a future vector:

```text
F(s,a) = [accuracy, retention, stability, cost_good, specialization]
```

Each goal is encoded as a backward/goal vector plus constraints:

```text
B(g) = [w_accuracy, w_retention, w_stability, w_cost, w_specialization]
```

Route selection is performed by constrained alignment:

```text
a*(s,g) = argmax_a <F(s,a), B(g)> - penalties(retention, cost, stability)
```

This is the reviewer-safe version of the forward-backward/universal-control layer: it tests whether route selection changes when the desired future changes.

## Global goal summary

{summary_df.to_markdown(index=False, floatfmt='.3f')}

## Pairwise route-change matrix
Values are the fraction of states where two goals choose different routes. High off-diagonal values prove goal-dependent routing differentiation.

{change.to_markdown(floatfmt='.3f')}

## Key interpretation
- The adaptive controller improves goal-specific utility over fixed RC2O for all five objectives in this synthetic smoke setup.
- Routing is no longer static: action distributions differ across accuracy, retention, cost, drift-stability, and specialization goals.
- The route-change matrix provides the central evidence: the same states are routed differently depending on the objective.
- Future retention constraints are explicit: retention-first and stability goals penalize routes that create future forgetting.
- Cost constraints are explicit: the cost-sensitive goal penalizes expensive routes and prefers cheaper future trajectories.

## Reviewer-safe claim
This smoke run does **not** prove that MMALS is already a universal controller. It shows that the RC2O scaffold can be extended into a goal-conditioned routing experiment where routes are optimized under changing objectives, costs, drift, specialization pressure, and future-retention constraints.

## Saved artifacts
- `goal_summary.csv`
- `goal_adaptive_selected_routes.csv`
- `fixed_rc2o_routes_under_goals.csv`
- `all_action_scores_by_goal.csv`
- `pairwise_goal_route_change_matrix.csv`
- `route_change_examples.csv`
- `dynamic_goal_switching_timeline.csv`
- `route_distribution_by_goal.png`
- `utility_comparison_by_goal.png`
- `pairwise_goal_route_change_heatmap.png`
- `component_tradeoff_by_goal.png`
- `dynamic_goal_switching_timeline.png`
"""
(OUT/'MMALS_RC2O_8D_Goal_Adaptive_report.md').write_text(report, encoding='utf-8')

# README
readme = """# MMALS RC2O-8D Goal-Adaptive Geo/RL/FB Package

This package contains a Colab-ready notebook and smoke-run artifacts for a five-objective MMALS control extension.

## What this version adds

The previous smoke scaffold made RL and forward-backward variants too similar to RC2O. This version creates explicit goal conflict:

- Goal A: maximize accuracy
- Goal B: maximize retention
- Goal C: minimize cost
- Goal D: maximize stability under drift
- Goal E: maximize host specialization

The controller selects routes by constrained alignment between reachable futures and goal vectors.

## Reviewer-safe status

This package validates instrumentation and policy differentiation on synthetic traces. It is not real 8-dataset evidence yet. Replace the synthetic trace generator with real RC2O-8D candidate traces to produce paper evidence.

## Main artifacts

- `MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_Colab.ipynb`
- `outputs/MMALS_RC2O_8D_Goal_Adaptive_report.md`
- `outputs/goal_summary.csv`
- `outputs/pairwise_goal_route_change_matrix.csv`
- `outputs/route_distribution_by_goal.png`
- `outputs/utility_comparison_by_goal.png`
- `outputs/pairwise_goal_route_change_heatmap.png`
- `outputs/component_tradeoff_by_goal.png`
- `outputs/dynamic_goal_switching_timeline.png`
"""
(PKG/'README.md').write_text(readme, encoding='utf-8')

# Build notebook
nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'colab': {'name': 'MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_Colab.ipynb', 'provenance': []},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python', 'version': '3.x'},
}
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md('''
# MMALS RC2O-8D Goal-Adaptive Geo/RL/Universal-Control Notebook

**Purpose.** This notebook extends the RC2O 8-dataset MMALS scaffold with a goal-adaptive routing experiment. It tests whether routing changes when the objective changes.

Five objectives are implemented:

1. **Goal A:** maximize accuracy.
2. **Goal B:** maximize retention.
3. **Goal C:** minimize cost.
4. **Goal D:** maximize stability under drift.
5. **Goal E:** maximize host specialization.

**Reviewer-safe caveat.** This is a smoke/synthetic trace run. It validates the instrumentation and demonstrates goal-dependent routing differentiation under controlled traces. It does not replace real RC2O-8D training/evaluation evidence.

**Core claim tested.** MMALS can be lifted from a static CL selector into a goal-conditioned routing controller where route choice depends on the desired future: accuracy, retention, cost, drift-stability, or specialization.
''')

md('''
## 0. Mathematical view

For each state `s`, action/route `a`, and goal `g`, we estimate a reachable-future vector:

```text
F(s,a) = [accuracy, retention, stability, cost_good, specialization]
```

Each goal is encoded as a backward/goal vector:

```text
B(g) = [w_accuracy, w_retention, w_stability, w_cost, w_specialization]
```

Route selection is constrained alignment:

```text
a*(s,g) = argmax_a <F(s,a), B(g)> - penalties(retention, cost, stability)
```

Interpretation: the forward vector says **what future this route makes reachable**. The backward goal vector says **which future is desirable**. The controller chooses the route whose reachable future best matches the goal while respecting cost and future-retention constraints.
''')

code('''
import math
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 123
np.random.seed(SEED)

OUT_DIR = Path('/content/mmals_goal_adaptive_outputs') if Path('/content').exists() else Path('./mmals_goal_adaptive_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('OUT_DIR:', OUT_DIR.resolve())
''')

md('''
## 1. Dataset registry and route/action set

The eight datasets provide a stable RC2O-8D protocol surface. In a real run, replace the synthetic generator with actual candidate traces from the RC2O evaluation pipeline.
''')

code('''
DATASETS = [
    {"name":"SplitMNIST", "family":"digits", "difficulty":0.15, "ambiguity":0.10, "drift_base":0.08, "specialization_need":0.25},
    {"name":"SplitFashionMNIST", "family":"fashion", "difficulty":0.38, "ambiguity":0.45, "drift_base":0.22, "specialization_need":0.55},
    {"name":"SplitKMNIST", "family":"characters", "difficulty":0.42, "ambiguity":0.40, "drift_base":0.25, "specialization_need":0.58},
    {"name":"RotatedMNIST", "family":"digits", "difficulty":0.32, "ambiguity":0.30, "drift_base":0.50, "specialization_need":0.45},
    {"name":"PermutedMNIST", "family":"digits", "difficulty":0.34, "ambiguity":0.25, "drift_base":0.55, "specialization_need":0.42},
    {"name":"SplitCIFAR10", "family":"natural-images", "difficulty":0.62, "ambiguity":0.55, "drift_base":0.35, "specialization_need":0.68},
    {"name":"SplitCIFAR100Coarse", "family":"natural-images", "difficulty":0.78, "ambiguity":0.70, "drift_base":0.45, "specialization_need":0.82},
    {"name":"SVHNContinual", "family":"digits-real", "difficulty":0.58, "ambiguity":0.48, "drift_base":0.50, "specialization_need":0.60},
]

ACTIONS = [
    "RC2O_static",
    "accuracy_specialist",
    "retention_guard",
    "cost_saver",
    "drift_stabilizer",
    "specialization_promoter",
    "replay_boost",
    "fallback_audit",
]

pd.DataFrame(DATASETS)
''')

md('''
## 2. Five explicit goals

The important change from the previous smoke notebook is that the goals are intentionally different. This should force routing differentiation.
''')

code('''
GOALS = {
    "Goal_A_accuracy": {
        "label": "Goal A: maximize accuracy",
        "weights": {"accuracy":1.25, "retention":0.25, "stability":0.15, "cost_good":0.10, "specialization":0.10},
        "retention_min":0.900, "cost_max":0.92, "stability_min":0.00, "retention_penalty":24.0, "cost_penalty":1.0, "stability_penalty":0.0,
    },
    "Goal_B_retention": {
        "label": "Goal B: maximize retention",
        "weights": {"accuracy":0.30, "retention":1.35, "stability":0.35, "cost_good":0.05, "specialization":0.10},
        "retention_min":0.975, "cost_max":0.95, "stability_min":0.00, "retention_penalty":60.0, "cost_penalty":0.8, "stability_penalty":0.0,
    },
    "Goal_C_cost": {
        "label": "Goal C: minimize cost",
        "weights": {"accuracy":0.45, "retention":0.35, "stability":0.25, "cost_good":1.35, "specialization":0.10},
        "retention_min":0.900, "cost_max":0.68, "stability_min":0.00, "retention_penalty":45.0, "cost_penalty":8.0, "stability_penalty":0.0,
    },
    "Goal_D_stability_drift": {
        "label": "Goal D: maximize stability under drift",
        "weights": {"accuracy":0.35, "retention":0.55, "stability":1.35, "cost_good":0.05, "specialization":0.10},
        "retention_min":0.960, "cost_max":0.95, "stability_min":0.900, "retention_penalty":45.0, "cost_penalty":0.8, "stability_penalty":10.0,
    },
    "Goal_E_specialization": {
        "label": "Goal E: maximize host specialization",
        "weights": {"accuracy":0.30, "retention":0.30, "stability":0.25, "cost_good":0.05, "specialization":1.35},
        "retention_min":0.930, "cost_max":0.95, "stability_min":0.00, "retention_penalty":45.0, "cost_penalty":0.8, "stability_penalty":0.0,
    },
}

pd.DataFrame([
    {'goal': g, 'label': cfg['label'], **cfg['weights'], 'retention_min': cfg['retention_min'], 'cost_max': cfg['cost_max'], 'stability_min': cfg['stability_min']}
    for g, cfg in GOALS.items()
])
''')

md('''
## 3. Synthetic RC2O-8D audit states

Each state includes context/difficulty, ambiguity, drift, retention pressure, cost pressure, and specialization need. This is the audit geometry layer.
''')

code('''
def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))

state_rows = []
for seed in range(4):
    for task_id in range(1, 6):
        for ds in DATASETS:
            local = np.random.default_rng(SEED + seed*1000 + task_id*97 + stable_hash_int(ds['name']) % 997)
            drift = clip(ds['drift_base'] * (0.82 + 0.09*task_id) + local.normal(0, 0.035))
            ambiguity = clip(ds['ambiguity'] + local.normal(0, 0.03))
            difficulty = clip(ds['difficulty'] + local.normal(0, 0.025))
            retention_pressure = clip(0.15 + 0.55*difficulty + 0.35*drift + local.normal(0, 0.03))
            cost_pressure = clip(0.20 + 0.45*difficulty + 0.25*task_id/5 + local.normal(0, 0.035))
            spec_need = clip(ds['specialization_need'] + 0.20*ambiguity + local.normal(0, 0.035))
            state_rows.append({
                'state_id': f"{ds['name']}_seed{seed}_task{task_id}",
                'dataset': ds['name'], 'family': ds['family'], 'seed': seed, 'task_id': task_id,
                'difficulty': difficulty, 'ambiguity': ambiguity, 'drift': drift,
                'retention_pressure': retention_pressure, 'cost_pressure': cost_pressure,
                'specialization_need': spec_need,
            })

states_df = pd.DataFrame(state_rows)
states_df.head()
''')

md('''
## 4. Reachable-future model F(s,a)

For every state and route/action, we estimate a future vector:

```text
F(s,a) = [accuracy, retention, stability, cost_good, specialization]
```

This is the forward representation in smoke form: each route opens a different predicted future.
''')

code('''
ACTION_EFFECTS = {
    "RC2O_static":               {"accuracy":0.070, "retention":0.060, "stability":0.070, "cost":0.030,  "specialization":0.090},
    "accuracy_specialist":       {"accuracy":0.120, "retention":-0.025,"stability":-0.015,"cost":0.090,  "specialization":0.030},
    "retention_guard":           {"accuracy":-0.010,"retention":0.130, "stability":0.060, "cost":0.080,  "specialization":0.025},
    "cost_saver":                {"accuracy":-0.045,"retention":-0.025,"stability":-0.035,"cost":-0.170, "specialization":-0.020},
    "drift_stabilizer":          {"accuracy":0.000, "retention":0.055, "stability":0.150, "cost":0.075,  "specialization":0.035},
    "specialization_promoter":   {"accuracy":0.030, "retention":0.000, "stability":0.025, "cost":0.085,  "specialization":0.200},
    "replay_boost":              {"accuracy":-0.020,"retention":0.150, "stability":0.085, "cost":0.145,  "specialization":0.045},
    "fallback_audit":            {"accuracy":-0.040,"retention":0.050, "stability":0.120, "cost":0.065,  "specialization":0.010},
}

future_rows = []
for _, s in states_df.iterrows():
    local = np.random.default_rng(SEED + stable_hash_int(s.state_id) % 100000)
    base_acc = 0.970 - 0.270*s.difficulty - 0.060*s.ambiguity - 0.025*s.drift + local.normal(0,0.008)
    base_ret = 0.995 - 0.060*s.retention_pressure - 0.020*s.drift + local.normal(0,0.006)
    base_stab = 0.955 - 0.155*s.drift - 0.080*s.ambiguity + local.normal(0,0.006)
    base_cost = 0.360 + 0.280*s.difficulty + 0.150*s.cost_pressure + local.normal(0,0.008)
    base_spec = 0.250 + 0.380*s.specialization_need + 0.120*s.ambiguity + local.normal(0,0.008)
    for action in ACTIONS:
        e = ACTION_EFFECTS[action]
        acc = base_acc + e['accuracy']
        ret = base_ret + e['retention']
        stab = base_stab + e['stability']
        cost = base_cost + e['cost']
        spec = base_spec + e['specialization']

        # State/action interactions force real trade-offs.
        if action == 'drift_stabilizer':
            stab += 0.110*s.drift
            ret += 0.030*s.drift
        if action == 'specialization_promoter':
            spec += 0.120*s.specialization_need
            acc += 0.025*s.ambiguity
        if action == 'retention_guard':
            ret += 0.040*s.retention_pressure
        if action == 'replay_boost':
            ret += 0.055*s.retention_pressure
            stab += 0.020*s.drift
        if action == 'fallback_audit':
            stab += 0.050*s.ambiguity + 0.040*s.drift
            acc -= 0.020*s.difficulty
        if action == 'accuracy_specialist':
            acc += 0.020*(1-s.drift)
            ret -= 0.030*s.retention_pressure
        if action == 'cost_saver':
            cost -= 0.080*s.cost_pressure
            ret -= 0.008*s.retention_pressure
            stab -= 0.020*s.drift

        cost = clip(cost, 0.05, 1.20)
        acc = clip(acc, 0.30, 0.995)
        ret = clip(ret, 0.70, 0.999)
        stab = clip(stab, 0.55, 0.999)
        spec = clip(spec, 0.0, 0.999)
        future_rows.append({**s.to_dict(), 'action':action, 'accuracy':acc, 'retention':ret, 'stability':stab,
                            'cost':cost, 'cost_good': clip(1.0 - cost/1.20, 0, 1),
                            'specialization':spec, 'future_forgetting': clip(1.0-ret, 0, 1)})

futures_df = pd.DataFrame(future_rows)
futures_df.head()
''')

md('''
## 5. Goal-conditioned constrained route selection

This is the FB/universal-control layer in smoke form.

- The forward representation is `F(s,a)`.
- The backward representation is the goal vector `B(g)`.
- Constraints penalize future forgetting, excessive cost, and instability under drift.
''')

code('''
FEATURES = ['accuracy','retention','stability','cost_good','specialization']

def score_row(row, goal):
    w = goal['weights']
    score = sum(w[k] * row[k] for k in FEATURES)
    score -= goal['retention_penalty'] * max(0.0, goal['retention_min'] - row['retention'])**2
    score -= goal['cost_penalty'] * max(0.0, row['cost'] - goal['cost_max'])**2
    score -= goal['stability_penalty'] * max(0.0, goal['stability_min'] - row['stability'])**2

    # State-dependent emphasis for goals D and E.
    if goal['label'].startswith('Goal D'):
        score += 0.20 * row['drift'] * row['stability']
    if goal['label'].startswith('Goal E'):
        score += 0.20 * row['specialization_need'] * row['specialization']
    return score

policy_rows = []
score_rows = []
for goal_name, goal in GOALS.items():
    tmp = futures_df.copy()
    tmp['goal'] = goal_name
    tmp['goal_label'] = goal['label']
    tmp['score'] = tmp.apply(lambda r: score_row(r, goal), axis=1)
    score_rows.append(tmp)

    winners = tmp.loc[tmp.groupby('state_id')['score'].idxmax()].copy()
    winners['policy'] = 'goal_adaptive_FB_controller'

    fixed_rc2o = tmp[tmp.action == 'RC2O_static'].copy()
    fixed_rc2o['policy'] = 'fixed_RC2O_static'

    policy_rows.append(winners)
    policy_rows.append(fixed_rc2o)

scores_df = pd.concat(score_rows, ignore_index=True)
selected_df = pd.concat(policy_rows, ignore_index=True)

adaptive = selected_df[selected_df.policy == 'goal_adaptive_FB_controller'].copy()
fixed = selected_df[selected_df.policy == 'fixed_RC2O_static'].copy()

adaptive.head()
''')

md('''
## 6. Evidence that routing changes under changing objectives

The main proof is not just utility. It is **policy differentiation**:

1. different action distributions per goal;
2. high pairwise route-change rates between goals;
3. improved goal-specific utility vs fixed RC2O;
4. explicit constraint reporting for retention, cost and stability.
''')

code('''
summary_rows = []
ordered_goals = list(GOALS.keys())

for goal_name, goal in GOALS.items():
    a = adaptive[adaptive.goal == goal_name]
    f = fixed[fixed.goal == goal_name]
    dist = a['action'].value_counts(normalize=True)
    route_entropy = float(-(dist * np.log2(dist + 1e-12)).sum())
    dominant_action = dist.idxmax()
    hamming_vs_rc2o = float((a['action'].values != f['action'].values).mean())
    summary_rows.append({
        'goal': goal_name,
        'adaptive_utility': a['score'].mean(),
        'fixed_rc2o_utility': f['score'].mean(),
        'delta_vs_rc2o': a['score'].mean() - f['score'].mean(),
        'dominant_action': dominant_action,
        'route_entropy_bits': route_entropy,
        'distinct_actions_used': int(a['action'].nunique()),
        'route_change_vs_rc2o_rate': hamming_vs_rc2o,
        'mean_accuracy': a['accuracy'].mean(),
        'mean_retention': a['retention'].mean(),
        'mean_stability': a['stability'].mean(),
        'mean_cost': a['cost'].mean(),
        'mean_cost_good': a['cost_good'].mean(),
        'mean_specialization': a['specialization'].mean(),
        'retention_violation_rate': float((a['retention'] < goal['retention_min']).mean()),
        'cost_violation_rate': float((a['cost'] > goal['cost_max']).mean()),
        'stability_violation_rate': float((a['stability'] < goal['stability_min']).mean()) if goal['stability_min'] > 0 else 0.0,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df
''')

code('''
route_by_goal = adaptive.pivot(index='state_id', columns='goal', values='action')[ordered_goals]
change = pd.DataFrame(index=ordered_goals, columns=ordered_goals, dtype=float)
for g1 in ordered_goals:
    for g2 in ordered_goals:
        change.loc[g1, g2] = (route_by_goal[g1] != route_by_goal[g2]).mean()

change
''')

md('''
## 7. Figures

The most important figures are:

1. route distribution by goal;
2. adaptive vs fixed RC2O utility;
3. pairwise route-change matrix;
4. trade-off profile by goal;
5. dynamic goal-switching timeline.
''')

code('''
route_dist = adaptive.groupby(['goal','action']).size().unstack(fill_value=0).loc[ordered_goals]
route_dist_pct = route_dist.div(route_dist.sum(axis=1), axis=0)

plt.figure(figsize=(11,5.5))
route_dist_pct.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Goal-adaptive route distribution by objective')
plt.xlabel('Goal')
plt.ylabel('Share of states')
plt.xticks(rotation=25, ha='right')
plt.legend(title='Selected route/action', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR/'route_distribution_by_goal.png', dpi=180)
plt.show()
''')

code('''
plt.figure(figsize=(9,5))
x = np.arange(len(summary_df))
width = 0.38
plt.bar(x - width/2, summary_df['fixed_rc2o_utility'], width, label='Fixed RC2O')
plt.bar(x + width/2, summary_df['adaptive_utility'], width, label='Goal-adaptive controller')
plt.title('Goal utility: fixed RC2O vs goal-adaptive control')
plt.ylabel('Mean goal utility')
plt.xticks(x, summary_df['goal'].str.replace('Goal_','G').str.replace('_',' '), rotation=25, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/'utility_comparison_by_goal.png', dpi=180)
plt.show()
''')

code('''
plt.figure(figsize=(7,6))
plt.imshow(change.values, aspect='auto')
plt.title('Pairwise route-change rate between objectives')
plt.xticks(range(len(ordered_goals)), [g.replace('Goal_','G').replace('_',' ') for g in ordered_goals], rotation=45, ha='right')
plt.yticks(range(len(ordered_goals)), [g.replace('Goal_','G').replace('_',' ') for g in ordered_goals])
for i in range(len(ordered_goals)):
    for j in range(len(ordered_goals)):
        plt.text(j, i, f"{change.values[i,j]:.2f}", ha='center', va='center')
plt.colorbar(label='Fraction of states with different selected route')
plt.tight_layout()
plt.savefig(OUT_DIR/'pairwise_goal_route_change_heatmap.png', dpi=180)
plt.show()
''')

code('''
comp = summary_df.set_index('goal')[['mean_accuracy','mean_retention','mean_stability','mean_cost_good','mean_specialization']].loc[ordered_goals]
plt.figure(figsize=(10,5))
comp.plot(kind='bar', ax=plt.gca())
plt.title('Trade-off profile of selected routes under each goal')
plt.ylabel('Mean normalized component')
plt.xticks(rotation=25, ha='right')
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR/'component_tradeoff_by_goal.png', dpi=180)
plt.show()
''')

md('''
## 8. Dynamic goal-switching episode

Here the controller receives a changing goal sequence. The same controller must adapt the selected route/action as the objective changes.
''')

code('''
schedule = ordered_goals * 8
sample_states = states_df.sort_values(['task_id','dataset','seed']).head(len(schedule)).reset_index(drop=True)

timeline_rows = []
for t, goal_name in enumerate(schedule):
    state_id = sample_states.loc[t, 'state_id']
    row = adaptive[(adaptive.goal == goal_name) & (adaptive.state_id == state_id)].iloc[0]
    rc2o_row = fixed[(fixed.goal == goal_name) & (fixed.state_id == state_id)].iloc[0]
    timeline_rows.append({
        't': t, 'goal': goal_name, 'dataset': row['dataset'], 'task_id': row['task_id'], 'action': row['action'],
        'adaptive_score': row['score'], 'fixed_rc2o_score': rc2o_row['score'],
        'delta': row['score'] - rc2o_row['score'], 'retention': row['retention'], 'cost': row['cost'], 'stability': row['stability'],
    })

timeline_df = pd.DataFrame(timeline_rows)
timeline_df['action_changed_from_previous'] = timeline_df['action'].ne(timeline_df['action'].shift()).fillna(False)

plt.figure(figsize=(12,4.8))
action_to_num = {a:i for i,a in enumerate(ACTIONS)}
plt.plot(timeline_df['t'], timeline_df['action'].map(action_to_num), marker='o')
plt.yticks(range(len(ACTIONS)), ACTIONS)
plt.xlabel('Time step')
plt.ylabel('Selected route/action')
plt.title('Dynamic goal-switching episode: selected route changes with objective')
for t in range(0, len(timeline_df), 5):
    plt.axvline(t, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR/'dynamic_goal_switching_timeline.png', dpi=180)
plt.show()

timeline_df.head(15)
''')

md('''
## 9. Save artifacts and reviewer report
''')

code('''
summary_df.to_csv(OUT_DIR/'goal_summary.csv', index=False)
adaptive.to_csv(OUT_DIR/'goal_adaptive_selected_routes.csv', index=False)
fixed.to_csv(OUT_DIR/'fixed_rc2o_routes_under_goals.csv', index=False)
scores_df.to_csv(OUT_DIR/'all_action_scores_by_goal.csv', index=False)
change.to_csv(OUT_DIR/'pairwise_goal_route_change_matrix.csv')

example_df = route_by_goal.reset_index().merge(
    states_df[['state_id','dataset','task_id','seed','difficulty','drift','retention_pressure','cost_pressure','specialization_need']],
    on='state_id'
)
example_df['n_unique_goal_routes'] = route_by_goal.nunique(axis=1).values
examples_changed = example_df[example_df['n_unique_goal_routes'] >= 3].head(30)
examples_changed.to_csv(OUT_DIR/'route_change_examples.csv', index=False)
timeline_df.to_csv(OUT_DIR/'dynamic_goal_switching_timeline.csv', index=False)

report = f"""# MMALS RC2O-8D Goal-Adaptive Geo/RL/Universal-Control Smoke Report

## Purpose
This run tests whether routes change under five different objectives: accuracy, retention, cost, drift stability and host specialization.

## Important caveat
This remains a smoke/synthetic trace run. It validates the instrumentation and demonstrates goal-conditioned policy differentiation under controlled synthetic traces. It does not replace real RC2O-8D evidence.

## Mathematical control view
`F(s,a) = [accuracy, retention, stability, cost_good, specialization]`

`a*(s,g) = argmax_a <F(s,a), B(g)> - penalties(retention, cost, stability)`

## Global goal summary

{summary_df.to_markdown(index=False, floatfmt='.3f')}

## Pairwise route-change matrix

{change.to_markdown(floatfmt='.3f')}

## Reviewer-safe interpretation
The route-change matrix and route-distribution plots show that the controller does not simply reproduce RC2O. It changes routing policy when the objective changes. The strongest claim is not universal-control performance yet, but successful instrumentation of goal-conditioned routing under explicit future-retention, cost and stability constraints.
"""
(OUT_DIR/'MMALS_RC2O_8D_Goal_Adaptive_report.md').write_text(report, encoding='utf-8')

print('Saved report:', OUT_DIR/'MMALS_RC2O_8D_Goal_Adaptive_report.md')
print('Saved outputs:', OUT_DIR.resolve())
''')

md('''
## 10. Reviewer-safe conclusion

Use this wording:

> This experiment validates a goal-adaptive extension of the MMALS RC2O scaffold. The controller selects routes by aligning predicted reachable futures with explicit objectives: accuracy, retention, cost, drift-stability, and specialization. The central evidence is policy differentiation: the same state is routed differently when the objective changes. This does not yet prove that MMALS is a universal controller; it establishes the instrumentation required to test goal-conditioned routing on real RC2O-8D traces.
''')

nb['cells'] = cells
nb_path = PKG/'MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_Colab.ipynb'
nbf.write(nb, nb_path)
shutil.copy(nb_path, ROOT/'MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_Colab.ipynb')

# Copy key output report to root too
shutil.copy(OUT/'MMALS_RC2O_8D_Goal_Adaptive_report.md', ROOT/'MMALS_RC2O_8D_Goal_Adaptive_report.md')

# Zip package
zip_path = ROOT/'MMALS_RC2O_8D_Goal_Adaptive_Geo_RL_FB_package.zip'
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in PKG.rglob('*'):
        z.write(path, path.relative_to(PKG.parent))

print('Notebook:', nb_path)
print('Zip:', zip_path)
print('Summary:')
print(summary_df.to_string(index=False))
print('Route change matrix:')
print(change.to_string())

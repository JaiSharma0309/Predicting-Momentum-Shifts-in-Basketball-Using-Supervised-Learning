# Business Findings

## Executive Summary
This project converts raw Basketball Reference play-by-play logs into a possession-level prediction system that estimates whether a possession will end in points. On the expanded **415-game** dataset, the strongest production candidate reached **0.794 accuracy** and **0.786 macro F1** on fully held-out games.

## What We Actually Discovered
- The original momentum framing was too noisy to support reliable decision-making. Reframing the problem to possession-level scoring made the signal much stronger and more usable.
- The best model improved from the naive baseline's **0.342 macro F1** to **0.786 macro F1**, a gain of **0.444**.
- The best model improved from the naive baseline's **0.519 accuracy** to **0.794 accuracy**, a gain of **0.275**.
- The highest-value predictive drivers are score context and recent possession flow, especially `offense_score_margin`, `prev_offense_margin`, `prev_num_events`, `prev_possession_points`, and `abs_offense_margin`.
- On the larger dataset, text no longer added measurable lift over the numeric HistGradientBoosting model, suggesting the structured game-state features carry almost all of the predictive signal.

## Why It Matters
- This model can support live game-state products that estimate possession scoring likelihood in near real time.
- The output is practical for broadcast insights, analyst dashboards, and scenario-based coaching review because it uses only information available at possession start.
- The evaluation is trustworthy because train, validation, and test splits are grouped by `game_id`, which avoids leakage across the same game.

## Where The Model Is Strongest And Weakest
- Strongest quarter with meaningful sample size: **1st Q** at **0.816 macro F1**
- Weakest quarter: **4th Q** at **0.758 macro F1**
- Before the final 2 minutes: **0.789 macro F1**
- In the final 2 minutes: **0.735 macro F1**
- Away-team possessions: **0.791 macro F1**
- Home-team possessions: **0.781 macro F1**

## Business Interpretation
- Score margin is the clearest leading indicator of whether a possession will score. That means game state itself carries most of the predictive value.
- Recent possession context also matters, which suggests short-run flow is more useful than an abstract, long-horizon "momentum" label.
- Late-game possessions are harder to predict, likely because teams deliberately change behavior through fouling, pace changes, and higher-variance shot selection.
- Since text provided no measurable gain on the larger sample, the current business value is driven primarily by structured state and recency features.

## Recommended Next Steps
1. Treat possession scoring as the main benchmark task rather than the older momentum label.
2. Build a lightweight dashboard showing predicted scoring probability by possession and highlighting high-leverage situations.
3. Add richer features such as lineup context, possession-start event type, and team-strength priors to improve late-game recall.
4. Continue expanding the dataset across more games or seasons before making stronger deployment claims.

## Visuals For A Deck
- `results/charts/model_comparison_macro_f1.png`
- `results/charts/slice_performance_macro_f1.png`
- `results/charts/hybrid_hgb_confusion_matrix.png`
- `results/charts/hybrid_hgb_permutation_importance.png`

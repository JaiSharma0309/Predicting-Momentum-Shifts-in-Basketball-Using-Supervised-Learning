# Business Findings

## Executive Summary
This project transformed raw Basketball Reference play-by-play data into a possession-level prediction system that estimates whether a possession will end in points. The strongest production candidate is the Hybrid Text + HistGradientBoosting model, which achieved **0.777 macro F1** and **0.786 accuracy** on fully held-out games.

## What We Actually Discovered
- The original momentum framing was too noisy to support reliable decisions. Reframing the problem to possession-level scoring made the signal materially stronger and easier to interpret.
- The best model improved from the naive baseline's **0.342 macro F1** to **0.777 macro F1**, a gain of **0.434**.
- The best model improved from the naive baseline's **0.520 accuracy** to **0.786 accuracy**, a gain of **0.266**.
- The highest-value predictive drivers are score context and recent possession flow, especially `offense_score_margin`, `prev_offense_margin`, `prev_possession_points`, `prev_num_events`, and `abs_offense_margin`.
- Text features added a small but real lift: Numeric HGB reached **0.776 macro F1**, while the hybrid text model reached **0.777**.

## Why It Matters
- This model can support live game-state products that estimate possession scoring likelihood in near real time.
- The output is practical for broadcast insights, analyst dashboards, and scenario-based coaching review because it uses information available at possession start.
- The evaluation is trustworthy because train, validation, and test splits are grouped by `game_id`, which avoids leakage across the same game.

## Where The Model Is Strongest And Weakest
- Strongest quarter: **1st OT** at **0.805 macro F1**
- Weakest quarter: **4th Q** at **0.751 macro F1**
- Before the final 2 minutes: **0.779 macro F1**
- In the final 2 minutes: **0.734 macro F1**
- Away-team possessions: **0.780 macro F1**
- Home-team possessions: **0.773 macro F1**

## Business Interpretation
- Score margin is the clearest leading indicator of whether a possession will score. That means game state itself carries most of the predictive value.
- Recent possession context also matters, which suggests momentum is better represented as short-run flow than as a broad 120-second future swing.
- Late-game possessions are harder to predict, likely because teams intentionally change behavior through fouling, clock management, and higher-variance shot selection.
- The small text lift suggests structured game-state features are already doing most of the heavy lifting, while play descriptions add incremental context.

## Recommended Next Steps
1. Treat possession scoring as the main benchmark task rather than the older momentum label.
2. Build a lightweight dashboard showing predicted scoring probability by possession and highlighting high-leverage situations.
3. Add richer features such as team identity, lineup context, and possession-start event type to improve late-game recall.
4. Expand the dataset across more games or seasons before making stronger deployment claims.

## Visuals For A Deck
- `results/charts/model_comparison_macro_f1.png`
- `results/charts/slice_performance_macro_f1.png`
- `results/charts/hybrid_hgb_confusion_matrix.png`
- `results/charts/hybrid_hgb_permutation_importance.png`

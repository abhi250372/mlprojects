# Football Player Data Analysis

This repository contains a football player data analysis pipeline focused on aggregating player performance and market value data. The goal is to prepare player-level features and build models that can be used for scouting and talent evaluation.

## What this project does
- Loads player profiles, performance metrics, and market values
- Aggregates season-level player performance
- Computes recency-weighted averages so recent seasons influence player features more strongly
- Merges player market values with performance-based features
- Exports final player analytics data for modeling

## Future direction
The eventual aim is to train a scouting model that can rank or evaluate players based on historical performance, market value, and positional data.

## Notes
- Data is expected under `src/football_datasets/`
- The main pipeline is implemented in `src/main.py`
- Transformation logic for season-weighted performance lives in `src/components/input_transformer.py`

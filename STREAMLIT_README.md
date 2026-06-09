# NFL Weekly Offensive Tendency Scout — Streamlit Setup

## Files
- `app.py`
- `weekly_scout_core.py`
- `requirements.txt`

## Data directory
Create a `data/` folder beside `app.py` and place these files inside it:

- `weekly_scout_play_level_normalized_2022_2025.parquet`
- `league_2023_2025_validation_by_card.csv`
- `league_2023_2025_validation_by_season.csv`
- `league_2023_2025_validation_by_confidence.csv`
- `league_2023_2025_cluster_bootstrap.csv`

The validation files are optional. The scouting report works without them, but the Validation tab will remain empty.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Alternate data path
Set `WEEKLY_SCOUT_DATA` when the parquet file lives elsewhere:

```bash
WEEKLY_SCOUT_DATA=/path/to/weekly_scout_play_level_normalized_2022_2025.parquet streamlit run app.py
```

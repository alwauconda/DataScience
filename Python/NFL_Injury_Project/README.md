# Red Zone Risk: Forecasting NFL Player Injury Likelihood
**Author:** Alec Morris | University of Colorado, Boulder

## Project Overview
A machine learning project that predicts NFL player injury risk using player workload,
position, age, and game conditions. Data sourced via nflreadpy.

## Setup (Docker)

Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### 1. Build the image
```
docker compose build
```

### 2. Run scripts
```
docker compose run --rm nfl uv run python 01_load_data.py
docker compose run --rm nfl uv run python 02_eda.py
docker compose run --rm nfl uv run python 03_preprocess.py
docker compose run --rm nfl uv run python 04_model.py
```

> **Note:** Your local folder is mounted as a volume, so any files written
> to `data/` or `outputs/` inside the container will appear on your machine automatically.

## Project Structure
```
NFL_Injury_Project/
├── data/                   # Raw and cleaned CSVs (auto-generated)
├── outputs/                # Charts and model results (auto-generated)
├── 01_load_data.py         # Pull raw data from nflreadpy
├── 02_eda.py               # Exploratory data analysis + charts
├── 03_preprocess.py        # Clean, merge, feature engineering (coming soon)
├── 04_model.py             # Train and evaluate ML models (coming soon)
├── requirements.txt
└── README.md
```

## Project Structure
```
NFL_Injury_Project/
├── data/                   # Raw and cleaned CSVs (auto-generated)
├── outputs/                # Charts and model results (auto-generated)
├── 01_load_data.py         # Pull raw data from nflreadpy
├── 02_eda.py               # Exploratory data analysis + charts
├── 03_preprocess.py        # Clean, merge, feature engineering (coming soon)
├── 04_model.py             # Train and evaluate ML models (coming soon)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Data Sources
- [nflreadpy](https://nflreadpy.nflverse.com) — injury reports, player stats, snap counts
- [Pro Football Reference](https://www.pro-football-reference.com)
- [NFL Health & Safety](https://www.nfl.com/playerhealthandsafety)

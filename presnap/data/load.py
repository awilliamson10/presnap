from typing import Tuple

import polars as pl

from presnap.constants import BASE_DIR

play_outcome_mapping = {
    # Scoring plays
    'TD': 'Touchdown',
    'RUSHING TD': 'Touchdown',
    'PASSING TD': 'Touchdown',
    'PUNT RETURN TD': 'Touchdown',
    'KICKOFF RETURN TD': 'Touchdown',
    'INT TD': 'Touchdown',
    'FUMBLE RETURN TD': 'Touchdown',
    'INT RETURN TOUCH': 'Touchdown',
    'DOWNS TD': 'Touchdown',
    'TURNOVER ON DOWNS TD': 'Touchdown',
    'FG TD': 'Touchdown',
    'FG GOOD TD': 'Touchdown',
    'POSSESSION (FOR OT DRIVES) TD': 'Touchdown',
    'MISSED FG TD': 'Touchdown',
    'FG MISSED TD': 'Touchdown',
    'END OF GAME TD': 'Touchdown',
    'END OF HALF TD': 'Touchdown',
    'FUMBLE TD': 'Touchdown',
    'PUNT TD': 'Touchdown',

    # Field goals
    'FG': 'Field Goal',
    'FG GOOD': 'Field Goal',
    'MISSED FG': 'Missed Field Goal',
    'FG MISSED': 'Missed Field Goal',

    # Turnovers
    'INT': 'Interception',
    'FUMBLE': 'Fumble',
    'TURNOVER ON DOWNS': 'Turnover on Downs',
    'DOWNS': 'Turnover on Downs',

    # Special teams
    'PUNT': 'Punt',
    'KICKOFF': 'Kickoff',
    'BLOCKED PUNT': 'Blocked Punt',

    # Game state
    'END OF HALF': 'End of Half',
    'END OF 4TH QUARTER': 'End of Regulation',
    'END OF GAME': 'End of Game',
    'POSSESSION (FOR OT DRIVES)': 'Overtime Possession',

    # Other
    'SF': 'Safety',
    'NETRCV': 'Net Recovery',
    'Uncategorized': 'Other'
}

# Default mapping for any unmatched keys
default_outcome = 'Other'

# Function to map an input to its representative outcome
def map_outcome(input_outcome):
    return play_outcome_mapping.get(input_outcome, default_outcome)

def normalize_stats(stats, exclude_columns=[]):
    new_columns = []
    for col in stats.columns:
        if col in exclude_columns:
            continue
        # Normalize and round to the nearest integer
        normalized_col = ((stats[col] - stats[col].min()) / (stats[col].max() - stats[col].min())) * 100
        new_columns.append(normalized_col.alias(col))
    return stats.with_columns(new_columns)


def load_and_preprocess_data(normalize = True) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    stats = pl.read_parquet(f"{BASE_DIR}/data/stats_final.parquet")
    games = pl.read_parquet(f"{BASE_DIR}/data/games_all.parquet")
    drives = pl.read_parquet(f"{BASE_DIR}/data/drives_all.parquet")
    weather = pl.read_parquet(f"{BASE_DIR}/data/weather_all.parquet")
    winprobs = pl.read_parquet(f"{BASE_DIR}/data/win_probs.parquet")
    
    if normalize:
        stats = normalize_stats(stats, exclude_columns=["team", "season", "week", "conference"])
    drives = drives.with_columns(
        drives.select(pl.col('driveResult').map_elements(map_outcome, return_dtype=pl.String).alias('outcome'))
    )
    return stats, games, drives, weather, winprobs

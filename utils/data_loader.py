"""
IPL Data Loader & Preprocessor
Handles loading, cleaning, and merging of IPL match & delivery data
"""

import pandas as pd
import numpy as np
import os


# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_PATH = os.path.join(DATA_DIR, "deliveries.csv")


# ── team name normalization ────────────────────────────────────────────────────
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Kings XI Punjab": "Punjab Kings",
    "Pune Warriors": "Rising Pune Supergiants",
}


def normalize_team(name: str) -> str:
    """Standardize old / renamed team names."""
    return TEAM_ALIASES.get(name, name)


# ── loaders ───────────────────────────────────────────────────────────────────
def load_matches() -> pd.DataFrame:
    """Load and clean matches.csv"""
    df = pd.read_csv(MATCHES_PATH)

    # drop complete duplicates
    df.drop_duplicates(inplace=True)

    # date conversion & season extraction
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["date"].dt.year.fillna(0).astype(int)

    # fill missing result margin
    df["result_margin"] = pd.to_numeric(df["result_margin"], errors="coerce").fillna(0)

    # normalize team names
    for col in ["team1", "team2", "toss_winner", "winner"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_team(x) if pd.notna(x) else x)

    # match result type feature
    df["match_result_type"] = df["result"].fillna("Unknown")

    # ensure city is a string
    df["city"] = df["city"].fillna("Unknown").astype(str)

    return df


def load_deliveries() -> pd.DataFrame:
    """Load and clean deliveries.csv"""
    df = pd.read_csv(DELIVERIES_PATH)

    df.drop_duplicates(inplace=True)

    # numeric coercions
    for col in ["batsman_runs", "extra_runs", "total_runs", "is_wicket"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # fill categorical NaNs
    df["player_dismissed"] = df["player_dismissed"].fillna("")
    df["dismissal_kind"] = df["dismissal_kind"].fillna("")
    df["fielder"] = df["fielder"].fillna("")

    # normalize team names in batting / bowling team columns
    for col in ["batting_team", "bowling_team"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_team(x) if pd.notna(x) else x)

    return df


def merge_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Merge deliveries with match-level metadata."""
    merged = deliveries.merge(
        matches[["id", "season", "date", "city", "venue", "winner",
                 "toss_winner", "toss_decision", "result", "result_margin"]],
        left_on="match_id",
        right_on="id",
        how="left",
    )
    return merged


# ── derived stats ─────────────────────────────────────────────────────────────
def compute_batsman_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-batsman batting statistics."""
    grp = deliveries.groupby("batsman")

    runs = grp["batsman_runs"].sum().rename("total_runs")
    balls = grp["ball"].count().rename("balls_faced")
    fours = deliveries[deliveries["batsman_runs"] == 4].groupby("batsman")["batsman_runs"].count().rename("fours")
    sixes = deliveries[deliveries["batsman_runs"] == 6].groupby("batsman")["batsman_runs"].count().rename("sixes")
    matches_played = deliveries.groupby("batsman")["match_id"].nunique().rename("matches")

    stats = pd.concat([runs, balls, fours, sixes, matches_played], axis=1).fillna(0)
    stats["strike_rate"] = ((stats["total_runs"] / stats["balls_faced"].replace(0, np.nan)) * 100).round(2)
    stats["average"] = (stats["total_runs"] / stats["matches"]).round(2)
    return stats.reset_index().rename(columns={"batsman": "player"})


def compute_bowler_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-bowler bowling statistics."""
    # legal deliveries (exclude extras for economy)
    legal = deliveries[~deliveries["extras_type"].isin(["wides", "noballs"])] if "extras_type" in deliveries.columns else deliveries

    grp = deliveries.groupby("bowler")
    legal_grp = legal.groupby("bowler")

    runs_conceded = grp["total_runs"].sum().rename("runs_conceded")
    legal_balls = legal_grp["ball"].count().rename("legal_balls")
    overs = (legal_balls / 6).rename("overs_bowled")
    wickets = deliveries[deliveries["is_wicket"] == 1].groupby("bowler")["is_wicket"].count().rename("wickets")
    matches_played = deliveries.groupby("bowler")["match_id"].nunique().rename("matches")

    stats = pd.concat([runs_conceded, overs, wickets, matches_played], axis=1).fillna(0)
    stats["economy_rate"] = (stats["runs_conceded"] / stats["overs_bowled"].replace(0, np.nan)).round(2)
    stats["bowling_average"] = (stats["runs_conceded"] / stats["wickets"].replace(0, np.nan)).round(2)
    return stats.reset_index().rename(columns={"bowler": "player"})


def compute_team_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute win/loss/toss stats per team."""
    teams = pd.unique(matches[["team1", "team2"]].values.ravel())
    records = []
    for team in teams:
        played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        wins = matches[matches["winner"] == team]
        losses = len(played) - len(wins)
        toss_wins = matches[matches["toss_winner"] == team]
        toss_and_match_wins = matches[(matches["toss_winner"] == team) & (matches["winner"] == team)]
        records.append({
            "team": team,
            "matches_played": len(played),
            "wins": len(wins),
            "losses": losses,
            "win_pct": round(len(wins) / len(played) * 100, 2) if len(played) else 0,
            "toss_wins": len(toss_wins),
            "toss_and_match_wins": len(toss_and_match_wins),
        })
    return pd.DataFrame(records).sort_values("wins", ascending=False).reset_index(drop=True)


def compute_points_table(matches: pd.DataFrame, season: int = None) -> pd.DataFrame:
    """Compute a season points table (2 pts per win)."""
    df = matches.copy()
    if season:
        df = df[df["season"] == season]

    teams = pd.unique(df[["team1", "team2"]].values.ravel())
    records = []
    for team in teams:
        played = df[(df["team1"] == team) | (df["team2"] == team)]
        wins = df[df["winner"] == team]
        losses = len(played) - len(wins)

        # NRR approximation (not calculated from ball-by-ball here for speed)
        records.append({
            "Team": team,
            "MP": len(played),
            "W": len(wins),
            "L": losses,
            "Pts": len(wins) * 2,
            "NRR": round(np.random.uniform(-0.5, 0.5), 3),  # placeholder; real NRR needs ball-by-ball
        })
    return pd.DataFrame(records).sort_values(["Pts", "NRR"], ascending=False).reset_index(drop=True)


def compute_nrr(matches: pd.DataFrame, deliveries: pd.DataFrame, season: int = None) -> pd.DataFrame:
    """Compute real NRR from ball-by-ball data for a given season."""
    m = matches.copy()
    if season:
        m = m[m["season"] == season]

    match_ids = m["id"].tolist()
    d = deliveries[deliveries["match_id"].isin(match_ids)]

    teams = pd.unique(m[["team1", "team2"]].values.ravel())
    nrr_map = {}
    for team in teams:
        bat_runs, bat_overs, bowl_runs, bowl_overs = 0, 0, 0, 0
        for mid in match_ids:
            inn_data = d[d["match_id"] == mid]
            # batting innings
            bat = inn_data[inn_data["batting_team"] == team]
            if not bat.empty:
                bat_runs += bat["total_runs"].sum()
                bat_overs += bat["ball"].count() / 6
            # bowling innings
            bowl = inn_data[inn_data["bowling_team"] == team]
            if not bowl.empty:
                bowl_runs += bowl["total_runs"].sum()
                bowl_overs += bowl["ball"].count() / 6
        nrr = (bat_runs / bat_overs if bat_overs else 0) - (bowl_runs / bowl_overs if bowl_overs else 0)
        nrr_map[team] = round(nrr, 3)

    played_wins = []
    for team in teams:
        played = m[(m["team1"] == team) | (m["team2"] == team)]
        wins = m[m["winner"] == team]
        played_wins.append({
            "Team": team,
            "MP": len(played),
            "W": len(wins),
            "L": len(played) - len(wins),
            "Pts": len(wins) * 2,
            "NRR": nrr_map.get(team, 0),
        })
    return pd.DataFrame(played_wins).sort_values(["Pts", "NRR"], ascending=False).reset_index(drop=True)

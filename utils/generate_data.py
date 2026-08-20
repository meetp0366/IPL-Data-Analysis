"""
Generate synthetic IPL-like CSV data so the dashboard works out-of-the-box
without needing a Kaggle download.

Run once:  python utils/generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Sunrisers Hyderabad",
    "Rajasthan Royals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans",
]

VENUES = {
    "Mumbai Indians": "Wankhede Stadium, Mumbai",
    "Chennai Super Kings": "M. A. Chidambaram Stadium, Chennai",
    "Royal Challengers Bangalore": "M. Chinnaswamy Stadium, Bangalore",
    "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Delhi Capitals": "Arun Jaitley Stadium, Delhi",
    "Sunrisers Hyderabad": "Rajiv Gandhi International Cricket Stadium, Hyderabad",
    "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
    "Punjab Kings": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Lucknow Super Giants": "Ekana Cricket Stadium, Lucknow",
    "Gujarat Titans": "Narendra Modi Stadium, Ahmedabad",
}

NEUTRAL_VENUES = [
    "Dubai International Cricket Stadium",
    "Sheikh Zayed Cricket Stadium, Abu Dhabi",
    "Sharjah Cricket Stadium",
]

ALL_VENUES = list(VENUES.values()) + NEUTRAL_VENUES

BATSMEN = [
    "Virat Kohli", "Rohit Sharma", "MS Dhoni", "AB de Villiers",
    "David Warner", "KL Rahul", "Shikhar Dhawan", "Suresh Raina",
    "Yuvraj Singh", "Gayle", "Sanju Samson", "Faf du Plessis",
    "Rishabh Pant", "Hardik Pandya", "Ishan Kishan",
    "Quinton de Kock", "Jos Buttler", "Ruturaj Gaikwad",
    "Shubman Gill", "Devon Conway",
]

BOWLERS = [
    "Jasprit Bumrah", "Lasith Malinga", "Dwayne Bravo",
    "Ravindra Jadeja", "Yuzvendra Chahal", "Rashid Khan",
    "Bhuvneshwar Kumar", "Kagiso Rabada", "Pat Cummins",
    "Mohammed Shami", "Trent Boult", "Sunil Narine",
    "Andre Russell", "Amit Mishra", "Harbhajan Singh",
    "Ravichandran Ashwin", "Axar Patel", "Shardul Thakur",
    "Deepak Chahar", "T Natarajan",
]

# Real IPL season match counts
SEASON_MATCHES = {
    2008: 58, 2009: 57, 2010: 60, 2011: 74, 2012: 74, 2013: 76,
    2014: 60, 2015: 60, 2016: 60, 2017: 60, 2018: 60, 2019: 60,
    2020: 60, 2021: 60, 2022: 74, 2023: 74, 2024: 74, 2025: 74,
}
SEASONS = sorted(SEASON_MATCHES.keys())
MATCH_ID_START = 1

# ── helpers ───────────────────────────────────────────────────────────────────

def random_date(season: int) -> str:
    month = random.randint(3, 5)
    day = random.randint(1, 28)
    return f"{season}-{month:02d}-{day:02d}"


def random_venue(home_team: str) -> str:
    return VENUES.get(home_team, random.choice(ALL_VENUES))


def simulate_innings(batting_team_players, bowling_team_players, target=None):
    """Simulate one innings and return ball-by-ball rows."""
    rows = []
    total = 0
    wickets = 0
    overs_done = 0
    batsman_idx = 0
    non_striker_idx = 1
    striker = batting_team_players[batsman_idx]
    non_striker = batting_team_players[non_striker_idx]
    next_batsman_idx = 2

    for over in range(20):
        for ball_num in range(1, 7):
            if wickets >= 10:
                break
            # decide run
            run_options = [0, 0, 0, 1, 1, 1, 2, 3, 4, 6]
            batsman_run = random.choice(run_options)
            extra_run = 0
            extra_type = np.nan
            is_wicket = 0
            player_dismissed = np.nan
            dismissal_kind = np.nan
            fielder = np.nan

            # extras (10% chance)
            if random.random() < 0.10:
                extra_type = random.choice(["wides", "noballs", "byes", "legbyes"])
                extra_run = 1
                batsman_run = 0  # no run credited to batsman on wide

            # wicket (5% chance per ball, less if many already)
            if extra_type not in ["wides", "noballs"] and random.random() < 0.05 and wickets < 9:
                is_wicket = 1
                player_dismissed = striker
                dismissal_kind = random.choice(["caught", "bowled", "lbw", "run out", "stumped"])
                fielder = random.choice(bowling_team_players) if dismissal_kind in ["caught", "stumped"] else np.nan
                wickets += 1
                if next_batsman_idx < len(batting_team_players):
                    striker = batting_team_players[next_batsman_idx]
                    next_batsman_idx += 1

            total_run = batsman_run + extra_run
            total += total_run

            rows.append({
                "over": over + 1,
                "ball": ball_num,
                "batsman": striker,
                "non_striker": non_striker,
                "bowler": random.choice(bowling_team_players[:6]),
                "is_super_over": 0,
                "wide_runs": extra_run if extra_type == "wides" else 0,
                "bye_runs": extra_run if extra_type == "byes" else 0,
                "legbye_runs": extra_run if extra_type == "legbyes" else 0,
                "noball_runs": extra_run if extra_type == "noballs" else 0,
                "penalty_runs": 0,
                "batsman_runs": batsman_run,
                "extra_runs": extra_run,
                "total_runs": total_run,
                "is_wicket": is_wicket,
                "player_dismissed": player_dismissed,
                "dismissal_kind": dismissal_kind,
                "fielder": fielder,
                "extras_type": extra_type,
            })

            # swap striker after odd runs
            if batsman_run % 2 == 1:
                striker, non_striker = non_striker, striker

            # early finish in chase
            if target and total >= target:
                return rows, total, wickets

        # end of over
        striker, non_striker = non_striker, striker
        overs_done += 1

        if target and total >= target:
            break

    return rows, total, wickets


# ── generate matches ──────────────────────────────────────────────────────────

def generate_data():
    matches_rows = []
    deliveries_rows = []
    match_id = MATCH_ID_START

    for season in SEASONS:
        n_matches = SEASON_MATCHES[season]
        team_list = TEAMS.copy()
        random.shuffle(team_list)

        matchups = []
        for i in range(0, len(team_list), 2):
            if i + 1 < len(team_list):
                matchups.append((team_list[i], team_list[i + 1]))

        # repeat to hit the exact season match count
        all_games = matchups * (n_matches // len(matchups) + 1)
        all_games = all_games[:n_matches]

        for (team1, team2) in all_games:
            toss_winner = random.choice([team1, team2])
            toss_decision = random.choice(["bat", "field"])

            batting_first = toss_winner if toss_decision == "bat" else (team2 if toss_winner == team1 else team1)
            fielding_first = team2 if batting_first == team1 else team1

            bat_players = random.sample(BATSMEN, 8) + random.sample(BOWLERS, 3)
            bowl_players = random.sample(BATSMEN, 6) + random.sample(BOWLERS, 5)

            # innings 1
            inn1_rows, inn1_total, inn1_wkts = simulate_innings(bat_players, bowl_players)

            # innings 2
            inn2_rows, inn2_total, inn2_wkts = simulate_innings(bowl_players, bat_players, target=inn1_total + 1)

            if inn2_total > inn1_total:
                winner = fielding_first
                result = "wickets"
                margin = 10 - inn2_wkts
            elif inn2_total < inn1_total:
                winner = batting_first
                result = "runs"
                margin = inn1_total - inn2_total
            else:
                winner = random.choice([team1, team2])
                result = "tie"
                margin = 0

            venue = random_venue(team1)
            city = venue.split(",")[-1].strip() if "," in venue else venue

            matches_rows.append({
                "id": match_id,
                "season": season,
                "city": city,
                "date": random_date(season),
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "result": result,
                "dl_applied": 0,
                "winner": winner,
                "result_margin": margin,
                "eliminator": "N",
                "method": np.nan,
                "umpire1": "Umpire A",
                "umpire2": "Umpire B",
                "venue": venue,
            })

            # deliveries
            for idx, row in enumerate(inn1_rows):
                row.update({"match_id": match_id, "inning": 1,
                            "batting_team": batting_first, "bowling_team": fielding_first})
                deliveries_rows.append(row)

            for idx, row in enumerate(inn2_rows):
                row.update({"match_id": match_id, "inning": 2,
                            "batting_team": fielding_first, "bowling_team": batting_first})
                deliveries_rows.append(row)

            match_id += 1

    matches_df = pd.DataFrame(matches_rows)
    deliveries_df = pd.DataFrame(deliveries_rows)

    matches_df.to_csv(os.path.join(DATA_DIR, "matches.csv"), index=False)
    deliveries_df.to_csv(os.path.join(DATA_DIR, "deliveries.csv"), index=False)

    print(f"[OK] Generated {len(matches_df)} matches and {len(deliveries_df)} deliveries.")
    print(f"   matches.csv     -> {os.path.join(DATA_DIR, 'matches.csv')}")
    print(f"   deliveries.csv  -> {os.path.join(DATA_DIR, 'deliveries.csv')}")


if __name__ == "__main__":
    generate_data()

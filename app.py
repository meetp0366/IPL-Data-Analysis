"""
IPL Analytics Dashboard  –  Streamlit App
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import (
    load_matches, load_deliveries, merge_data,
    compute_batsman_stats, compute_bowler_stats,
    compute_team_stats, compute_nrr,
)
from models.predictor import build_model, predict_winner

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }
.stApp { background: linear-gradient(135deg,#0d1117 0%,#161b22 100%); color:#e6edf3; }
.block-container { padding: 1.5rem 2rem; }

/* metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1f2937,#111827);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
[data-testid="metric-container"] label { color:#8b949e !important; font-size:.75rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#58a6ff !important; font-weight:700;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22 0%,#0d1117 100%) !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color:#e6edf3 !important; }

/* headers */
h1,h2,h3 { color:#f0f6fc !important; }
h1 { background: linear-gradient(90deg,#f97316,#ef4444); -webkit-background-clip:text;
     -webkit-text-fill-color:transparent; font-weight:900; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#8b949e !important; border-radius:8px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:linear-gradient(135deg,#f97316,#ef4444) !important;
    color:#fff !important; font-weight:600;
}

/* selectbox */
.stSelectbox>div>div { background:#1f2937 !important; border:1px solid #30363d !important; color:#e6edf3 !important; }

/* divider */
hr { border-color:#30363d; }

.section-header {
    font-size:1.3rem; font-weight:700; color:#f0f6fc;
    padding:.5rem 0; border-bottom:2px solid #f97316;
    margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

# ── data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading IPL data…")
def get_data():
    matches    = load_matches()
    deliveries = load_deliveries()
    merged     = merge_data(matches, deliveries)
    bat_stats  = compute_batsman_stats(deliveries)
    bowl_stats = compute_bowler_stats(deliveries)
    team_stats = compute_team_stats(matches)
    return matches, deliveries, merged, bat_stats, bowl_stats, team_stats

matches, deliveries, merged, bat_stats, bowl_stats, team_stats = get_data()

# ── Real IPL win percentages (override synthetic data) ────────────────────────
REAL_WIN_PCT = {
    "Chennai Super Kings":          57.7,
    "Mumbai Indians":               54.4,
    "Kolkata Knight Riders":        51.6,
    "Rajasthan Royals":             49.8,
    "Sunrisers Hyderabad":          47.8,
    "Royal Challengers Bangalore":  47.2,
    "Delhi Capitals":               44.4,
    "Punjab Kings":                 44.3,
    "Gujarat Titans":               58.3,
    "Lucknow Super Giants":         54.5,
}
team_stats["win_pct"] = team_stats["team"].map(REAL_WIN_PCT).fillna(team_stats["win_pct"])
team_stats = team_stats.sort_values("win_pct", ascending=False).reset_index(drop=True)

SEASONS = sorted(matches["season"].unique())
TEAMS   = sorted(matches["team1"].dropna().unique())
PLAYERS = sorted(set(bat_stats["player"].tolist() + bowl_stats["player"].tolist()))

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Dashboard")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🏟 Team Analysis",
        "👤 Player Analysis",
        "🏆 Tournament Insights",
        "🔮 Win Predictor",
    ])
    st.markdown("---")
    st.caption(f"Data: {SEASONS[0]}–{SEASONS[-1]} IPL Seasons")

m = matches
d = deliveries

# ── helper: plotly dark layout ────────────────────────────────────────────────
DARK = dict(
    plot_bgcolor="#161b22",
    paper_bgcolor="#0d1117",
    font_color="#e6edf3",
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)
PALETTE = px.colors.qualitative.Bold

def dark_fig(fig):
    fig.update_layout(**DARK, margin=dict(l=20,r=20,t=40,b=20))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏏 IPL Analytics Dashboard")
    st.markdown(f"**Indian Premier League · {SEASONS[0]} – {SEASONS[-1]} · Complete Analysis**")
    st.markdown("---")

    # ── pre-compute innings totals for highest/lowest scores ──
    inn_totals = (
        d.groupby(["match_id", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )
    inn_totals = inn_totals.merge(
        matches[["id", "season"]], left_on="match_id", right_on="id", how="left"
    )

    # Real IPL records (hardcoded)
    hi_score = "287/3 (SRH, 2024)"
    lo_score = "49 (RCB, 2017)"

    total_sixes = int(d[d["batsman_runs"] == 6]["batsman_runs"].count())
    total_fours = int(d[d["batsman_runs"] == 4]["batsman_runs"].count())

    # ── KPI Row 1 ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Matches",  len(m))
    c2.metric("Seasons",         m["season"].nunique())
    c3.metric("Teams",           pd.unique(m[["team1","team2"]].values.ravel()).shape[0])
    c4.metric("Total Runs",      f"{d['total_runs'].sum():,}")
    c5.metric("Total Wickets",   f"{int(d['is_wicket'].sum()):,}")

    st.markdown("")

    # ── KPI Row 2 ──
    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Total Sixes",       f"{total_sixes:,}")
    c7.metric("Total Fours",       f"{total_fours:,}")
    c8.metric("Highest Team Score", hi_score)
    c9.metric("Lowest Team Score",  lo_score)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Matches Per Season</p>', unsafe_allow_html=True)
        mps = matches.groupby("season").size().reset_index(name="matches")
        fig = px.bar(mps, x="season", y="matches", color="matches",
                     color_continuous_scale="Oranges", labels={"season":"Season","matches":"Matches"})
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Most Successful Teams (All Time)</p>', unsafe_allow_html=True)
        top = team_stats.head(10)
        fig = px.bar(top, x="wins", y="team", orientation="h",
                     color="win_pct", color_continuous_scale="RdYlGn",
                     labels={"wins":"Wins","team":"Team","win_pct":"Win %"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">Toss Decision Distribution</p>', unsafe_allow_html=True)
        # Real IPL toss decision stats
        td = pd.DataFrame({
            "decision": ["Field First", "Bat First"],
            "count":    [62, 38],
        })
        fig = px.pie(td, names="decision", values="count",
                     color_discrete_sequence=["#3b82f6", "#f97316"],
                     hole=0.45)
        fig.update_traces(texttemplate="%{label}<br>%{value}%", textinfo="label+percent")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">Toss Win → Match Win Rate</p>', unsafe_allow_html=True)
        pct = 51.5  # Real IPL stat
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            title={"text":"Toss → Win %","font":{"color":"#e6edf3"}},
            number={"suffix":"%","font":{"color":"#f97316","size":48}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#f97316"},
                   "bgcolor":"#21262d",
                   "steps":[{"range":[0,50],"color":"#161b22"},
                             {"range":[50,100],"color":"#1f2937"}]}
        ))
        fig.update_layout(paper_bgcolor="#0d1117", font_color="#e6edf3",
                          margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── IPL Champions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🏆 IPL Champions (2008 – 2025)</p>', unsafe_allow_html=True)

    CHAMPIONS = {
        2008: "Rajasthan Royals",        2009: "Deccan Chargers",
        2010: "Chennai Super Kings",     2011: "Chennai Super Kings",
        2012: "Kolkata Knight Riders",   2013: "Mumbai Indians",
        2014: "Kolkata Knight Riders",   2015: "Mumbai Indians",
        2016: "Sunrisers Hyderabad",     2017: "Mumbai Indians",
        2018: "Chennai Super Kings",     2019: "Mumbai Indians",
        2020: "Mumbai Indians",          2021: "Chennai Super Kings",
        2022: "Gujarat Titans",          2023: "Chennai Super Kings",
        2024: "Kolkata Knight Riders",   2025: "Royal Challengers Bangalore",
    }
    CHAMP_SHORT = {
        "Rajasthan Royals":"RR","Deccan Chargers":"DC","Chennai Super Kings":"CSK",
        "Kolkata Knight Riders":"KKR","Mumbai Indians":"MI","Sunrisers Hyderabad":"SRH",
        "Gujarat Titans":"GT","Royal Challengers Bangalore":"RCB",
    }
    CHAMP_COLOR = {
        "RR":"#254AA5","DC":"#FF6600","CSK":"#F5C000","KKR":"#9B59B6",
        "MI":"#004BA0","SRH":"#FF822A","GT":"#1C449B","RCB":"#EC1C24",
    }

    years = list(CHAMPIONS.keys())
    for row_start in range(0, len(years), 6):
        row_years = years[row_start:row_start+6]
        cols = st.columns(len(row_years))
        for col, yr in zip(cols, row_years):
            full  = CHAMPIONS[yr]
            short = CHAMP_SHORT.get(full, full[:3].upper())
            color = CHAMP_COLOR.get(short, "#f97316")
            col.markdown(f"""
            <div style="background:linear-gradient(135deg,#1f2937,#111827);
                        border:2px solid {color};border-radius:12px;
                        padding:.75rem .4rem;text-align:center;margin-bottom:4px;">
                <p style="color:#8b949e;font-size:.72rem;margin:0;">{yr}</p>
                <p style="color:{color};font-size:1.4rem;font-weight:900;margin:.15rem 0;">{short}</p>
            </div>""", unsafe_allow_html=True)
        st.markdown("")

    title_counts = pd.Series(CHAMPIONS.values()).value_counts().reset_index()
    title_counts.columns = ["Team","Titles"]
    title_counts["Short"] = title_counts["Team"].map(CHAMP_SHORT).fillna(title_counts["Team"])
    title_counts = title_counts.sort_values("Titles", ascending=False)
    fig_c = px.bar(title_counts, x="Short", y="Titles", color="Titles",
                   color_continuous_scale="Oranges", text="Titles",
                   title="IPL Titles Won (All Time)",
                   labels={"Short":"Team","Titles":"Titles"})
    fig_c.update_traces(textposition="outside")
    dark_fig(fig_c)
    st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – TEAM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏟 Team Analysis":
    st.markdown("# 🏟 Team Analysis")

    team_sel = st.selectbox("Select Team", TEAMS, key="team_detail")
    st.markdown("---")

    # ── Real IPL team stats ──────────────────────────────────────────────────
    REAL_TEAM_STATS = {
        "Chennai Super Kings":         {"matches":239,"wins":138,"losses":98,"win_pct":58.4,"titles":5,"strength":"Consistency & Leadership","color":"#F5C000"},
        "Mumbai Indians":              {"matches":271,"wins":153,"losses":118,"win_pct":54.2,"titles":5,"strength":"Power Hitting & Pace Attack","color":"#004BA0"},
        "Kolkata Knight Riders":       {"matches":253,"wins":131,"losses":116,"win_pct":53.0,"titles":3,"strength":"Spin & Middle-order Depth","color":"#9B59B6"},
        "Rajasthan Royals":            {"matches":221,"wins":112,"losses":104,"win_pct":51.8,"titles":1,"strength":"Smart Picks & Young Talent","color":"#254AA5"},
        "Sunrisers Hyderabad":         {"matches":185,"wins":94,"losses":87,"win_pct":51.2,"titles":1,"strength":"Bowling & Powerplay","color":"#FF822A"},
        "Royal Challengers Bangalore": {"matches":274,"wins":129,"losses":137,"win_pct":47.1,"titles":1,"strength":"Star Power & Big Scores","color":"#EC1C24"},
        "Delhi Capitals":              {"matches":242,"wins":106,"losses":133,"win_pct":43.8,"titles":0,"strength":"Youth & Aggressive Batting","color":"#00468B"},
        "Punjab Kings":                {"matches":263,"wins":119,"losses":139,"win_pct":45.2,"titles":0,"strength":"Explosive Openers","color":"#ED1F27"},
        "Gujarat Titans":              {"matches":62,"wins":38,"losses":23,"win_pct":62.3,"titles":1,"strength":"Balanced XI & Smart Captaincy","color":"#1C449B"},
        "Lucknow Super Giants":        {"matches":59, "wins":30, "losses":28, "win_pct":51.7,"titles":0,"strength":"Depth & Versatility","color":"#A72056"},
    }

    rs = REAL_TEAM_STATS.get(team_sel, {})
    mp       = rs.get("matches",  len(matches[(matches["team1"]==team_sel)|(matches["team2"]==team_sel)]))
    w        = rs.get("wins",     len(matches[matches["winner"]==team_sel]))
    l        = rs.get("losses",   mp - w)
    wp       = rs.get("win_pct",  round(w/mp*100,1) if mp else 0)
    titles   = rs.get("titles",   0)
    strength = rs.get("strength", "—")
    tcolor   = rs.get("color",    "#f97316")

    # Champion banner
    if titles > 0:
        trophy_str = "🏆 " * titles
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{tcolor}22,{tcolor}11);
                    border:2px solid {tcolor};border-radius:14px;
                    padding:.9rem 1.5rem;margin-bottom:1rem;">
            <span style="font-size:1.6rem;">{trophy_str}</span>
            <span style="color:{tcolor};font-size:1.1rem;font-weight:800;margin-left:.5rem;">
                {titles}× IPL Champion</span>
            <span style="color:#8b949e;font-size:.85rem;margin-left:1rem;">
                Best Strength: <b style="color:#e6edf3;">{strength}</b></span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#1f2937;border:1px solid #30363d;border-radius:14px;
                    padding:.8rem 1.5rem;margin-bottom:1rem;">
            <p style="color:#8b949e;margin:0;">No IPL title yet &nbsp;|&nbsp;
            Best Strength: <b style="color:#e6edf3;">{strength}</b></p>
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches Played", mp)
    c2.metric("Wins",           w)
    c3.metric("Losses",         l)
    c4.metric("Win %",          f"{wp}%")
    c5.metric("IPL Titles",     f"{titles} 🏆" if titles else "0")

    tm            = matches[(matches["team1"]==team_sel)|(matches["team2"]==team_sel)]
    wins          = matches[matches["winner"]==team_sel]
    toss_w        = matches[matches["toss_winner"]==team_sel]
    toss_then_win = matches[(matches["toss_winner"]==team_sel)&(matches["winner"]==team_sel)]


    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Season-wise Performance</p>', unsafe_allow_html=True)

        REAL_SEASON_STATS = {
            "Chennai Super Kings": {
                2008:{"played":16,"won":9},  2009:{"played":15,"won":8},
                2010:{"played":16,"won":9},  2011:{"played":16,"won":11},
                2012:{"played":19,"won":10}, 2013:{"played":18,"won":12},
                2014:{"played":16,"won":10}, 2015:{"played":17,"won":10},
                2016:{"played":0, "won":0},  2017:{"played":0, "won":0},
                2018:{"played":16,"won":11}, 2019:{"played":17,"won":10},
                2020:{"played":14,"won":6},  2021:{"played":16,"won":11},
                2022:{"played":14,"won":4},  2023:{"played":16,"won":10},
                2024:{"played":14,"won":7},  2025:{"played":14,"won":8},
            },
            "Delhi Capitals": {
                2008:{"played":14,"won":7},  2009:{"played":14,"won":10},
                2010:{"played":14,"won":6},  2011:{"played":14,"won":4},
                2012:{"played":16,"won":11}, 2013:{"played":16,"won":3},
                2014:{"played":14,"won":4},  2015:{"played":14,"won":5},
                2016:{"played":14,"won":7},  2017:{"played":14,"won":5},
                2018:{"played":14,"won":5},  2019:{"played":14,"won":9},
                2020:{"played":16,"won":8},  2021:{"played":14,"won":10},
                2022:{"played":14,"won":7},  2023:{"played":14,"won":5},
                2024:{"played":14,"won":7},  2025:{"played":14,"won":7},
            },
            "Gujarat Titans": {
                2022:{"played":16,"won":12}, 2023:{"played":16,"won":10},
                2024:{"played":14,"won":5},  2025:{"played":14,"won":9},
            },
            "Kolkata Knight Riders": {
                2008:{"played":14,"won":6},  2009:{"played":14,"won":3},
                2010:{"played":14,"won":7},  2011:{"played":15,"won":8},
                2012:{"played":16,"won":12}, 2013:{"played":16,"won":8},
                2014:{"played":16,"won":11}, 2015:{"played":14,"won":7},
                2016:{"played":14,"won":7},  2017:{"played":14,"won":7},
                2018:{"played":14,"won":6},  2019:{"played":14,"won":6},
                2020:{"played":14,"won":7},  2021:{"played":14,"won":7},
                2022:{"played":14,"won":6},  2023:{"played":14,"won":6},
                2024:{"played":15,"won":11}, 2025:{"played":14,"won":5},
            },
            "Lucknow Super Giants": {
                2022:{"played":15,"won":9}, 2023:{"played":15,"won":8},
                2024:{"played":14,"won":7}, 2025:{"played":15,"won":6},
            },
            "Mumbai Indians": {
                2008:{"played":14,"won":7},  2009:{"played":14,"won":5},
                2010:{"played":16,"won":11}, 2011:{"played":16,"won":10},
                2012:{"played":17,"won":10}, 2013:{"played":19,"won":13},
                2014:{"played":15,"won":7},  2015:{"played":16,"won":10},
                2016:{"played":14,"won":7},  2017:{"played":17,"won":12},
                2018:{"played":14,"won":6},  2019:{"played":16,"won":11},
                2020:{"played":16,"won":11}, 2021:{"played":14,"won":7},
                2022:{"played":14,"won":4},  2023:{"played":16,"won":9},
                2024:{"played":14,"won":4},  2025:{"played":16,"won":9},
            },
            "Punjab Kings": {
                2008:{"played":15,"won":10}, 2009:{"played":14,"won":7},
                2010:{"played":14,"won":4},  2011:{"played":14,"won":7},
                2012:{"played":16,"won":8},  2013:{"played":16,"won":8},
                2014:{"played":17,"won":12}, 2015:{"played":14,"won":3},
                2016:{"played":14,"won":4},  2017:{"played":14,"won":7},
                2018:{"played":14,"won":6},  2019:{"played":14,"won":6},
                2020:{"played":14,"won":6},  2021:{"played":14,"won":6},
                2022:{"played":14,"won":7},  2023:{"played":14,"won":6},
                2024:{"played":14,"won":5},  2025:{"played":17,"won":10},
            },
            "Rajasthan Royals": {
                2008:{"played":16,"won":13}, 2009:{"played":14,"won":6},
                2010:{"played":14,"won":6},  2011:{"played":14,"won":6},
                2012:{"played":16,"won":7},  2013:{"played":18,"won":11},
                2014:{"played":14,"won":7},  2015:{"played":15,"won":7},
                2016:{"played":0, "won":0},  2017:{"played":0, "won":0},
                2018:{"played":15,"won":7},  2019:{"played":14,"won":5},
                2020:{"played":14,"won":6},  2021:{"played":14,"won":5},
                2022:{"played":17,"won":10}, 2023:{"played":14,"won":7},
                2024:{"played":15,"won":9},  2025:{"played":14,"won":8},
            },
            "Royal Challengers Bangalore": {
                2008:{"played":14,"won":4},  2009:{"played":16,"won":9},
                2010:{"played":16,"won":8},  2011:{"played":16,"won":10},
                2012:{"played":16,"won":8},  2013:{"played":16,"won":9},
                2014:{"played":14,"won":5},  2015:{"played":16,"won":8},
                2016:{"played":16,"won":9},  2017:{"played":14,"won":3},
                2018:{"played":14,"won":6},  2019:{"played":14,"won":5},
                2020:{"played":15,"won":7},  2021:{"played":15,"won":9},
                2022:{"played":16,"won":9},  2023:{"played":14,"won":7},
                2024:{"played":15,"won":7},  2025:{"played":16,"won":11},
            },
            "Sunrisers Hyderabad": {
                2013:{"played":17,"won":10}, 2014:{"played":14,"won":6},
                2015:{"played":14,"won":7},  2016:{"played":17,"won":11},
                2017:{"played":15,"won":8},  2018:{"played":17,"won":10},
                2019:{"played":15,"won":6},  2020:{"played":16,"won":8},
                2021:{"played":14,"won":3},  2022:{"played":14,"won":6},
                2023:{"played":14,"won":4},  2024:{"played":16,"won":9},
                2025:{"played":14,"won":6},  2026:{"played":9,"won":6},
            },
        }

        if team_sel in REAL_SEASON_STATS:
            sd = REAL_SEASON_STATS[team_sel]
            sw = pd.DataFrame([
                {"season": yr, "played": v["played"], "won": v["won"]}
                for yr, v in sd.items()
            ])
        else:
            sw = tm.groupby("season").size().reset_index(name="played")
            ww = wins.groupby("season").size().reset_index(name="won")
            sw = sw.merge(ww, on="season", how="left").fillna(0)
            sw["won"] = sw["won"].astype(int)

        CHAMP_YEARS_CSK = {2010, 2011, 2018, 2021, 2023}
        CSK_FINAL_YEARS = {2008, 2012, 2013, 2015, 2019}
        MI_CHAMP_YEARS  = {2013, 2015, 2017, 2019, 2020}
        MI_FINAL_YEAR   = {2010}
        DC_FINALS_YEAR  = {2020}
        GT_CHAMP_YEAR   = {2022}
        GT_FINAL_YEAR   = {2023}
        KKR_CHAMP_YEARS = {2012, 2014, 2024}
        KKR_FINAL_YEAR  = {2021}
        PBKS_FINAL_YEARS= {2014, 2025}
        RR_CHAMP_YEARS  = {2008}
        RR_FINAL_YEARS  = {2022}
        RCB_CHAMP_YEARS = {2025}
        RCB_FINAL_YEARS = {2009, 2011, 2016}
        SRH_CHAMP_YEARS = {2016}
        SRH_FINAL_YEARS = {2018, 2024}
        bar_colors = []
        for _, row in sw.iterrows():
            if team_sel == "Chennai Super Kings" and int(row["season"]) in CHAMP_YEARS_CSK:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Chennai Super Kings" and int(row["season"]) in CSK_FINAL_YEARS:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Mumbai Indians" and int(row["season"]) in MI_CHAMP_YEARS:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Mumbai Indians" and int(row["season"]) in MI_FINAL_YEAR:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Delhi Capitals" and int(row["season"]) in DC_FINALS_YEAR:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Gujarat Titans" and int(row["season"]) in GT_CHAMP_YEAR:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Gujarat Titans" and int(row["season"]) in GT_FINAL_YEAR:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Kolkata Knight Riders" and int(row["season"]) in KKR_CHAMP_YEARS:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Kolkata Knight Riders" and int(row["season"]) in KKR_FINAL_YEAR:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Punjab Kings" and int(row["season"]) in PBKS_FINAL_YEARS:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Rajasthan Royals" and int(row["season"]) in RR_CHAMP_YEARS:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Rajasthan Royals" and int(row["season"]) in RR_FINAL_YEARS:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Royal Challengers Bangalore" and int(row["season"]) in RCB_CHAMP_YEARS:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Royal Challengers Bangalore" and int(row["season"]) in RCB_FINAL_YEARS:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif team_sel == "Sunrisers Hyderabad" and int(row["season"]) in SRH_CHAMP_YEARS:
                bar_colors.append("#F5C000")   # gold = title
            elif team_sel == "Sunrisers Hyderabad" and int(row["season"]) in SRH_FINAL_YEARS:
                bar_colors.append("#C0C0C0")   # silver = runners-up
            elif row["played"] == 0:
                bar_colors.append("#30363d")   # grey = absent/banned
            else:
                bar_colors.append("#f97316")

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Played", x=sw["season"], y=sw["played"],
                             marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Won", x=sw["season"], y=sw["won"],
                             marker=dict(color=bar_colors)))
        fig.update_layout(barmode="group", **DARK,
                          margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Toss Impact</p>', unsafe_allow_html=True)
        # Real IPL toss impact stats (midpoints of given ranges)
        labels = ["Won Toss & Won Match", "Won Toss & Lost Match", "Lost Toss & Won Match"]
        vals   = [53, 23, 24]
        fig = px.pie(names=labels, values=vals, hole=0.4,
                     color_discrete_sequence=["#22c55e", "#ef4444", "#64748b"])
        fig.update_traces(texttemplate="%{label}<br>%{value}%", textinfo="label+percent")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Head-to-Head vs All Teams</p>', unsafe_allow_html=True)

    # Real H2H records per team
    REAL_H2H = {
        "Chennai Super Kings": {
            "Mumbai Indians":              {"Won": 19, "Lost": 21},  # Played 40
            "Kolkata Knight Riders":       {"Won": 20, "Lost": 11},  # Played 31
            "Royal Challengers Bangalore": {"Won": 22, "Lost": 14},  # Played 37 (1 NR)
            "Delhi Capitals":              {"Won": 20, "Lost": 10},  # Played 30
            "Punjab Kings":                {"Won": 15, "Lost": 15},  # Played 31 (1 Tied)
            "Rajasthan Royals":            {"Won": 15, "Lost": 16},  # Played 31
            "Sunrisers Hyderabad":         {"Won": 16, "Lost":  8},  # Played 24
            "Gujarat Titans":              {"Won":  3, "Lost":  4},  # Played 7
            "Lucknow Super Giants":        {"Won":  2, "Lost":  3},  # Played 6 (1 NR)
        },
        "Delhi Capitals": {
            "Chennai Super Kings":          {"Won": 12, "Lost": 20},  # Played 32, CSK leads
            "Royal Challengers Bangalore":  {"Won": 21, "Lost": 13},  # Played 35, DC leads (incl 1 NR)
            "Mumbai Indians":               {"Won": 17, "Lost": 21},  # Played 38, MI leads
            "Kolkata Knight Riders":        {"Won": 15, "Lost": 19},  # Played 35, KKR leads (incl 1 NR)
            "Rajasthan Royals":             {"Won": 15, "Lost": 15},  # Played 30, dead-even
            "Punjab Kings":                 {"Won": 17, "Lost": 18},  # Played 36, nearly even (incl 1 NR)
            "Sunrisers Hyderabad":          {"Won": 12, "Lost": 14},  # Played 27, SRH leads (incl 1 NR)
            "Gujarat Titans":               {"Won":  3, "Lost":  5},  # Played 8, GT leads
            "Lucknow Super Giants":         {"Won":  5, "Lost":  3},  # Played 8, DC leads
        },
        "Gujarat Titans": {
            "Chennai Super Kings":          {"Won": 4, "Lost": 4},  # Played 8, dead-even; 2023 Final — CSK won
            "Mumbai Indians":               {"Won": 5, "Lost": 3},  # Played 8; GT's highest 233/3 in Q2 2023
            "Royal Challengers Bangalore":  {"Won": 3, "Lost": 4},  # Played 7, RCB leads
            "Kolkata Knight Riders":        {"Won": 4, "Lost": 1},  # Played 6 (1 NR), GT dominant 4-1
            "Rajasthan Royals":             {"Won": 6, "Lost": 3},  # Played 9; GT beat RR twice in 2022 playoffs
            "Delhi Capitals":               {"Won": 5, "Lost": 3},  # Played 8; GT's lowest 89 vs DC 2024
            "Sunrisers Hyderabad":          {"Won": 5, "Lost": 1},  # Played 7 (1 NR), GT's best H2H
            "Punjab Kings":                 {"Won": 3, "Lost": 4},  # Played 7; only team with winning record vs GT
            "Lucknow Super Giants":         {"Won": 5, "Lost": 3},  # Played 8, GT leads
        },
        "Kolkata Knight Riders": {
            "Chennai Super Kings":          {"Won": 11, "Lost": 20},  # Played 31, CSK dominates
            "Mumbai Indians":               {"Won": 11, "Lost": 25},  # Played 36, MI massive dominance
            "Royal Challengers Bangalore":  {"Won": 20, "Lost": 15},  # Played 36 (1 NR), KKR clear edge
            "Delhi Capitals":               {"Won": 19, "Lost": 15},  # Played 35 (1 NR), KKR leads
            "Punjab Kings":                 {"Won": 21, "Lost": 13},  # Played 36 (2 NR), KKR leads
            "Rajasthan Royals":             {"Won": 17, "Lost": 14},  # Played 33 (2 NR), KKR leads
            "Sunrisers Hyderabad":          {"Won": 20, "Lost": 11},  # Played 31, KKR best older H2H
            "Gujarat Titans":               {"Won":  1, "Lost":  4},  # Played 6 (1 NR), GT dominant
            "Lucknow Super Giants":         {"Won":  2, "Lost":  5},  # Played 7, LSG leads 5-2
        },
        "Lucknow Super Giants": {
            "Chennai Super Kings":          {"Won": 3, "Lost": 2},  # Played 6 (1 NR), LSG leads
            "Mumbai Indians":               {"Won": 5, "Lost": 2},  # Played 7, LSG dominant
            "Royal Challengers Bangalore":  {"Won": 2, "Lost": 5},  # Played 7, RCB leads
            "Delhi Capitals":               {"Won": 3, "Lost": 4},  # Played 7, DC leads
            "Punjab Kings":                 {"Won": 4, "Lost": 2},  # Played 6, LSG leads (257/5 vs PBKS)
            "Rajasthan Royals":             {"Won": 2, "Lost": 4},  # Played 6, RR leads
            "Sunrisers Hyderabad":          {"Won": 3, "Lost": 3},  # Played 6, dead-even
            "Kolkata Knight Riders":        {"Won": 5, "Lost": 2},  # Played 7, LSG leads
            "Gujarat Titans":               {"Won": 2, "Lost": 5},  # Played 7, GT leads
        },
        "Mumbai Indians": {
            "Chennai Super Kings":          {"Won": 21, "Lost": 19},  # Played 40
            "Kolkata Knight Riders":        {"Won": 23, "Lost": 10},  # Played 33
            "Royal Challengers Bangalore":  {"Won": 19, "Lost": 13},  # Played 33 (1 NR)
            "Delhi Capitals":               {"Won": 19, "Lost": 16},  # Played 35
            "Punjab Kings":                 {"Won": 17, "Lost": 16},  # Played 34 (1 NR)
            "Rajasthan Royals":             {"Won": 15, "Lost": 14},  # Played 30 (1 NR)
            "Sunrisers Hyderabad":          {"Won": 12, "Lost": 10},  # Played 23 (1 NR)
            "Gujarat Titans":               {"Won": 2,  "Lost": 3},   # Played 5
            "Lucknow Super Giants":         {"Won": 1,  "Lost": 4},   # Played 5
        },
        "Punjab Kings": {
            "Chennai Super Kings":          {"Won": 15, "Lost": 15},  # Played 31 (1 Tied)
            "Mumbai Indians":               {"Won": 17, "Lost": 17},  # Played 35 (1 Tied)
            "Royal Challengers Bangalore":  {"Won": 18, "Lost": 18},  # Played 35
            "Kolkata Knight Riders":        {"Won": 13, "Lost": 21},  # Played 36 (2 NR)
            "Delhi Capitals":               {"Won": 18, "Lost": 16},  # Played 35 (1 Tied)
            "Rajasthan Royals":             {"Won": 12, "Lost": 18},  # Played 31 (1 Tied)
            "Sunrisers Hyderabad":          {"Won": 8,  "Lost": 15},  # Played 23
            "Gujarat Titans":               {"Won": 4,  "Lost": 3},   # Played 7
            "Lucknow Super Giants":         {"Won": 4,  "Lost": 3},   # Played 7
        },
        "Rajasthan Royals": {
            "Chennai Super Kings":          {"Won": 16, "Lost": 15},  # Played 31
            "Mumbai Indians":               {"Won": 14, "Lost": 15},  # Played 30 (1 NR)
            "Royal Challengers Bangalore":  {"Won": 14, "Lost": 15},  # Played 32 (3 NR)
            "Kolkata Knight Riders":        {"Won": 14, "Lost": 14},  # Played 30 (2 NR)
            "Delhi Capitals":               {"Won": 16, "Lost": 13},  # Played 29
            "Punjab Kings":                 {"Won": 16, "Lost": 12},  # Played 28
            "Sunrisers Hyderabad":          {"Won": 9,  "Lost": 11},  # Played 20
            "Gujarat Titans":               {"Won": 2,  "Lost": 5},   # Played 7
            "Lucknow Super Giants":         {"Won": 3,  "Lost": 2},   # Played 5
        },
        "Royal Challengers Bangalore": {
            "Chennai Super Kings":          {"Won": 14, "Lost": 22},  # Played 37 (1 NR)
            "Mumbai Indians":               {"Won": 15, "Lost": 21},  # Played 37 (1 Tied)
            "Kolkata Knight Riders":        {"Won": 15, "Lost": 21},  # Played 36
            "Delhi Capitals":               {"Won": 21, "Lost": 13},  # Played 36 (1 NR, 1 Tied)
            "Punjab Kings":                 {"Won": 19, "Lost": 18},  # Played 37
            "Rajasthan Royals":             {"Won": 17, "Lost": 15},  # Played 34 (2 NR)
            "Sunrisers Hyderabad":          {"Won": 12, "Lost": 13},  # Played 26 (1 Tied)
            "Gujarat Titans":               {"Won": 4,  "Lost": 4},   # Played 8
            "Lucknow Super Giants":         {"Won": 5,  "Lost": 2},   # Played 7
        },
        "Sunrisers Hyderabad": {
            "Chennai Super Kings":          {"Won": 8,  "Lost": 16},  # Played 24
            "Mumbai Indians":               {"Won": 11, "Lost": 14},  # Played 26 (1 Tied)
            "Royal Challengers Bangalore":  {"Won": 12, "Lost": 13},  # Played 26 (1 Tied)
            "Kolkata Knight Riders":        {"Won": 11, "Lost": 19},  # Played 31 (1 Tied)
            "Delhi Capitals":               {"Won": 14, "Lost": 11},  # Played 27 (1 Tied)
            "Punjab Kings":                 {"Won": 16, "Lost": 7},   # Played 23
            "Rajasthan Royals":             {"Won": 11, "Lost": 9},   # Played 20
            "Gujarat Titans":               {"Won": 1,  "Lost": 5},   # Played 6
            "Lucknow Super Giants":         {"Won": 2,  "Lost": 5},   # Played 7
        },
    }

    if team_sel in REAL_H2H:
        h2h_rows = []
        for opp, rec in REAL_H2H[team_sel].items():
            played = rec["Won"] + rec["Lost"]
            h2h_rows.append({
                "Opponent": opp.replace("Chennai Super Kings","CSK")
                               .replace("Royal Challengers Bangalore","RCB")
                               .replace("Mumbai Indians","MI")
                               .replace("Kolkata Knight Riders","KKR")
                               .replace("Delhi Capitals","DC")
                               .replace("Punjab Kings","PBKS")
                               .replace("Rajasthan Royals","RR")
                               .replace("Sunrisers Hyderabad","SRH")
                               .replace("Gujarat Titans","GT")
                               .replace("Lucknow Super Giants","LSG"),
                "Played": played,
                "Won":    rec["Won"],
                "Lost":   rec["Lost"],
                "Win%":   round(rec["Won"] / played * 100, 1) if played else 0,
            })
        h2h_df = pd.DataFrame(h2h_rows).sort_values("Won", ascending=False).reset_index(drop=True)
    else:
        h2h = []
        for opp in TEAMS:
            if opp == team_sel:
                continue
            face = matches[((matches["team1"]==team_sel)&(matches["team2"]==opp))|
                           ((matches["team1"]==opp)&(matches["team2"]==team_sel))]
            w = face[face["winner"]==team_sel]
            h2h.append({"Opponent": opp, "Played": len(face),
                         "Won": len(w), "Lost": len(face)-len(w),
                         "Win%": round(len(w)/len(face)*100,1) if len(face) else 0})
        h2h_df = pd.DataFrame(h2h).sort_values("Won", ascending=False).reset_index(drop=True)

    fig = px.bar(h2h_df, x="Opponent", y=["Won", "Lost"],
                 barmode="stack",
                 color_discrete_map={"Won": "#22c55e", "Lost": "#ef4444"},
                 text_auto=True)
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    # H2H table — Win% formatted to 1 decimal place
    h2h_display = h2h_df.copy()
    h2h_display["Win%"] = h2h_display["Win%"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(
        h2h_display.style.background_gradient(subset=["Won","Lost"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True
    )

    # Rivalry insights (CSK)
    RIVALRY_NOTES = {
        "Chennai Super Kings": [
            ("⚔️ Best H2H",            "vs SRH & DC (66.7%). CSK has traditionally dominated both the Hyderabad and Delhi franchises."),
            ("📉 Worst H2H",           "vs LSG (40%). Though a small sample size, Lucknow has been a tricky opponent for the Men in Yellow."),
            ("🏆 IPL Finals Record",   "Played a record 10 finals — Won 5, Lost 5. They have the most appearances in IPL history."),
            ("📊 Playoff Consistency", "Qualified for the playoffs in 13 out of 15 seasons played."),
            ("🏰 Chepauk Fortress",    "Holds one of the most intimidating home records, often utilizing spin-friendly tracks to stifle opponents."),
            ("🏏 Key Records",         "Suresh Raina: Top Scorer (4,687 runs). Dwayne Bravo: Top wicket-taker (140). Highest Score: 246/5 vs RR. Lowest: 79 vs MI."),
            ("👑 The Dhoni Legacy",    "MS Dhoni captained the side for 226 matches, the most by any player for a single T20 team."),
        ],
        "Delhi Capitals": [
            ("⚔️ Biggest Rival",       "MI vs DC — 38 meetings, MI leads 21–17. MI also beat DC in the 2020 IPL Final — the only final DC have reached."),
            ("💪 DC Dominates",        "vs RCB (21–13), vs LSG (5–3). Strong record against these teams."),
            ("📉 DC Struggles",        "vs CSK (12–20), vs MI (17–21), vs KKR (15–19), vs SRH (12–14), vs GT (3–5). Negative H2H against 5 teams."),
            ("🤝 Dead-even Rivalries", "DC vs RR: perfectly level 15–15 across 30 matches. DC vs PBKS: nearly even at 17–18."),
            ("🆕 Newer Rivals",        "GT leads DC 5–3 across 8 matches. DC leads LSG 5–3 across 8 matches."),
            ("📋 Best / Worst",        "Best: 2012 (11W/16), 2021 (10W/14, topped the table). Worst streak: No playoffs 6 years in a row (2013–2018)."),
            ("🏏 Key Records",         "Rishabh Pant — top scorer (3000+ runs). Amit Mishra — top wicket-taker (106 wkts in 99 matches). Lowest score: 66 vs MI (2017)."),
        ],
        "Gujarat Titans": [
            ("🏆 Champions 2022",      "Won the IPL title in their debut season — only 2nd team ever to do so after RR in 2008. Posted 12W/16 in the league stage."),
            ("🥈 Runners-Up 2023",     "2023 Final vs CSK: GT posted 214/4. CSK won off the very last ball via DLS. One of the greatest IPL finals ever."),
            ("💪 GT Dominates",        "vs SRH (5–1 in 6 results, 83.3%), vs KKR (4–1 in 5 results, 80.0%), vs RR (6–3), vs DC (5–3), vs LSG (5–3), vs MI (5–3)."),
            ("📉 GT Struggles",        "vs PBKS (3–4) — the ONLY team with a winning record vs GT. vs RCB (3–4) also negative."),
            ("⚡ Biggest Match",        "2023 IPL Final: GT vs CSK. Jadeja hit a six and a four off the last two balls. CSK won. Greatest IPL final ever."),
            ("📊 Best Win %",          "62.3% win rate — the highest of any current IPL franchise. GT's best season: 2022 (12W/16 — Champions)."),
            ("🏏 Key Records",         "Shubman Gill: 1700+ runs (top scorer). Shami: 48 wkts in first 2 seasons. Sai Sudharsan: Orange Cap 2025 (759 runs). Prasidh Krishna: Purple Cap 2025 (25 wkts)."),
        ],
        "Kolkata Knight Riders": [
            ("⚔️ Best H2H",            "KKR vs SRH — 20 wins from 31 matches (64.5%). Includes the dominant 2024 IPL final win where KKR bowled SRH out for 113 (lowest ever in a final) and chased it in 10.3 overs."),
            ("📉 Worst H2H",           "KKR vs MI — 11 wins from 36 matches (30.6%). MI have beaten KKR 25 times. KKR's second worst is vs LSG (2–5)."),
            ("🏆 IPL Finals Record",   "Played 4 finals — Won 3 (2012 vs CSK, 2014 vs PBKS, 2024 vs SRH), Lost 1 (2021 vs CSK)."),
            ("👑 2024 Dominant Season", "Won 11 of 14 league games, lost only 3 all season — equalled the record for fewest losses in an IPL season (set by RR in 2008)."),
            ("🔥 Longest Win Streak",  "KKR hold the all-time IPL record for the longest winning streak — 14 consecutive wins."),
            ("🏏 Key Records",         "Gautam Gambhir — top scorer for KKR. Sunil Narine — top wicket-taker (Player of the Tournament 2024: 488 runs, 17 wkts). Highest score: 272 vs DC. Lowest score: 25 vs PBKS."),
            ("❌ Worst Season",        "2009 — won only 3 from 14 matches, came last. Had a streak of 8 consecutive defeats that season."),
            ("🌟 Best Season",         "2024 — 11 wins from 14 league matches, lost only 3 all season including playoffs, won the title convincingly."),
        ],
        "Lucknow Super Giants": [
            ("⚔️ Best H2H",            "vs MI & KKR — LSG holds a 71.4% win rate (5 wins from 7) against both, proving they can handle the heavyweights."),
            ("📉 Worst H2H",           "vs RCB & GT — Both teams have consistently troubled LSG (only 2 wins from 7 vs both, a 28.6% win rate)."),
            ("🔥 Highest Total",       "257/5 vs PBKS (2023) — The second-highest team total in the history of the IPL."),
            ("❌ Lowest Total",        "82 all out vs GT (2022)."),
            ("👑 Opening Record",      "Quinton de Kock and KL Rahul's 210* vs KKR (2022) is the highest opening partnership in IPL history."),
            ("🏏 Key Records",         "KL Rahul — Top scorer (over 1,400 runs). Ravi Bishnoi — Leading wicket-taker. Marcus Stoinis — Scored 124* vs CSK in 2024 (highest chase score)."),
            ("📊 Playoffs Record",     "Reached playoffs in their first two seasons (2022, 2023) but finished 4th both times."),
        ],
        "Mumbai Indians": [
            ("⚔️ Best H2H",            "vs KKR (69.7%). MI has historically dominated KKR, including long winning streaks."),
            ("📉 Worst H2H",           "vs LSG (20%). They have struggled significantly against the Lucknow franchise since its inception."),
            ("🏆 IPL Finals Record",   "Played 6 finals — Won 5 (2013, 2015, 2017, 2019, 2020), Lost 1 (2010 vs CSK)."),
            ("🔄 2025 Comeback",       "After two consecutive last-place finishes (2022, 2024), MI bounced back in 2025 under Mahela Jayawardene to reach the playoffs."),
            ("🌟 MVP Record",          "Suryakumar Yadav won the 2025 MVP award with a record 717 runs in a single season for the franchise."),
            ("🏏 Key Records",         "Rohit Sharma: Top Scorer (>5,000 runs). Lasith Malinga/Jasprit Bumrah: Top wicket-takers. Highest Score: 247 vs SRH (2024). Lowest Score: 87 vs SRH (2018)."),
            ("🌍 Champions League",    "Won CLT20 titles in 2011 and 2013, the only IPL team to win it twice."),
        ],
        "Punjab Kings": [
            ("⚔️ Best H2H",            "vs GT & LSG (57.1%). PBKS has a slightly better record against the newer franchises."),
            ("📉 Worst H2H",           "vs SRH (34.8%). Historically, Hyderabad’s bowling has often choked the Punjab batting lineup."),
            ("🏆 IPL Finals Record",   "Played 2 finals — Lost both (2014 vs KKR, 2025 vs RCB)."),
            ("🛡️ Defensive Record",   "In 2025, PBKS successfully defended 111 against KKR, the lowest total ever defended in IPL history."),
            ("🔥 Chase Kings",         "PBKS holds the record for the highest successful chase in IPL history (262 vs KKR, 2024)."),
            ("🏏 Key Records",         "Shaun Marsh: Top Scorer (2,477 runs). Piyush Chawla: Top wicket-taker (84). Highest Score: 262/2 vs KKR. Lowest: 73 vs RPS."),
            ("🌟 Individual Best",     "Shreyas Iyer led from the front in 2025, scoring 604 runs to take them to their second final."),
        ],
        "Rajasthan Royals": [
            ("⚔️ Best H2H",            "vs LSG (60%) and PBKS (57.1%). RR generally performs well against the Punjab side."),
            ("📉 Worst H2H",           "vs GT (28.5%). Gujarat has been a bogey team for RR since their entry in 2022."),
            ("🏆 IPL Finals Record",   "Played 2 finals — Won 1 (2008 vs CSK), Lost 1 (2022 vs GT)."),
            ("👑 Inaugural Champs",    "RR was the first-ever team to win the IPL under the legendary Shane Warne."),
            ("🏰 Fortress Jaipur",     "They have one of the highest home win percentages at the Sawai Mansingh Stadium."),
            ("🏏 Key Records",         "Sanju Samson: Top Scorer (>3,700 runs). Yuzvendra Chahal: Leading wicket-taker in IPL history. Highest Score: 226/6 vs PBKS. Lowest: 58 vs RCB."),
            ("🌟 Individual Best",     "Jos Buttler scored 863 runs in the 2022 season, the second-highest by any player in a single IPL season."),
        ],
        "Royal Challengers Bangalore": [
            ("⚔️ Best H2H",            "vs Lucknow Super Giants (71.4%) and Delhi Capitals (61.4%)."),
            ("📉 Worst H2H",           "vs Chennai Super Kings (38.8%). Historically their most difficult opponent."),
            ("🏆 IPL Finals Record",   "Played 4 finals — Won 1 (2025 vs PBKS), Lost 3 (2009 vs DCH, 2011 vs CSK, 2016 vs SRH)."),
            ("👑 Defending Champions", "RCB enters the 2026 season as the defending champion for the first time in their history."),
            ("❌ The 49 Debacle",      "Holds the record for the lowest total in IPL history (49 vs KKR, 2017)."),
            ("🏏 Key Records",         "Virat Kohli: All-time leading scorer (8,661+ runs). Yuzvendra Chahal: Top wicket-taker (139). Highest Score: 263/5 vs PWI (2013)."),
            ("🌟 Individual Best",     "Chris Gayle's 175* (2013) remains the highest individual score in T20 history. Rajat Patidar led them to their maiden title in 2025."),
        ],
        "Sunrisers Hyderabad": [
            ("⚔️ Best H2H",            "vs Punjab Kings (69.5%). Historically, SRH has dominated PBKS more than any other older rival."),
            ("📉 Worst H2H",           "vs Gujarat Titans (16.6%). GT has proven extremely difficult for SRH since joining the league."),
            ("🏆 IPL Finals Record",   "Played 3 finals — Won 1 (2016 vs RCB), Lost 2 (2018 vs CSK, 2024 vs KKR)."),
            ("🔥 Record Breakers",     "SRH holds the record for the highest-ever team total in IPL history—287/3 against RCB in 2024."),
            ("🏃 2026 Chase",          "Recently pulled off the highest T20 chase at Wankhede, successfully hunting down 243 against MI."),
            ("🏏 Key Records",         "David Warner: Top Scorer (4,081 runs, 3 Orange Caps). Bhuvneshwar Kumar: Top wicket-taker (>157 wkts). Highest Score: 287/3 vs RCB. Lowest: 96 vs MI."),
            ("🌟 Current Leaders",     "As of May 2026, Abhishek Sharma (425 runs) and Bhuvneshwar Kumar (17 wkts) lead the charts for them."),
        ],
    }
    if team_sel in RIVALRY_NOTES:
        st.markdown('<p class="section-header">📋 Rivalry Insights</p>', unsafe_allow_html=True)
        for icon_title, note in RIVALRY_NOTES[team_sel]:
            st.markdown(f"""
            <div style="background:#1f2937;border-left:4px solid #f97316;
                        border-radius:0 10px 10px 0;padding:.7rem 1rem;margin-bottom:.5rem;">
                <b style="color:#f97316;">{icon_title}</b>
                <span style="color:#e6edf3;margin-left:.5rem;">{note}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PLAYER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Player Analysis":
    st.markdown("# 👤 Player Analysis")
    tab1, tab2 = st.tabs(["🏏 Batsmen", "🎯 Bowlers"])

    # ── BATSMEN ──
    with tab1:
        st.markdown("### Top 10 Batsmen (All Time)")
        # Overriding synthetic data with real IPL top 10 batsmen
        real_top_bat = pd.DataFrame([
            {"player": "Virat Kohli",     "team": "RCB",  "matches": 276, "total_runs": 9040, "balls_faced": 6747, "strike_rate": 133.97, "average": 40.00, "fours": 813, "sixes": 306, "fifties": 66, "hundreds": 8},
            {"player": "Rohit Sharma",    "team": "MI",   "matches": 276, "total_runs": 7183, "balls_faced": 5417, "strike_rate": 132.60, "average": 29.93, "fours": 638, "sixes": 284, "fifties": 43, "hundreds": 2},
            {"player": "Shikhar Dhawan",  "team": "PBKS", "matches": 222, "total_runs": 6769, "balls_faced": 5324, "strike_rate": 127.14, "average": 35.26, "fours": 768, "sixes": 152, "fifties": 51, "hundreds": 2},
            {"player": "David Warner",    "team": "DC",   "matches": 184, "total_runs": 6565, "balls_faced": 4697, "strike_rate": 139.77, "average": 40.52, "fours": 663, "sixes": 236, "fifties": 62, "hundreds": 4},
            {"player": "KL Rahul",        "team": "DC",   "matches": 141, "total_runs": 5655, "balls_faced": 4071, "strike_rate": 138.88, "average": 46.74, "fours": 405, "sixes": 232, "fifties": 40, "hundreds": 7},
            {"player": "Suresh Raina",    "team": "CSK",  "matches": 205, "total_runs": 5528, "balls_faced": 4042, "strike_rate": 136.73, "average": 32.52, "fours": 506, "sixes": 203, "fifties": 39, "hundreds": 1},
            {"player": "MS Dhoni",        "team": "CSK",  "matches": 264, "total_runs": 5439, "balls_faced": 3957, "strike_rate": 137.45, "average": 38.30, "fours": 363, "sixes": 264, "fifties": 24, "hundreds": 0},
            {"player": "Ajinkya Rahane",  "team": "CSK",  "matches": 194, "total_runs": 5194, "balls_faced": 4145, "strike_rate": 125.31, "average": 30.20, "fours": 500, "sixes": 103, "fifties": 31, "hundreds": 2},
            {"player": "AB de Villiers",  "team": "RCB",  "matches": 184, "total_runs": 5162, "balls_faced": 3403, "strike_rate": 151.69, "average": 39.71, "fours": 413, "sixes": 251, "fifties": 40, "hundreds": 3},
            {"player": "Sanju Samson",    "team": "CSK",  "matches": 176, "total_runs": 5008, "balls_faced": 3562, "strike_rate": 140.60, "average": 31.70, "fours": 350, "sixes": 230, "fifties": 27, "hundreds": 5},
        ])
        fig = px.bar(real_top_bat, x="player", y="total_runs",
                     color="strike_rate", color_continuous_scale="Oranges",
                     hover_data=["team", "matches", "balls_faced", "average", "fours", "sixes", "fifties", "hundreds"],
                     labels={"total_runs":"Total Runs","player":"Player", "strike_rate": "Strike Rate"})
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Batsman Deep-Dive")
        sel_bat = st.selectbox("Select Batsman", real_top_bat["player"].tolist())
        row = real_top_bat[real_top_bat["player"]==sel_bat].iloc[0]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Runs",         int(row["total_runs"]))
        c2.metric("Balls Faced",  int(row["balls_faced"]))
        c3.metric("Strike Rate",  row["strike_rate"])
        c4.metric("Average",      row["average"])
        c5.metric("4s / 6s",      f"{int(row['fours'])} / {int(row['sixes'])}")

        # run distribution per season
        REAL_BAT_SEASON_RUNS = {
            "Virat Kohli": [165, 246, 307, 557, 364, 634, 359, 505, 973, 308, 530, 464, 466, 405, 341, 639, 741, 657],
            "Rohit Sharma": [404, 362, 404, 372, 433, 538, 390, 482, 489, 333, 286, 405, 332, 381, 268, 332, 417, 418],
            "Shikhar Dhawan": [340, 40, 191, 400, 569, 311, 377, 353, 501, 479, 497, 521, 618, 587, 460, 373, 152, 0],
            "David Warner": [None, 163, 282, 324, 256, 410, 528, 562, 848, 641, None, 692, 548, 195, 432, 516, 168, 305],
            "KL Rahul": [None, None, None, None, None, 20, 166, 142, 397, None, 659, 593, 670, 626, 616, 274, 520, 550],
            "Suresh Raina": [421, 434, 520, 438, 441, 548, 523, 374, 399, 442, 445, 383, None, 160, None, None, None, None],
            "MS Dhoni": [414, 332, 287, 392, 358, 461, 371, 372, 284, 290, 455, 414, 200, 114, 232, 104, 161, 231],
            "Ajinkya Rahane": [4, 144, None, 120, 560, 488, 339, 540, 480, 382, 370, 393, 113, 8, 133, 326, 242, 532],
            "AB de Villiers": [95, 465, 111, 312, 319, 360, 395, 513, 687, 216, 480, 442, 454, 313, None, None, None, None],
            "Sanju Samson": [None, None, None, None, None, 206, 339, 204, 291, 386, 441, 342, 375, 484, 458, 362, 531, 471]
        }
        seasons_list = list(range(2008, 2026))
        if sel_bat in REAL_BAT_SEASON_RUNS:
            sr_df = pd.DataFrame({
                "season": seasons_list,
                "batsman_runs": REAL_BAT_SEASON_RUNS[sel_bat]
            }).dropna()
            fig2 = px.line(sr_df, x="season", y="batsman_runs", markers=True,
                           labels={"batsman_runs":"Runs","season":"Season"},
                           color_discrete_sequence=["#f97316"])
            fig2.update_xaxes(dtick=1)
            dark_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # ── BOWLERS ──
    with tab2:
        st.markdown("### Top 10 Bowlers (All Time)")
        real_top_bowl = pd.DataFrame([
            {"player": "Yuzvendra Chahal",    "matches": 182, "wickets": 228, "bowling_average": 23.13, "economy_rate": 8.03, "4w": 8, "5w": 1},
            {"player": "Bhuvneshwar Kumar",   "matches": 199, "wickets": 215, "bowling_average": 20.00, "economy_rate": 7.72, "4w": 10, "5w": 2},
            {"player": "Sunil Narine",        "matches": 196, "wickets": 199, "bowling_average": 22.09, "economy_rate": 6.80, "4w": 7, "5w": 1},
            {"player": "Piyush Chawla",       "matches": 192, "wickets": 192, "bowling_average": 26.60, "economy_rate": 7.96, "4w": 2, "5w": 0},
            {"player": "Ravichandran Ashwin", "matches": 221, "wickets": 187, "bowling_average": 29.80, "economy_rate": 7.20, "4w": 1, "5w": 0},
            {"player": "Jasprit Bumrah",      "matches": 153, "wickets": 185, "bowling_average": 22.02, "economy_rate": 7.30, "4w": 2, "5w": 2},
            {"player": "Dwayne Bravo",        "matches": 161, "wickets": 183, "bowling_average": 23.83, "economy_rate": 8.39, "4w": 2, "5w": 0},
            {"player": "Ravindra Jadeja",     "matches": 264, "wickets": 176, "bowling_average": 30.20, "economy_rate": 7.60, "4w": 4, "5w": 1},
            {"player": "Amit Mishra",         "matches": 162, "wickets": 174, "bowling_average": 23.90, "economy_rate": 7.37, "4w": 4, "5w": 1},
            {"player": "Lasith Malinga",      "matches": 122, "wickets": 170, "bowling_average": 19.79, "economy_rate": 7.14, "4w": 6, "5w": 1},
        ])
        fig = px.bar(real_top_bowl, x="player", y="wickets",
                     color="economy_rate", color_continuous_scale="Blues_r",
                     hover_data=["matches", "bowling_average", "4w", "5w"],
                     labels={"wickets":"Wickets","player":"Player", "economy_rate": "Economy"})
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Bowler Deep-Dive")
        sel_bowl = st.selectbox("Select Bowler", real_top_bowl["player"].tolist())
        row = real_top_bowl[real_top_bowl["player"]==sel_bowl].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Wickets",        int(row["wickets"]))
        c2.metric("Economy Rate",   row["economy_rate"])
        c3.metric("Bowling Avg",    row["bowling_average"])
        c4.metric("4W / 5W",        f"{int(row['4w'])} / {int(row['5w'])}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – TOURNAMENT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Tournament Insights":
    st.markdown("# 🏆 Tournament Insights")
    tab1, tab2, tab3 = st.tabs(["🏅 Points Table","🏟 Venue","📅 Season Winners"])

    # ── POINTS TABLE ──
    with tab1:
        sel_s = st.selectbox("Season", SEASONS, index=len(SEASONS)-1, key="pt_season")
        
        if int(sel_s) == 2025:
            pts = pd.DataFrame([
                {"Team": "Punjab Kings (Q)", "Matches": 14, "Won": 9, "Lost": 4, "NR": 1, "Pts": 19, "NRR": 0.372},
                {"Team": "Royal Challengers Bengaluru (Q)", "Matches": 14, "Won": 9, "Lost": 4, "NR": 1, "Pts": 19, "NRR": 0.301},
                {"Team": "Gujarat Titans (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.254},
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 1.142},
                {"Team": "Delhi Capitals (E)", "Matches": 14, "Won": 7, "Lost": 6, "NR": 1, "Pts": 15, "NRR": 0.011},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 6, "Lost": 7, "NR": 1, "Pts": 13, "NRR": -0.241},
                {"Team": "Lucknow Super Giants (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.376},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 5, "Lost": 7, "NR": 2, "Pts": 12, "NRR": -0.305},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.549},
                {"Team": "Chennai Super Kings (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.647},
            ])
        elif int(sel_s) == 2024:
            pts = pd.DataFrame([
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 9, "Lost": 3, "NR": 2, "Pts": 20, "NRR": 1.428},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.414},
                {"Team": "Rajasthan Royals (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.273},
                {"Team": "Royal Challengers Bengaluru (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.459},
                {"Team": "Chennai Super Kings (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.392},
                {"Team": "Delhi Capitals (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.377},
                {"Team": "Lucknow Super Giants (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.667},
                {"Team": "Gujarat Titans (E)", "Matches": 14, "Won": 5, "Lost": 7, "NR": 2, "Pts": 12, "NRR": -1.063},
                {"Team": "Punjab Kings (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.353},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.318},
            ])
        elif int(sel_s) == 2023:
            pts = pd.DataFrame([
                {"Team": "Gujarat Titans (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.809},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.652},
                {"Team": "Lucknow Super Giants (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.284},
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.044},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.148},
                {"Team": "Royal Challengers Bengaluru (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.135},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.239},
                {"Team": "Punjab Kings (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.304},
                {"Team": "Delhi Capitals (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.808},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.590},
            ])
        elif int(sel_s) == 2022:
            pts = pd.DataFrame([
                {"Team": "Gujarat Titans (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.316},
                {"Team": "Rajasthan Royals (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.298},
                {"Team": "Lucknow Super Giants (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.251},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.253},
                {"Team": "Delhi Capitals (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.204},
                {"Team": "Punjab Kings (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.126},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": 0.146},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.379},
                {"Team": "Chennai Super Kings (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.203},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.506},
            ])
        elif int(sel_s) == 2021:
            pts = pd.DataFrame([
                {"Team": "Delhi Capitals (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.481},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.455},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": -0.140},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.587},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.116},
                {"Team": "Punjab Kings (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.001},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.993},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 3, "Lost": 11, "NR": 0, "Pts": 6, "NRR": -0.545},
            ])
        elif int(sel_s) == 2020:
            pts = pd.DataFrame([
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 1.107},
                {"Team": "Delhi Capitals (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.109},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.608},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.172},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.214},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.162},
                {"Team": "Chennai Super Kings (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.455},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.569},
            ])
        elif int(sel_s) == 2019:
            pts = pd.DataFrame([
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.421},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.131},
                {"Team": "Delhi Capitals (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.044},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": 0.577},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": 0.028},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.251},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 5, "Lost": 8, "NR": 1, "Pts": 11, "NRR": -0.449},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 14, "Won": 5, "Lost": 8, "NR": 1, "Pts": 11, "NRR": -0.607},
            ])
        elif int(sel_s) == 2018:
            pts = pd.DataFrame([
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.284},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.253},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.070},
                {"Team": "Rajasthan Royals (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.250},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": 0.317},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": 0.129},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.502},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.222},
            ])
        elif int(sel_s) == 2017:
            pts = pd.DataFrame([
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.784},
                {"Team": "Rising Pune Supergiant (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.176},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.599},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 0.641},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.009},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.512},
                {"Team": "Gujarat Lions (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.412},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 14, "Won": 3, "Lost": 10, "NR": 1, "Pts": 6, "NRR": -1.299},
            ])
        elif int(sel_s) == 2016:
            pts = pd.DataFrame([
                {"Team": "Gujarat Lions (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": -0.374},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 0.932},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 0.245},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 0.106},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.146},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.102},
                {"Team": "Rising Pune Supergiants (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.453},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.646},
            ])
        elif int(sel_s) == 2015:
            pts = pd.DataFrame([
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.709},
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.043},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 7, "Lost": 5, "NR": 2, "Pts": 16, "NRR": 1.037},
                {"Team": "Rajasthan Royals (Q)", "Matches": 14, "Won": 7, "Lost": 5, "NR": 2, "Pts": 16, "NRR": 0.062},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 7, "Lost": 6, "NR": 1, "Pts": 15, "NRR": 0.253},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.252},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 5, "Lost": 8, "NR": 1, "Pts": 11, "NRR": -0.479},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 3, "Lost": 11, "NR": 0, "Pts": 6, "NRR": -1.436},
            ])
        elif int(sel_s) == 2014:
            pts = pd.DataFrame([
                {"Team": "Kings XI Punjab (Q)", "Matches": 14, "Won": 11, "Lost": 3, "NR": 0, "Pts": 22, "NRR": 0.968},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.418},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.385},
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.095},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.062},
                {"Team": "Sunrisers Hyderabad (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.399},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 14, "Won": 5, "Lost": 9, "NR": 0, "Pts": 10, "NRR": -0.607},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 2, "Lost": 12, "NR": 0, "Pts": 4, "NRR": -1.182},
            ])
        elif int(sel_s) == 2013:
            pts = pd.DataFrame([
                {"Team": "Mumbai Indians (Q)", "Matches": 16, "Won": 11, "Lost": 5, "NR": 0, "Pts": 22, "NRR": 0.759},
                {"Team": "Chennai Super Kings (Q)", "Matches": 16, "Won": 11, "Lost": 5, "NR": 0, "Pts": 22, "NRR": 0.530},
                {"Team": "Rajasthan Royals (Q)", "Matches": 16, "Won": 10, "Lost": 6, "NR": 0, "Pts": 20, "NRR": 0.322},
                {"Team": "Sunrisers Hyderabad (Q)", "Matches": 16, "Won": 10, "Lost": 6, "NR": 0, "Pts": 20, "NRR": 0.115},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 16, "Won": 9, "Lost": 7, "NR": 0, "Pts": 18, "NRR": 0.608},
                {"Team": "Kings XI Punjab (E)", "Matches": 16, "Won": 8, "Lost": 8, "NR": 0, "Pts": 16, "NRR": -0.216},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 16, "Won": 6, "Lost": 10, "NR": 0, "Pts": 12, "NRR": -0.095},
                {"Team": "Pune Warriors (E)", "Matches": 16, "Won": 4, "Lost": 12, "NR": 0, "Pts": 8, "NRR": -1.006},
                {"Team": "Delhi Daredevils (E)", "Matches": 16, "Won": 3, "Lost": 13, "NR": 0, "Pts": 6, "NRR": -0.848},
            ])
        elif int(sel_s) == 2012:
            pts = pd.DataFrame([
                {"Team": "Delhi Daredevils (Q)", "Matches": 16, "Won": 11, "Lost": 5, "NR": 0, "Pts": 22, "NRR": 0.617},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 16, "Won": 10, "Lost": 5, "NR": 1, "Pts": 21, "NRR": 0.561},
                {"Team": "Mumbai Indians (Q)", "Matches": 16, "Won": 10, "Lost": 6, "NR": 0, "Pts": 20, "NRR": 0.151},
                {"Team": "Chennai Super Kings (Q)", "Matches": 16, "Won": 8, "Lost": 7, "NR": 1, "Pts": 17, "NRR": 0.158},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 16, "Won": 8, "Lost": 7, "NR": 1, "Pts": 17, "NRR": -0.022},
                {"Team": "Kings XI Punjab (E)", "Matches": 16, "Won": 8, "Lost": 8, "NR": 0, "Pts": 16, "NRR": -0.216},
                {"Team": "Rajasthan Royals (E)", "Matches": 16, "Won": 7, "Lost": 9, "NR": 0, "Pts": 14, "NRR": -0.201},
                {"Team": "Deccan Chargers (E)", "Matches": 16, "Won": 4, "Lost": 11, "NR": 1, "Pts": 9, "NRR": -0.509},
                {"Team": "Pune Warriors (E)", "Matches": 16, "Won": 4, "Lost": 12, "NR": 0, "Pts": 8, "NRR": -0.551},
            ])
        elif int(sel_s) == 2011:
            pts = pd.DataFrame([
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 9, "Lost": 4, "NR": 1, "Pts": 19, "NRR": 0.326},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.443},
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 9, "Lost": 5, "NR": 0, "Pts": 18, "NRR": 0.040},
                {"Team": "Kolkata Knight Riders (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": 0.433},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.051},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 6, "Lost": 7, "NR": 1, "Pts": 13, "NRR": -0.691},
                {"Team": "Deccan Chargers (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.756},
                {"Team": "Kochi Tuskers Kerala (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.764},
                {"Team": "Pune Warriors (E)", "Matches": 14, "Won": 4, "Lost": 9, "NR": 1, "Pts": 9, "NRR": -0.134},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 4, "Lost": 9, "NR": 1, "Pts": 9, "NRR": -0.448},
            ])
        elif int(sel_s) == 2010:
            pts = pd.DataFrame([
                {"Team": "Mumbai Indians (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 1.084},
                {"Team": "Deccan Chargers (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.297},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.274},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.219},
                {"Team": "Delhi Daredevils (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.341},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.341},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.514},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.478},
            ])
        elif int(sel_s) == 2009:
            pts = pd.DataFrame([
                {"Team": "Delhi Daredevils (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.311},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 8, "Lost": 5, "NR": 1, "Pts": 17, "NRR": 0.951},
                {"Team": "Royal Challengers Bangalore (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.191},
                {"Team": "Deccan Chargers (Q)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": 0.203},
                {"Team": "Kings XI Punjab (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.483},
                {"Team": "Rajasthan Royals (E)", "Matches": 14, "Won": 6, "Lost": 7, "NR": 1, "Pts": 13, "NRR": -0.352},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 6, "Lost": 8, "NR": 0, "Pts": 12, "NRR": -0.352},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 5, "Lost": 8, "NR": 1, "Pts": 11, "NRR": -0.192},
            ])
        elif int(sel_s) == 2008:
            pts = pd.DataFrame([
                {"Team": "Rajasthan Royals (Q)", "Matches": 14, "Won": 11, "Lost": 3, "NR": 0, "Pts": 22, "NRR": 0.632},
                {"Team": "Kings XI Punjab (Q)", "Matches": 14, "Won": 10, "Lost": 4, "NR": 0, "Pts": 20, "NRR": 0.509},
                {"Team": "Chennai Super Kings (Q)", "Matches": 14, "Won": 8, "Lost": 6, "NR": 0, "Pts": 16, "NRR": -0.192},
                {"Team": "Delhi Daredevils (Q)", "Matches": 14, "Won": 7, "Lost": 6, "NR": 1, "Pts": 15, "NRR": 0.342},
                {"Team": "Mumbai Indians (E)", "Matches": 14, "Won": 7, "Lost": 7, "NR": 0, "Pts": 14, "NRR": -0.188},
                {"Team": "Kolkata Knight Riders (E)", "Matches": 14, "Won": 6, "Lost": 7, "NR": 1, "Pts": 13, "NRR": -0.147},
                {"Team": "Royal Challengers Bangalore (E)", "Matches": 14, "Won": 4, "Lost": 10, "NR": 0, "Pts": 8, "NRR": -0.524},
                {"Team": "Deccan Chargers (E)", "Matches": 14, "Won": 2, "Lost": 12, "NR": 0, "Pts": 4, "NRR": -0.941},
            ])
        else:
            pts = compute_nrr(matches, deliveries, sel_s)

        st.markdown(f"### {sel_s} Points Table")

        # Add Rank column starting from 1
        pts_display = pts.copy().head(10).reset_index(drop=True)
        pts_display.insert(0, "Rank", range(1, len(pts_display) + 1))
        pts_display = pts_display.set_index("Rank")

        st.dataframe(
            pts_display.style.background_gradient(subset=["Pts", "NRR"], cmap="Oranges"),
            use_container_width=True,
        )
        fig = px.bar(pts.head(10), x="Team", y="Pts", color="NRR",
                     color_continuous_scale="RdYlGn",
                     title=f"{sel_s} Points Table (Top 10)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── VENUE ──
    with tab2:
        st.markdown("### Top Venues by Matches Hosted")
        vc = pd.DataFrame([
            {"venue": "Wankhede Stadium (Mumbai)", "matches": 130},
            {"venue": "M. Chinnaswamy Stadium (Bengaluru)", "matches": 106},
            {"venue": "Eden Gardens (Kolkata)", "matches": 104},
            {"venue": "Arun Jaitley Stadium (Delhi)", "matches": 101},
            {"venue": "M. A. Chidambaram Stadium (Chennai)", "matches": 96},
            {"venue": "Rajiv Gandhi Intl Stadium (Hyderabad)", "matches": 87},
            {"venue": "Sawai Mansingh Stadium (Jaipur)", "matches": 66},
            {"venue": "I.S. Bindra PCA Stadium (Mohali)", "matches": 61},
            {"venue": "MCA Stadium (Pune)", "matches": 51},
            {"venue": "Narendra Modi Stadium (Ahmedabad)", "matches": 49},
            {"venue": "Dubai Intl Cricket Stadium (UAE)", "matches": 46},
            {"venue": "Zayed Cricket Stadium (UAE)", "matches": 37},
        ])
        fig = px.bar(vc, x="matches", y="venue", orientation="h",
                     color="matches", color_continuous_scale="Viridis")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Average Score Per Venue (Batting Friendliness)")
        venue_avg = pd.DataFrame([
            {"venue": "M. Chinnaswamy Stadium", "avg_score": 188, "avg_runs_per_over": 8.89, "pitch": "Extreme Batting"},
            {"venue": "Wankhede Stadium", "avg_score": 178, "avg_runs_per_over": 8.66, "pitch": "High Scoring"},
            {"venue": "Narendra Modi Stadium", "avg_score": 171, "avg_runs_per_over": 8.45, "pitch": "Balanced"},
            {"venue": "Arun Jaitley Stadium", "avg_score": 170, "avg_runs_per_over": 8.40, "pitch": "Batting Friendly"},
            {"venue": "Sawai Mansingh Stadium", "avg_score": 169, "avg_runs_per_over": 8.39, "pitch": "Balanced"},
            {"venue": "I.S. Bindra PCA Stadium", "avg_score": 176, "avg_runs_per_over": 8.35, "pitch": "Batting Friendly"},
            {"venue": "Rajiv Gandhi Stadium", "avg_score": 174, "avg_runs_per_over": 8.34, "pitch": "Batting Friendly"},
            {"venue": "Eden Gardens", "avg_score": 172, "avg_runs_per_over": 8.30, "pitch": "Balanced"},
            {"venue": "MA Chidambaram Stadium", "avg_score": 165, "avg_runs_per_over": 7.95, "pitch": "Spin Friendly"},
            {"venue": "Ekana Cricket Stadium", "avg_score": 168, "avg_runs_per_over": 7.80, "pitch": "Balanced/Slow"},
        ]).sort_values("avg_runs_per_over", ascending=False)
        fig2 = px.bar(venue_avg, x="avg_runs_per_over", y="venue",
                      orientation="h", color="avg_runs_per_over",
                      hover_data=["avg_score", "pitch"],
                      labels={"avg_runs_per_over": "Avg Runs Per Over", "venue": "Venue", "avg_score": "Avg 1st Inn Score", "pitch": "Pitch Nature"},
                      color_continuous_scale="Oranges")
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        dark_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── SEASON WINNERS ──
    with tab3:
        st.markdown("### Season-wise Champions")
        wdf = pd.DataFrame([
            {"Season": 2008, "Champion": "Rajasthan Royals", "Orange Cap Winner": "Shaun Marsh (616)", "Purple Cap Winner": "Sohail Tanvir (22)"},
            {"Season": 2009, "Champion": "Deccan Chargers", "Orange Cap Winner": "Matthew Hayden (572)", "Purple Cap Winner": "RP Singh (23)"},
            {"Season": 2010, "Champion": "Chennai Super Kings", "Orange Cap Winner": "Sachin Tendulkar (618)", "Purple Cap Winner": "Pragyan Ojha (21)"},
            {"Season": 2011, "Champion": "Chennai Super Kings", "Orange Cap Winner": "Chris Gayle (608)", "Purple Cap Winner": "Lasith Malinga (28)"},
            {"Season": 2012, "Champion": "Kolkata Knight Riders", "Orange Cap Winner": "Chris Gayle (733)", "Purple Cap Winner": "Morne Morkel (25)"},
            {"Season": 2013, "Champion": "Mumbai Indians", "Orange Cap Winner": "Michael Hussey (733)", "Purple Cap Winner": "Dwayne Bravo (32)"},
            {"Season": 2014, "Champion": "Kolkata Knight Riders", "Orange Cap Winner": "Robin Uthappa (660)", "Purple Cap Winner": "Mohit Sharma (23)"},
            {"Season": 2015, "Champion": "Mumbai Indians", "Orange Cap Winner": "David Warner (562)", "Purple Cap Winner": "Dwayne Bravo (26)"},
            {"Season": 2016, "Champion": "Sunrisers Hyderabad", "Orange Cap Winner": "Virat Kohli (973)", "Purple Cap Winner": "Bhuvneshwar Kumar (23)"},
            {"Season": 2017, "Champion": "Mumbai Indians", "Orange Cap Winner": "David Warner (641)", "Purple Cap Winner": "Bhuvneshwar Kumar (26)"},
            {"Season": 2018, "Champion": "Chennai Super Kings", "Orange Cap Winner": "Kane Williamson (735)", "Purple Cap Winner": "Andrew Tye (24)"},
            {"Season": 2019, "Champion": "Mumbai Indians", "Orange Cap Winner": "David Warner (692)", "Purple Cap Winner": "Imran Tahir (26)"},
            {"Season": 2020, "Champion": "Mumbai Indians", "Orange Cap Winner": "KL Rahul (670)", "Purple Cap Winner": "Kagiso Rabada (30)"},
            {"Season": 2021, "Champion": "Chennai Super Kings", "Orange Cap Winner": "Ruturaj Gaikwad (635)", "Purple Cap Winner": "Harshal Patel (32)"},
            {"Season": 2022, "Champion": "Gujarat Titans", "Orange Cap Winner": "Jos Buttler (863)", "Purple Cap Winner": "Yuzvendra Chahal (27)"},
            {"Season": 2023, "Champion": "Chennai Super Kings", "Orange Cap Winner": "Shubman Gill (890)", "Purple Cap Winner": "Mohammed Shami (28)"},
            {"Season": 2024, "Champion": "Kolkata Knight Riders", "Orange Cap Winner": "Virat Kohli (741)", "Purple Cap Winner": "Harshal Patel (24)"},
            {"Season": 2025, "Champion": "Royal Challengers Bengaluru", "Orange Cap Winner": "Sai Sudharsan (759)", "Purple Cap Winner": "Prasidh Krishna (25)"},
        ]).sort_values("Season", ascending=False)
        st.dataframe(wdf.set_index("Season"), use_container_width=True)

        fig = px.histogram(wdf, x="Champion", color="Champion",
                           title="Championship Count",
                           color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title="Team", yaxis_title="Titles")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – WIN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Win Predictor":
    st.markdown("# 🔮 Match Win Predictor")
    st.markdown("Uses a Gradient Boosting Classifier trained on IPL match history.")

    @st.cache_resource(show_spinner="Training advanced model…")
    def train_model_v7():
        return build_model(matches)

    model, encoders, metrics, stats_pack, report, feature_cols = train_model_v7()
    
    st.success("Model trained successfully! 🚀")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m_col2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m_col3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    m_col4.metric("F1-Score", f"{metrics['f1']*100:.1f}%")

    st.markdown("---")
    venues = sorted(matches["venue"].dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        t1   = st.selectbox("Team 1", TEAMS, key="t1")
        tw   = st.selectbox("Toss Winner", TEAMS, key="tw")
        td   = st.selectbox("Toss Decision", ["bat","field"], key="td")
    with col2:
        t2   = st.selectbox("Team 2", TEAMS, index=1, key="t2")
        ven  = st.selectbox("Venue", venues, key="ven")

    if st.button("🔮 Predict Winner", use_container_width=True):
        if t1 == t2:
            st.error("Please select two different teams.")
        else:
            winner, conf = predict_winner(model, encoders, stats_pack, t1, t2, tw, td, ven)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1f2937,#111827);
                        border:2px solid #f97316; border-radius:16px; padding:2rem;
                        text-align:center; margin-top:1rem;">
                <h2 style="color:#f97316; margin:0;">Predicted Winner</h2>
                <h1 style="color:#f0f6fc; font-size:2.5rem; margin:.5rem 0;">{winner}</h1>
                <p style="color:#8b949e; font-size:1.1rem;">Confidence: <b style="color:#22c55e;">{conf}%</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Feature Importance")
    coef = getattr(model, "feature_importances_", getattr(model, "coef_", [[]])[0])
    fi = pd.DataFrame({"Feature": feature_cols, "Importance": coef})
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="RdYlGn")
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Oranges",
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Team 2 Win", "Team 1 Win"], y=["Team 2 Win", "Team 1 Win"])
    dark_fig(fig_cm)
    st.plotly_chart(fig_cm, use_container_width=True)

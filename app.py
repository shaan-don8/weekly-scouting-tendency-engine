from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from weekly_scout_core import (
    SECTION_TITLES,
    build_film_review_queue_v2,
    build_identity_card,
    build_markdown_report,
    build_v2_opponent_report,
    collapse_related_balance_warnings,
    create_staff_note,
    create_staff_shift_note,
    format_identity_table,
    load_normalized_data,
    prepare_tendency_fields,
)

st.set_page_config(
    page_title="NFL Weekly Offensive Tendency Scout",
    page_icon="🏈",
    layout="wide",
)

DATA_PATH = Path(os.environ.get(
    "WEEKLY_SCOUT_DATA",
    "data/weekly_scout_play_level_normalized_2022_2025.parquet",
))
VALIDATION_DIR = Path(os.environ.get("WEEKLY_SCOUT_VALIDATION_DIR", "data"))

@st.cache_data(show_spinner="Loading normalized play-level data...")
def load_app_data(path: str) -> pd.DataFrame:
    return prepare_tendency_fields(load_normalized_data(path))

@st.cache_data(show_spinner="Building opponent report...")
def build_cached_report(data: pd.DataFrame, team: str, season: int, through_week: int):
    report = build_v2_opponent_report(data, team, season, through_week)
    report["balance_warnings"] = collapse_related_balance_warnings(report["balance_warnings"], max_items=3)
    identity = build_identity_card(data, team, season, through_week, recent_games=4)
    queue_long, queue_staff = build_film_review_queue_v2(report, data, team, season, through_week)
    markdown = build_markdown_report(team, season, through_week, identity, report, queue_staff)
    return report, identity, queue_long, queue_staff, markdown


def render_alert_section(title: str, frame: pd.DataFrame, shifts: bool = False) -> None:
    st.subheader(title)
    if frame.empty:
        st.caption("No qualifying alerts.")
        return
    for idx, (_, row) in enumerate(frame.iterrows(), start=1):
        note = create_staff_shift_note(row) if shifts else create_staff_note(row)
        st.markdown(f"**{idx}.** {note}")


def load_optional_csv(filename: str) -> pd.DataFrame:
    path = VALIDATION_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

st.title("NFL Weekly Offensive Tendency Scout")
st.caption("Opponent-specific pre-snap call tells, personnel priorities, balance warnings, identity shifts, and a film-review queue.")

if not DATA_PATH.exists():
    st.error(f"Data file not found: {DATA_PATH}")
    st.code("Set WEEKLY_SCOUT_DATA or place the normalized parquet at data/weekly_scout_play_level_normalized_2022_2025.parquet")
    st.stop()

try:
    tendency_data = load_app_data(str(DATA_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

available_seasons = sorted(tendency_data["season"].dropna().astype(int).unique().tolist(), reverse=True)

with st.sidebar:
    st.header("Report controls")
    season = st.selectbox("Season", available_seasons, index=0)
    season_frame = tendency_data.loc[tendency_data["season"].eq(season)].copy()
    teams = sorted(season_frame["posteam"].dropna().unique().tolist())
    default_team_index = teams.index("DET") if "DET" in teams else 0
    team = st.selectbox("Opponent offense", teams, index=default_team_index)
    team_frame = season_frame.loc[season_frame["posteam"].eq(team)].copy()
    weeks = sorted(team_frame["week"].dropna().astype(int).unique().tolist())
    through_week = st.selectbox("Through week", weeks, index=len(weeks) - 1)
    st.divider()
    st.caption("Frozen v2 rules: high-confidence live tells first; stable or strengthening recency prioritized; weakening and reversing signals moved to identity shifts.")

report, identity, queue_long, queue_staff, markdown_report = build_cached_report(tendency_data, team, int(season), int(through_week))

report_tab, film_tab, validation_tab, methodology_tab = st.tabs([
    "Opponent report", "Film-review queue", "Validation", "Methodology"
])

with report_tab:
    st.header(f"{team} offensive tendency scout")
    st.caption(f"{season} season · through Week {through_week}")
    st.subheader("Offensive identity snapshot")
    st.dataframe(format_identity_table(identity), hide_index=True, use_container_width=True)

    render_alert_section(SECTION_TITLES["active_live_tells"], report["active_live_tells"])
    render_alert_section(SECTION_TITLES["emerging_live_tells"], report["emerging_live_tells"])
    render_alert_section(SECTION_TITLES["enriched_film_priorities"], report["enriched_film_priorities"])
    render_alert_section(SECTION_TITLES["balance_warnings"], report["balance_warnings"])
    render_alert_section(SECTION_TITLES["identity_shifts"], report["identity_shifts"], shifts=True)

    st.download_button(
        "Download Markdown report",
        data=markdown_report,
        file_name=f"{team}_{season}_through_week_{through_week}_internal_scout_report.md",
        mime="text/markdown",
    )

with film_tab:
    st.header("Film-review queue")
    st.caption("Each staff-facing row is a unique snap. A play can retain multiple surfaced reasons without appearing twice.")
    st.metric("Unique plays", len(queue_staff))
    if queue_staff.empty:
        st.info("No representative plays were surfaced for this report.")
    else:
        display_columns = [
            "week", "game_id", "play_id", "play_family", "epa", "selection_reasons",
            "alert_situations", "call_families", "description",
        ]
        st.dataframe(queue_staff[display_columns], hide_index=True, use_container_width=True)
        st.download_button(
            "Download staff film queue",
            data=queue_staff.to_csv(index=False),
            file_name=f"{team}_{season}_through_week_{through_week}_film_review_queue_staff.csv",
            mime="text/csv",
        )
        with st.expander("Show long-form analytical queue"):
            st.dataframe(queue_long, hide_index=True, use_container_width=True)
            st.download_button(
                "Download long-form queue",
                data=queue_long.to_csv(index=False),
                file_name=f"{team}_{season}_through_week_{through_week}_film_review_queue_long.csv",
                mime="text/csv",
            )

with validation_tab:
    st.header("2023–2025 rolling next-game validation")
    by_card = load_optional_csv("league_2023_2025_validation_by_card.csv")
    by_season = load_optional_csv("league_2023_2025_validation_by_season.csv")
    by_confidence = load_optional_csv("league_2023_2025_validation_by_confidence.csv")
    bootstrap = load_optional_csv("league_2023_2025_cluster_bootstrap.csv")
    if by_card.empty:
        st.info("Place the exported validation CSV files in the data directory to populate this tab.")
    else:
        st.markdown("Positive lift means the opponent-specific estimate improved on the contextual league baseline.")
        st.subheader("By report module")
        st.dataframe(by_card, hide_index=True, use_container_width=True)
        if not by_season.empty:
            st.subheader("By season")
            st.dataframe(by_season, hide_index=True, use_container_width=True)
        if not by_confidence.empty:
            st.subheader("By confidence tier")
            st.dataframe(by_confidence, hide_index=True, use_container_width=True)
        if not bootstrap.empty:
            st.subheader("Team-game-cluster bootstrap")
            st.dataframe(bootstrap, hide_index=True, use_container_width=True)

with methodology_tab:
    st.header("Methodology")
    st.markdown(
        """
### Purpose
Surface stable, opponent-specific pre-snap run-pass tendencies that can guide weekly defensive preparation and prioritize film review.

### Live-weekly layer
Uses fields that can support a realistic in-season refresh: quarterback location, motion, down, distance, previous drive play, and standard game context.

### Historical enrichment layer
Adds personnel and formation fields for deeper film-review priorities.

### Alert types
- **Call tell:** the opponent's most likely call is unusually strong relative to the league context.
- **Balance warning:** the offense still leans toward the conventional call, but the minority call appears materially more often than the league baseline suggests.
- **Identity shift:** a season-long tendency has weakened or reversed over the most recent four games.

### Validation
Rolling reports are generated using only information available through each report week and scored against the offense's next game. Opponent-specific probabilities are compared with contextual league expectations using Brier score and log loss.

### Limits
The tool prioritizes film study. It does not replace film, prescribe defensive calls, or establish that a productive constraint should be called more frequently.
        """
    )

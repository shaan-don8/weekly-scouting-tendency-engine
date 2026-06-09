from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Frozen v2 configuration
# -----------------------------------------------------------------------------

TENDENCY_TEMPLATES: list[dict[str, Any]] = [
    {"template": "QB location", "feature_layer": "LIVE_WEEKLY", "columns": ["qb_location_group"]},
    {"template": "Down + distance", "feature_layer": "LIVE_WEEKLY", "columns": ["down_group", "distance_bucket"]},
    {"template": "QB location + motion", "feature_layer": "LIVE_WEEKLY", "columns": ["qb_location_group", "motion_group"]},
    {"template": "QB location + down + distance", "feature_layer": "LIVE_WEEKLY", "columns": ["qb_location_group", "down_group", "distance_bucket"]},
    {"template": "QB location + motion + down", "feature_layer": "LIVE_WEEKLY", "columns": ["qb_location_group", "motion_group", "down_group"]},
    {"template": "Previous play + QB location", "feature_layer": "LIVE_WEEKLY", "columns": ["prev_drive_play_family", "qb_location_group"]},
    {"template": "Personnel", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["personnel_code"]},
    {"template": "Formation", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["offense_formation_group"]},
    {"template": "QB location + personnel", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["qb_location_group", "personnel_code"]},
    {"template": "Formation + personnel", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["offense_formation_group", "personnel_code"]},
    {"template": "Personnel + motion", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["personnel_code", "motion_group"]},
    {"template": "Personnel + down + distance", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["personnel_code", "down_group", "distance_bucket"]},
    {"template": "QB location + personnel + down", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["qb_location_group", "personnel_code", "down_group"]},
    {"template": "Formation + personnel + down", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["offense_formation_group", "personnel_code", "down_group"]},
    {"template": "QB location + personnel + motion", "feature_layer": "HISTORICAL_ENRICHMENT", "columns": ["qb_location_group", "personnel_code", "motion_group"]},
]

LIVE_ACTIVE_TEMPLATES = {
    "QB location", "Down + distance", "QB location + motion",
    "QB location + down + distance", "QB location + motion + down",
    "Previous play + QB location",
}

ENRICHED_ACTIVE_TEMPLATES = {
    "Personnel", "Formation", "QB location + personnel", "Formation + personnel",
    "Personnel + motion", "Personnel + down + distance",
    "QB location + personnel + down", "Formation + personnel + down",
    "QB location + personnel + motion",
}

BALANCE_ACTIVE_TEMPLATES = {
    "Personnel", "Personnel + motion", "Personnel + down + distance",
    "QB location + personnel + down", "QB location + personnel + motion",
    "Down + distance", "QB location + motion + down",
    "QB location + down + distance", "Formation + personnel",
    "Formation + personnel + down",
}

IDENTITY_METRIC_ORDER = [
    "EPA / play", "Success rate", "Explosive-play rate", "Designed-run rate",
    "Dropback rate", "RPO rate", "Screen rate", "Motion rate",
    "Play action on pass calls", "Under-center rate", "11 personnel rate",
    "12 personnel rate",
]

SECTION_TITLES = {
    "active_live_tells": "Active Live-Weekly Tells",
    "emerging_live_tells": "Emerging Live Tendencies",
    "enriched_film_priorities": "Enriched Film-Review Priorities",
    "balance_warnings": "Balance Warnings",
    "identity_shifts": "Identity Shifts",
}

# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------

def load_normalized_data(path: str | Path) -> pd.DataFrame:
    """Load the normalized play-level parquet produced by the notebook pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized scouting parquet not found: {path}")
    return pd.read_parquet(path)


def prepare_tendency_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare regular-season offensive plays for the frozen tendency engine."""
    output = data.loc[data["season_type"].eq("REG")].copy()

    output["binary_call"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output.loc[output["play_family"].eq("designed_run"), "binary_call"] = "RUN"
    output.loc[output["play_family"].isin(["dropback", "screen"]), "binary_call"] = "PASS"

    output["down_group"] = pd.Series("OTHER", index=output.index, dtype="string")
    output.loc[output["down"].eq(1), "down_group"] = "1ST"
    output.loc[output["down"].eq(2), "down_group"] = "2ND"
    output.loc[output["down"].isin([3, 4]), "down_group"] = "3RD_4TH"

    output["motion_group"] = np.where(output["ftn_is_motion"].fillna(False), "MOTION", "NO_MOTION")
    output["motion_group"] = pd.Series(output["motion_group"], index=output.index, dtype="string")

    output = output.sort_values(["game_id", "posteam", "fixed_drive", "play_id"]).copy()
    sequence_group = output.groupby(["game_id", "posteam", "fixed_drive"], sort=False, dropna=False)
    output["prev_drive_play_family"] = sequence_group["play_family"].shift(1).fillna("START")
    output["prev_drive_success"] = sequence_group["success_flag"].shift(1)
    return output

# -----------------------------------------------------------------------------
# Identity snapshot
# -----------------------------------------------------------------------------

METRIC_SPECS: dict[str, tuple[str, Any, str]] = {
    "EPA / play": ("epa", "mean", "decimal"),
    "Success rate": ("success_flag", "mean", "percent"),
    "Explosive-play rate": ("explosive_play", "mean", "percent"),
    "Designed-run rate": ("run_call_flag", "mean", "percent"),
    "Dropback rate": ("pass_call_flag", "mean", "percent"),
    "RPO rate": ("rpo_call_flag", "mean", "percent"),
    "Screen rate": ("screen_call_flag", "mean", "percent"),
    "Motion rate": ("ftn_is_motion", "mean", "percent"),
    "Play-action rate": ("ftn_is_play_action", "mean", "percent"),
    "Play action on pass calls": ("play_action_on_pass_call", "mean", "percent"),
    "Under-center rate": ("under_center_flag", "mean", "percent"),
    "Shotgun rate": ("shotgun_location_flag", "mean", "percent"),
    "Pistol rate": ("pistol_location_flag", "mean", "percent"),
    "No-huddle rate": ("ftn_is_no_huddle", "mean", "percent"),
    "Neutral early-down dropback rate": ("neutral_early_down_pass_call", "mean", "percent"),
    "11 personnel rate": ("personnel_code", lambda values: values.eq("11").mean(), "percent"),
    "12 personnel rate": ("personnel_code", lambda values: values.eq("12").mean(), "percent"),
    "13+ personnel rate": ("n_off_te", lambda values: values.ge(3).mean(), "percent"),
    "21+ personnel rate": ("n_off_back", lambda values: values.ge(2).mean(), "percent"),
    "Extra-OL rate": ("jumbo_ol_flag", "mean", "percent"),
}


def aggregate_metric(frame: pd.DataFrame, column: str, aggregator: Any) -> float:
    if callable(aggregator):
        return float(aggregator(frame[column]))
    return float(frame[column].agg(aggregator))


def build_identity_card(
    data: pd.DataFrame,
    team: str,
    season: int,
    through_week: int | None = None,
    recent_games: int = 4,
) -> pd.DataFrame:
    season_data = data.loc[data["season"].eq(season)].copy()
    if through_week is None:
        through_week = int(season_data["week"].max())
    season_data = season_data.loc[season_data["week"].le(through_week)].copy()
    team_data = season_data.loc[season_data["posteam"].eq(team)].copy()
    if team_data.empty:
        raise ValueError(f"No offensive plays found for team={team!r}, season={season}, through_week={through_week}.")

    recent_game_ids = (
        team_data[["game_id", "week"]].drop_duplicates().sort_values(["week", "game_id"]).tail(recent_games)["game_id"].tolist()
    )
    recent_data = team_data.loc[team_data["game_id"].isin(recent_game_ids)].copy()
    rows: list[dict[str, Any]] = []
    for label, (column, aggregator, format_type) in METRIC_SPECS.items():
        league_rows = []
        for offense, offense_data in season_data.groupby("posteam", dropna=False):
            league_rows.append({"posteam": offense, "value": aggregate_metric(offense_data, column, aggregator)})
        league_table = pd.DataFrame(league_rows)
        season_value = aggregate_metric(team_data, column, aggregator)
        recent_value = aggregate_metric(recent_data, column, aggregator)
        league_avg = float(league_table["value"].mean())
        rank = int(league_table["value"].rank(method="min", ascending=False).loc[league_table["posteam"].eq(team)].iloc[0])
        rows.append({
            "metric": label,
            "season_to_date": season_value,
            "last_4_games": recent_value,
            "league_avg": league_avg,
            "league_rank": rank,
            "recent_shift": recent_value - season_value,
            "format_type": format_type,
        })
    return pd.DataFrame(rows)


def format_identity_table(identity_card: pd.DataFrame) -> pd.DataFrame:
    output = identity_card.loc[identity_card["metric"].isin(IDENTITY_METRIC_ORDER)].copy()
    output["metric_order"] = output["metric"].map({metric: order for order, metric in enumerate(IDENTITY_METRIC_ORDER)})
    output = output.sort_values("metric_order")

    def format_metric_value(row: pd.Series, column: str) -> str:
        value = row[column]
        if pd.isna(value):
            return ""
        return f"{float(value):.1%}" if row["format_type"] == "percent" else f"{float(value):+.3f}"

    return pd.DataFrame({
        "Metric": output["metric"],
        "Season": output.apply(format_metric_value, axis=1, column="season_to_date"),
        "Last 4": output.apply(format_metric_value, axis=1, column="last_4_games"),
        "NFL Avg": output.apply(format_metric_value, axis=1, column="league_avg"),
        "NFL Rank": output["league_rank"].astype(int).astype(str),
    })

# -----------------------------------------------------------------------------
# Alert engine
# -----------------------------------------------------------------------------

def filter_as_of(data: pd.DataFrame, season: int, through_week: int, baseline_years: int = 2) -> pd.DataFrame:
    min_season = season - baseline_years + 1
    return data.loc[
        data["season"].between(min_season, season)
        & (data["season"].lt(season) | data["week"].le(through_week))
    ].copy()


def make_context_mask(frame: pd.DataFrame, columns: list[str], values: tuple[Any, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column, value in zip(columns, values):
        mask &= frame[column].isna() if pd.isna(value) else frame[column].eq(value)
    return mask


def format_situation(columns: list[str], values: tuple[Any, ...]) -> str:
    label_map = {
        "qb_location_group": "QB", "personnel_code": "PERS", "offense_formation_group": "FORM",
        "motion_group": "MOTION", "down_group": "DOWN", "distance_bucket": "DIST",
        "prev_drive_play_family": "PREV",
    }
    return " | ".join(f"{label_map.get(column, column)}={value}" for column, value in zip(columns, values))


def context_signature(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    keys = frame[["game_id", "play_id"]].astype("string")
    return int(pd.util.hash_pandas_object(keys, index=False).sum())


def add_constraint_family(frame: pd.DataFrame, primary_call: str) -> pd.DataFrame:
    output = frame.copy()
    output["constraint_call"] = pd.Series("OTHER", index=output.index, dtype="string")
    if primary_call == "RUN":
        is_play_action_pass = (
            output["play_family"].isin(["dropback", "screen"])
            & pd.to_numeric(output["ftn_is_play_action"], errors="coerce").fillna(0).eq(1)
        )
        output.loc[is_play_action_pass, "constraint_call"] = "PLAY_ACTION_PASS"
        output.loc[output["play_family"].eq("screen") & ~is_play_action_pass, "constraint_call"] = "SCREEN"
        output.loc[output["play_family"].eq("dropback") & ~is_play_action_pass, "constraint_call"] = "STANDARD_DROPBACK"
        output.loc[output["play_family"].eq("rpo"), "constraint_call"] = "RPO"
    else:
        output.loc[output["play_family"].eq("designed_run"), "constraint_call"] = "DESIGNED_RUN"
        output.loc[output["play_family"].eq("rpo"), "constraint_call"] = "RPO"
    return output


def summarize_best_constraint(frame: pd.DataFrame, primary_call: str, min_constraint_plays: int = 3) -> dict[str, Any]:
    constraint_frame = add_constraint_family(frame, primary_call)
    constraint_frame = constraint_frame.loc[constraint_frame["constraint_call"].ne("OTHER")].copy()
    if constraint_frame.empty:
        return {"best_counter_call": pd.NA, "counter_plays": 0, "counter_epa": np.nan, "counter_success_rate": np.nan}
    summary = (
        constraint_frame.groupby("constraint_call", dropna=False)
        .agg(counter_plays=("play_family", "size"), counter_epa=("epa", "mean"), counter_success_rate=("success_flag", "mean"))
        .reset_index()
    )
    eligible = summary.loc[summary["counter_plays"].ge(min_constraint_plays)].copy()
    if eligible.empty:
        return {"best_counter_call": pd.NA, "counter_plays": 0, "counter_epa": np.nan, "counter_success_rate": np.nan}
    best = eligible.sort_values(["counter_epa", "counter_plays"], ascending=[False, False]).iloc[0]
    return {
        "best_counter_call": best["constraint_call"],
        "counter_plays": int(best["counter_plays"]),
        "counter_epa": float(best["counter_epa"]),
        "counter_success_rate": float(best["counter_success_rate"]),
    }


def build_tendency_alerts(
    data: pd.DataFrame,
    team: str,
    season: int,
    through_week: int,
    templates: list[dict[str, Any]] = TENDENCY_TEMPLATES,
    baseline_years: int = 2,
    min_team_plays: int = 18,
    min_league_plays: int = 150,
    prior_strength: float = 20.0,
    min_rate_delta: float = 0.07,
    call_tell_threshold: float = 0.60,
    neutral_only: bool = False,
) -> pd.DataFrame:
    as_of = filter_as_of(data, season, through_week, baseline_years)
    if neutral_only:
        as_of = as_of.loc[as_of["neutral_script"]].copy()

    team_all = as_of.loc[as_of["season"].eq(season) & as_of["posteam"].eq(team)].copy()
    league_all = as_of.loc[as_of["posteam"].ne(team)].copy()
    team_binary = team_all.loc[team_all["binary_call"].notna()].copy()
    league_binary = league_all.loc[league_all["binary_call"].notna()].copy()
    team_games = team_all[["game_id", "week"]].drop_duplicates().sort_values(["week", "game_id"])
    recent_game_ids = set(team_games.tail(4)["game_id"].astype(str))
    rows: list[dict[str, Any]] = []

    for specification in templates:
        template = specification["template"]
        feature_layer = specification["feature_layer"]
        columns = specification["columns"]
        available_contexts = team_binary[columns].dropna().drop_duplicates()
        for values in available_contexts.itertuples(index=False, name=None):
            if not isinstance(values, tuple):
                values = (values,)
            team_subset = team_binary.loc[make_context_mask(team_binary, columns, values)].copy()
            league_subset = league_binary.loc[make_context_mask(league_binary, columns, values)].copy()
            team_plays, league_plays = len(team_subset), len(league_subset)
            if team_plays < min_team_plays or league_plays < min_league_plays:
                continue

            team_runs = int(team_subset["binary_call"].eq("RUN").sum())
            league_run_rate = float(league_subset["binary_call"].eq("RUN").mean())
            team_run_rate_smoothed = float((team_runs + prior_strength * league_run_rate) / (team_plays + prior_strength))
            run_delta = team_run_rate_smoothed - league_run_rate
            over_index_call = "RUN" if run_delta >= 0 else "PASS"
            team_over_index_rate = team_run_rate_smoothed if over_index_call == "RUN" else 1 - team_run_rate_smoothed
            league_expected_rate = league_run_rate if over_index_call == "RUN" else 1 - league_run_rate
            rate_delta = abs(run_delta)
            if rate_delta < min_rate_delta:
                continue

            most_likely_call = "RUN" if team_run_rate_smoothed >= 0.50 else "PASS"
            most_likely_rate = team_run_rate_smoothed if most_likely_call == "RUN" else 1 - team_run_rate_smoothed
            if most_likely_call == over_index_call and most_likely_rate >= call_tell_threshold:
                alert_type = "CALL_TELL"
            elif most_likely_call != over_index_call:
                alert_type = "BALANCE_WARNING"
            else:
                alert_type = "RELATIVE_LEAN"

            game_profile = team_subset.groupby("game_id").agg(
                context_plays=("binary_call", "size"),
                run_rate=("binary_call", lambda values: values.eq("RUN").mean()),
            ).reset_index()
            stability_games = game_profile.loc[game_profile["context_plays"].ge(2)].copy()
            if stability_games.empty:
                stability_games = game_profile.copy()
            stability = float(stability_games["run_rate"].ge(league_run_rate).mean() if over_index_call == "RUN" else stability_games["run_rate"].le(league_run_rate).mean())
            games = int(game_profile["game_id"].nunique())

            recent_subset = team_subset.loc[team_subset["game_id"].astype(str).isin(recent_game_ids)].copy()
            if recent_subset.empty:
                recent_4_over_index_rate = np.nan
            else:
                recent_run_rate = float(recent_subset["binary_call"].eq("RUN").mean())
                recent_4_over_index_rate = recent_run_rate if over_index_call == "RUN" else 1 - recent_run_rate

            standard_error = math.sqrt(max(league_expected_rate * (1 - league_expected_rate), 1e-9) / (team_plays + prior_strength))
            signal_z = float(rate_delta / standard_error)
            context_all = team_all.loc[make_context_mask(team_all, columns, values)].copy()
            primary_calls = context_all.loc[context_all["binary_call"].eq(most_likely_call)].copy()
            alternative_calls = context_all.loc[context_all["binary_call"].notna() & context_all["binary_call"].ne(most_likely_call)].copy()
            counter_summary = summarize_best_constraint(context_all, most_likely_call, min_constraint_plays=3)
            specificity = len(columns)
            type_weight = {"CALL_TELL": 1.15, "BALANCE_WARNING": 1.00, "RELATIVE_LEAN": 0.85}[alert_type]
            alert_score = float(rate_delta * math.log1p(team_plays) * max(stability, 0.25) * (1 + 0.12 * (specificity - 1)) * type_weight)

            if team_plays >= 35 and games >= 6 and rate_delta >= 0.10 and stability >= 0.65 and signal_z >= 2.00:
                confidence = "HIGH"
            elif team_plays >= 20 and games >= 4 and rate_delta >= 0.07 and stability >= 0.55 and signal_z >= 1.25:
                confidence = "MONITOR"
            else:
                confidence = "FILM_REVIEW"

            rows.append({
                "feature_layer": feature_layer, "template": template,
                "situation": format_situation(columns, values), "alert_type": alert_type,
                "over_index_call": over_index_call, "most_likely_call": most_likely_call,
                "team_plays": team_plays, "league_plays": league_plays, "games": games,
                "team_over_index_rate": team_over_index_rate, "league_expected_rate": league_expected_rate,
                "rate_delta": rate_delta, "most_likely_rate": most_likely_rate,
                "recent_4_over_index_rate": recent_4_over_index_rate, "stability": stability,
                "signal_z": signal_z, "primary_call_epa": primary_calls["epa"].mean(),
                "alternative_call_epa": alternative_calls["epa"].mean(), **counter_summary,
                "confidence": confidence, "alert_score": alert_score,
                "context_signature": context_signature(team_subset),
            })

    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["confidence_order"] = output["confidence"].map({"HIGH": 0, "MONITOR": 1, "FILM_REVIEW": 2})
    output["alert_type_order"] = output["alert_type"].map({"CALL_TELL": 0, "BALANCE_WARNING": 1, "RELATIVE_LEAN": 2})
    return output.sort_values(["confidence_order", "alert_type_order", "alert_score", "team_plays"], ascending=[True, True, False, False]).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Report selection and staff-facing language
# -----------------------------------------------------------------------------

def classify_recency_status(row: pd.Series) -> str:
    recent_rate = row["recent_4_over_index_rate"]
    if pd.isna(recent_rate):
        return "NO_RECENT_SAMPLE"
    season_edge = row["team_over_index_rate"] - row["league_expected_rate"]
    recent_edge = recent_rate - row["league_expected_rate"]
    if recent_edge <= 0:
        return "REVERSING"
    if recent_edge < 0.50 * season_edge:
        return "WEAKENING"
    if recent_edge > season_edge + 0.05:
        return "STRENGTHENING"
    return "STABLE"


def classify_counter_tier(counter_plays: Any) -> str:
    if pd.isna(counter_plays):
        return "SUPPRESS"
    counter_plays = int(counter_plays)
    if counter_plays >= 10:
        return "PROMINENT"
    if counter_plays >= 5:
        return "LOW_SAMPLE_FILM_REVIEW"
    return "SUPPRESS"


def add_v2_reporting_fields(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return alerts.copy()
    output = alerts.copy()
    output["recency_status"] = output.apply(classify_recency_status, axis=1)
    output["counter_tier"] = output["counter_plays"].map(classify_counter_tier)
    return output


def parse_situation(situation: str) -> tuple[list[str], tuple[str, ...]]:
    label_to_column = {
        "QB": "qb_location_group", "PERS": "personnel_code", "FORM": "offense_formation_group",
        "MOTION": "motion_group", "DOWN": "down_group", "DIST": "distance_bucket",
        "PREV": "prev_drive_play_family",
    }
    columns, values = [], []
    for component in situation.split(" | "):
        label, value = component.split("=", maxsplit=1)
        columns.append(label_to_column[label])
        values.append(value)
    return columns, tuple(values)


def get_matching_subset(frame: pd.DataFrame, situation: str) -> pd.DataFrame:
    columns, values = parse_situation(situation)
    return frame.loc[make_context_mask(frame, columns, values)].copy()


def get_play_key_set(frame: pd.DataFrame, situation: str) -> frozenset[str]:
    matched = get_matching_subset(frame, situation)
    if matched.empty:
        return frozenset()
    keys = matched["game_id"].astype("string") + "::" + matched["play_id"].astype("string")
    return frozenset(keys.tolist())


def overlap_coefficient(first: frozenset[str], second: frozenset[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first.intersection(second)) / min(len(first), len(second))


def select_nonredundant_scout_card(
    alerts: pd.DataFrame,
    data: pd.DataFrame,
    team: str,
    season: int,
    through_week: int,
    alert_type: str,
    feature_layer: str | None = None,
    max_alerts: int = 6,
    max_per_template: int = 2,
    overlap_threshold: float = 0.85,
    allowed_confidence: tuple[str, ...] = ("HIGH", "MONITOR"),
) -> pd.DataFrame:
    if alerts.empty:
        return alerts.copy()
    candidates = alerts.loc[alerts["alert_type"].eq(alert_type) & alerts["confidence"].isin(allowed_confidence)].copy()
    if feature_layer is not None:
        candidates = candidates.loc[candidates["feature_layer"].eq(feature_layer)].copy()
    if candidates.empty:
        return candidates
    candidates["specificity"] = candidates["situation"].str.count(r"\|").add(1)
    candidates = candidates.sort_values(["confidence_order", "specificity", "alert_score", "team_plays"], ascending=[True, False, False, False]).reset_index(drop=True)
    history = data.loc[data["season"].eq(season) & data["week"].le(through_week) & data["posteam"].eq(team) & data["binary_call"].notna()].copy()
    kept_rows: list[pd.Series] = []
    kept_play_sets: list[tuple[str, frozenset[str]]] = []
    template_counts: dict[str, int] = {}
    for _, row in candidates.iterrows():
        template = row["template"]
        if template_counts.get(template, 0) >= max_per_template:
            continue
        play_set = get_play_key_set(history, row["situation"])
        redundant = any(
            kept_call == row["over_index_call"] and overlap_coefficient(play_set, kept_set) >= overlap_threshold
            for kept_call, kept_set in kept_play_sets
        )
        if redundant:
            continue
        kept_rows.append(row)
        kept_play_sets.append((row["over_index_call"], play_set))
        template_counts[template] = template_counts.get(template, 0) + 1
        if len(kept_rows) >= max_alerts:
            break
    return pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else candidates.head(0)


def restrict_alert_pool(alerts: pd.DataFrame, templates: set[str], allowed_confidence: tuple[str, ...], allowed_recency: tuple[str, ...]) -> pd.DataFrame:
    if alerts.empty:
        return alerts.copy()
    return alerts.loc[
        alerts["template"].isin(templates)
        & alerts["confidence"].isin(allowed_confidence)
        & alerts["recency_status"].isin(allowed_recency)
    ].copy()


def select_report_section(
    alerts: pd.DataFrame, data: pd.DataFrame, team: str, season: int, through_week: int,
    alert_type: str, feature_layer: str | None, max_alerts: int,
    allowed_confidence: tuple[str, ...],
) -> pd.DataFrame:
    if alerts.empty:
        return alerts.copy()
    return select_nonredundant_scout_card(
        alerts=alerts, data=data, team=team, season=season, through_week=through_week,
        alert_type=alert_type, feature_layer=feature_layer, max_alerts=max_alerts,
        max_per_template=2, overlap_threshold=0.85, allowed_confidence=allowed_confidence,
    )


def select_identity_shifts(alerts: pd.DataFrame, max_alerts: int = 4) -> pd.DataFrame:
    if alerts.empty:
        return alerts.copy()
    shifts = alerts.loc[
        alerts["confidence"].isin(["HIGH", "MONITOR"])
        & alerts["recency_status"].isin(["WEAKENING", "REVERSING"])
    ].copy()
    if shifts.empty:
        return shifts
    shifts["shift_priority"] = shifts["recency_status"].map({"REVERSING": 0, "WEAKENING": 1})
    return shifts.sort_values(["shift_priority", "confidence_order", "alert_score"], ascending=[True, True, False]).drop_duplicates(subset=["situation"]).head(max_alerts).reset_index(drop=True)


def build_v2_opponent_report(data: pd.DataFrame, team: str, season: int, through_week: int) -> dict[str, pd.DataFrame]:
    alerts = add_v2_reporting_fields(build_tendency_alerts(data, team, season, through_week))
    live_active_pool = restrict_alert_pool(alerts, LIVE_ACTIVE_TEMPLATES, ("HIGH",), ("STABLE", "STRENGTHENING"))
    live_monitor_pool = restrict_alert_pool(alerts, LIVE_ACTIVE_TEMPLATES, ("MONITOR",), ("STABLE", "STRENGTHENING"))
    enriched_pool = restrict_alert_pool(alerts, ENRICHED_ACTIVE_TEMPLATES, ("HIGH",), ("STABLE", "STRENGTHENING"))
    balance_pool = restrict_alert_pool(alerts, BALANCE_ACTIVE_TEMPLATES, ("HIGH", "MONITOR"), ("STABLE", "STRENGTHENING"))
    return {
        "active_live_tells": select_report_section(live_active_pool, data, team, season, through_week, "CALL_TELL", "LIVE_WEEKLY", 5, ("HIGH",)),
        "emerging_live_tells": select_report_section(live_monitor_pool, data, team, season, through_week, "CALL_TELL", "LIVE_WEEKLY", 3, ("MONITOR",)),
        "enriched_film_priorities": select_report_section(enriched_pool, data, team, season, through_week, "CALL_TELL", "HISTORICAL_ENRICHMENT", 5, ("HIGH",)),
        "balance_warnings": select_report_section(balance_pool, data, team, season, through_week, "BALANCE_WARNING", None, 3, ("HIGH", "MONITOR")),
        "identity_shifts": select_identity_shifts(alerts, max_alerts=4),
        "all_alerts": alerts,
    }

# -----------------------------------------------------------------------------
# Human-readable notes
# -----------------------------------------------------------------------------

def parse_situation_parts(situation: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for component in situation.split(" | "):
        label, value = component.split("=", maxsplit=1)
        output[label] = value
    return output


def prettify_situation(situation: str) -> str:
    parts = parse_situation_parts(situation)
    readable: list[str] = []
    if parts.get("PREV") == "START":
        readable.append("first snap of drive")
    if "QB" in parts:
        readable.append({"SHOTGUN": "shotgun", "UNDER_CENTER": "under center", "PISTOL": "pistol"}.get(parts["QB"], parts["QB"].lower()))
    if "PERS" in parts:
        readable.append(f"{parts['PERS']} personnel")
    if "FORM" in parts:
        readable.append(parts["FORM"].replace("_", " ").lower() + " formation")
    if "MOTION" in parts:
        readable.append({"MOTION": "with motion", "NO_MOTION": "without motion"}.get(parts["MOTION"], parts["MOTION"].lower()))
    if "DOWN" in parts:
        readable.append({"1ST": "first down", "2ND": "second down", "3RD_4TH": "third or fourth down"}.get(parts["DOWN"], parts["DOWN"].lower()))
    if "DIST" in parts:
        readable.append({"1": "1 yard to go", "2-3": "2–3 yards to go", "4-6": "4–6 yards to go", "7-10": "7–10 yards to go", "11+": "11+ yards to go"}.get(parts["DIST"], f"{parts['DIST']} yards to go"))
    return "; ".join(readable)


def prettify_call(call: Any) -> str:
    if pd.isna(call):
        return ""
    return {
        "RPO": "RPO", "PLAY_ACTION_PASS": "play-action pass", "STANDARD_DROPBACK": "standard dropback",
        "DESIGNED_RUN": "designed run", "SCREEN": "screen", "PASS": "pass", "RUN": "run",
    }.get(str(call), str(call).replace("_", " ").lower())


def format_pct(value: Any) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.1%}"


def build_counter_note(row: pd.Series) -> str:
    counter_plays = row.get("counter_plays", 0)
    counter_epa = row.get("counter_epa", np.nan)
    counter_call = row.get("best_counter_call", pd.NA)
    if pd.isna(counter_call) or pd.isna(counter_plays) or int(counter_plays) < 5 or pd.isna(counter_epa):
        return ""
    counter_plays = int(counter_plays)
    call_label = prettify_call(counter_call)
    if counter_epa >= 0.05:
        prefix = "Constraint candidate" if counter_plays >= 10 else "Low-sample constraint candidate"
        return f"{prefix}: {call_label} ({counter_plays} plays, {counter_epa:+.3f} EPA/play)."
    if counter_epa <= -0.05:
        return f"{call_label.capitalize()} has not punished the tendency to date ({counter_plays} plays, {counter_epa:+.3f} EPA/play)."
    return f"{call_label.capitalize()} has produced approximately neutral results to date ({counter_plays} plays, {counter_epa:+.3f} EPA/play)."


def create_staff_note(row: pd.Series) -> str:
    situation = prettify_situation(row["situation"])
    over_index_call = prettify_call(row["over_index_call"])
    most_likely_call = prettify_call(row["most_likely_call"])
    team_rate = format_pct(row["team_over_index_rate"])
    league_rate = format_pct(row["league_expected_rate"])
    recent_rate = format_pct(row["recent_4_over_index_rate"])
    stability = format_pct(row["stability"])
    recency = str(row["recency_status"]).replace("_", " ").lower()
    if row["alert_type"] == "CALL_TELL":
        note = f"{situation}: {over_index_call} on {team_rate} of qualifying snaps versus a {league_rate} league expectation. The tendency held across {stability} of games and was {recent_rate} over the last four games. Status: {recency}."
    else:
        note = f"{situation}: the offense still leans {most_likely_call}, but it calls {over_index_call} at a {team_rate} rate versus a {league_rate} league expectation. Do not overplay the conventional tendency. The over-index direction held across {stability} of games and was {recent_rate} over the last four games. Status: {recency}."
    counter_note = build_counter_note(row)
    return note + (" " + counter_note if counter_note else "")


def create_staff_shift_note(row: pd.Series) -> str:
    situation = prettify_situation(row["situation"])
    call = prettify_call(row["over_index_call"])
    recency = str(row["recency_status"]).replace("_", " ").lower()
    return f"{situation}: the season-long {call} tendency was {format_pct(row['team_over_index_rate'])} versus a {format_pct(row['league_expected_rate'])} league expectation, but moved to {format_pct(row['recent_4_over_index_rate'])} over the last four games. Status: {recency}. Treat this as an identity shift rather than an active key."


def balance_warning_similarity(first: pd.Series, second: pd.Series) -> bool:
    first_parts, second_parts = parse_situation_parts(first["situation"]), parse_situation_parts(second["situation"])
    shared_keys = set(first_parts).intersection(second_parts)
    shared_values_match = all(first_parts[key] == second_parts[key] for key in shared_keys)
    same_direction = first["over_index_call"] == second["over_index_call"]
    similar_rate = abs(float(first["team_over_index_rate"]) - float(second["team_over_index_rate"])) < 0.05
    return len(shared_keys) >= 2 and shared_values_match and same_direction and similar_rate


def collapse_related_balance_warnings(frame: pd.DataFrame, max_items: int = 3) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["specificity"] = output["situation"].str.count(r"\|").add(1)
    output = output.sort_values(["confidence_order", "specificity", "alert_score"], ascending=[True, True, False]).reset_index(drop=True)
    kept: list[pd.Series] = []
    for _, row in output.iterrows():
        if any(balance_warning_similarity(row, existing) for existing in kept):
            continue
        kept.append(row)
        if len(kept) >= max_items:
            break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else output.head(0)

# -----------------------------------------------------------------------------
# Film queue
# -----------------------------------------------------------------------------

def dedupe_play_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates(subset=["game_id", "play_id"], keep="first").copy() if not frame.empty else frame.copy()


def add_play_key(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["play_key"] = output["game_id"].astype("string") + "::" + output["play_id"].astype("string")
    return output


def select_recent_example(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.sort_values(["week", "game_id", "play_id"], ascending=[False, False, False]).head(1).copy()
    output["selection_reason"] = "RECENT_EXAMPLE"
    return output


def select_representative_example(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    median_epa = pd.to_numeric(output["epa"], errors="coerce").median()
    output["_distance_from_median_epa"] = pd.to_numeric(output["epa"], errors="coerce").sub(median_epa).abs()
    output = output.sort_values(["_distance_from_median_epa", "week", "play_id"], ascending=[True, False, False]).head(1).drop(columns=["_distance_from_median_epa"], errors="ignore").copy()
    output["selection_reason"] = "REPRESENTATIVE_EXAMPLE"
    return output


def select_outcome_illustrative_example(frame: pd.DataFrame, aggregate_epa: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["epa"] = pd.to_numeric(output["epa"], errors="coerce")
    if aggregate_epa >= 0.05:
        selected = output.sort_values(["epa", "week", "play_id"], ascending=[False, False, False]).head(1).copy()
        selected["selection_reason"] = "PRODUCTIVE_COUNTER_EXAMPLE"
        return selected
    if aggregate_epa <= -0.05:
        selected = output.sort_values(["epa", "week", "play_id"], ascending=[True, False, False]).head(1).copy()
        selected["selection_reason"] = "INEFFECTIVE_COUNTER_EXAMPLE"
        return selected
    selected = select_representative_example(output)
    selected["selection_reason"] = "NEUTRAL_COUNTER_EXAMPLE"
    return selected


def exclude_selected_plays(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or selected.empty:
        return frame.copy()
    output = add_play_key(frame)
    selected_keys = set(add_play_key(selected)["play_key"].tolist())
    return output.loc[~output["play_key"].isin(selected_keys)].drop(columns=["play_key"], errors="ignore").copy()


def package_film_rows(frame: pd.DataFrame, section_name: str, alert: pd.Series, example_type: str, call_label: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, play in frame.iterrows():
        rows.append({
            "section": section_name, "alert_situation": prettify_situation(alert["situation"]),
            "example_type": example_type, "selection_reason": play.get("selection_reason", ""),
            "call_family": call_label, "week": play.get("week", np.nan), "game_id": play.get("game_id", ""),
            "play_id": play.get("play_id", ""), "play_family": play.get("play_family", ""),
            "epa": play.get("epa", np.nan), "description": play.get("desc", ""),
        })
    return rows


def create_example_rows_v2(alert: pd.Series, history: pd.DataFrame, section_name: str) -> list[dict[str, Any]]:
    matched = dedupe_play_rows(get_matching_subset(history, alert["situation"]))
    if matched.empty:
        return []
    rows: list[dict[str, Any]] = []
    primary = matched.loc[matched["binary_call"].eq(alert["most_likely_call"])].copy()
    primary_recent = select_recent_example(primary)
    primary_representative = select_representative_example(exclude_selected_plays(primary, primary_recent))
    primary_selected = pd.concat([primary_recent, primary_representative], ignore_index=True) if not primary_recent.empty or not primary_representative.empty else pd.DataFrame()
    rows.extend(package_film_rows(primary_selected, section_name, alert, "PRIMARY_TENDENCY", prettify_call(alert["most_likely_call"])))

    counter_call, counter_plays, counter_epa = alert.get("best_counter_call", pd.NA), alert.get("counter_plays", 0), alert.get("counter_epa", np.nan)
    if pd.isna(counter_call) or pd.isna(counter_plays) or int(counter_plays) < 5 or pd.isna(counter_epa):
        return rows
    counter_frame = add_constraint_family(matched, alert["most_likely_call"])
    counter_frame = counter_frame.loc[counter_frame["constraint_call"].eq(counter_call)].copy()
    counter_recent = select_recent_example(counter_frame)
    counter_illustrative = select_outcome_illustrative_example(exclude_selected_plays(counter_frame, counter_recent), float(counter_epa))
    counter_selected = pd.concat([counter_recent, counter_illustrative], ignore_index=True) if not counter_recent.empty or not counter_illustrative.empty else pd.DataFrame()
    rows.extend(package_film_rows(counter_selected, section_name, alert, "CONSTRAINT_EXAMPLE", prettify_call(counter_call)))
    return rows


def concatenate_unique_labels(values: pd.Series) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return " || ".join(ordered)


def collapse_film_review_queue(queue_long: pd.DataFrame) -> pd.DataFrame:
    if queue_long.empty:
        return queue_long.copy()
    collapsed = queue_long.groupby(["week", "game_id", "play_id", "play_family", "epa", "description"], dropna=False, as_index=False).agg(
        sections=("section", concatenate_unique_labels), alert_situations=("alert_situation", concatenate_unique_labels),
        example_types=("example_type", concatenate_unique_labels), selection_reasons=("selection_reason", concatenate_unique_labels),
        call_families=("call_family", concatenate_unique_labels),
    )
    return collapsed.sort_values(["week", "game_id", "play_id"], ascending=[False, False, False]).reset_index(drop=True)


def build_film_review_queue_v2(report: dict[str, pd.DataFrame], data: pd.DataFrame, team: str, season: int, through_week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = data.loc[data["season"].eq(season) & data["posteam"].eq(team) & data["week"].le(through_week)].copy()
    rows: list[dict[str, Any]] = []
    for section_name in ["active_live_tells", "emerging_live_tells", "enriched_film_priorities", "balance_warnings"]:
        section = report.get(section_name, pd.DataFrame())
        if section.empty:
            continue
        for _, alert in section.iterrows():
            rows.extend(create_example_rows_v2(alert, history, section_name))
    queue_long = pd.DataFrame(rows)
    return queue_long, collapse_film_review_queue(queue_long)

# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------

def markdown_escape(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No qualifying rows._"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(markdown_escape(row[column]) for column in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, divider, *rows])


def build_markdown_report(team: str, season: int, through_week: int, identity_card: pd.DataFrame, report: dict[str, pd.DataFrame], film_queue: pd.DataFrame) -> str:
    lines = [f"# {team} Offensive Tendency Scout", "", f"**Season:** {season}  ", f"**Through week:** {through_week}  ", "", "## Offensive Identity Snapshot", "", dataframe_to_markdown(format_identity_table(identity_card)), ""]
    for section_name in ["active_live_tells", "emerging_live_tells", "enriched_film_priorities", "balance_warnings", "identity_shifts"]:
        lines.extend([f"## {SECTION_TITLES[section_name]}", ""])
        section = report.get(section_name, pd.DataFrame())
        if section.empty:
            lines.extend(["_No qualifying alerts._", ""])
            continue
        for number, (_, row) in enumerate(section.iterrows(), start=1):
            note = create_staff_shift_note(row) if section_name == "identity_shifts" else create_staff_note(row)
            lines.extend([f"{number}. {note}", ""])
    lines.extend(["## Film-Review Queue", "", f"{len(film_queue)} representative play examples were exported with this report." if not film_queue.empty else "No representative play examples were exported.", ""])
    return "\n".join(lines)

"""NFL sport adapter — namespaced game keys; does not use MLB game_pk."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlb_metrics.sports.base import (
    SPORT_SCHEDULE_COLUMNS,
    SportCapabilities,
    namespaced_game_key,
)


class NflSportAdapter:
    sport_id = "nfl"

    def capabilities(self) -> SportCapabilities:
        return SportCapabilities(
            sport_id="nfl",
            schedule=True,
            official_results=True,
            participant_features=True,
            polymarket_league_slug="nfl",
            game_key_namespace="nfl",
            notes="Native identity is nflreadpy game_id; sport_game_key is nfl:{game_id}.",
        )

    def load_schedule_for_mapping(
        self,
        *,
        lookahead_days: int = 7,
        as_of_local=None,
        seasons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Load NFL schedules for mapping (public nflreadpy via nfl_data)."""
        from mlb_metrics import nfl_data, schedule

        as_of = as_of_local or schedule.today_local()
        year = int(pd.Timestamp(as_of).year) if not hasattr(as_of, "year") else int(as_of.year)
        seasons = seasons or [year - 1, year]
        try:
            raw = nfl_data.fetch_schedules(seasons)
        except Exception:
            return pd.DataFrame(columns=SPORT_SCHEDULE_COLUMNS)

        if raw is None or raw.empty:
            return pd.DataFrame(columns=SPORT_SCHEDULE_COLUMNS)

        frame = raw.copy()
        # nflreadpy schedules commonly use home_team / away_team / gameday / gametime / game_id
        home_col = "home_team" if "home_team" in frame.columns else None
        away_col = "away_team" if "away_team" in frame.columns else None
        id_col = "game_id" if "game_id" in frame.columns else None
        if not home_col or not away_col or not id_col:
            return pd.DataFrame(columns=SPORT_SCHEDULE_COLUMNS)

        if "gameday" in frame.columns:
            date_series = pd.to_datetime(frame["gameday"], errors="coerce")
        elif "game_date" in frame.columns:
            date_series = pd.to_datetime(frame["game_date"], errors="coerce")
        else:
            date_series = pd.Series(pd.NaT, index=frame.index)

        # Combine local US/Eastern wall-clock kickoff into UTC when gametime is present.
        game_dt = date_series.copy()
        if "gametime" in frame.columns:
            combined = []
            for day, t in zip(date_series, frame["gametime"]):
                if pd.isna(day):
                    combined.append(pd.NaT)
                    continue
                try:
                    stamp = pd.Timestamp(f"{pd.Timestamp(day).strftime('%Y-%m-%d')} {t}")
                    if stamp.tzinfo is None:
                        stamp = stamp.tz_localize(
                            "America/New_York", ambiguous="NaT", nonexistent="NaT"
                        )
                    combined.append(stamp.tz_convert("UTC"))
                except Exception:
                    combined.append(pd.Timestamp(day, tz="UTC"))
            game_dt = pd.Series(combined, index=frame.index)

        out = pd.DataFrame(
            {
                "sport_id": "nfl",
                "sport_game_key": frame[id_col].map(lambda x: namespaced_game_key("nfl", x)),
                "home_team": frame[home_col].astype(str).str.upper(),
                "away_team": frame[away_col].astype(str).str.upper(),
                "game_datetime": game_dt,
                "date": date_series.dt.strftime("%Y-%m-%d"),
                "status": frame["game_status"] if "game_status" in frame.columns else pd.NA,
                "home_score": frame["home_score"] if "home_score" in frame.columns else pd.NA,
                "away_score": frame["away_score"] if "away_score" in frame.columns else pd.NA,
                "native_game_id": frame[id_col].astype(str),
            }
        )
        # Optional lookahead filter from as_of
        if as_of is not None and out["date"].notna().any():
            start = pd.Timestamp(as_of).normalize()
            end = start + pd.Timedelta(days=int(lookahead_days))
            day = pd.to_datetime(out["date"], errors="coerce")
            out = out[(day >= start) & (day <= end)].reset_index(drop=True)
        return out

    def official_winners(self, results: pd.DataFrame) -> dict[str, str]:
        winners: dict[str, str] = {}
        if results is None or results.empty:
            return winners
        for _, row in results.iterrows():
            key = row.get("sport_game_key")
            if key is None or (isinstance(key, float) and pd.isna(key)):
                native = row.get("native_game_id") or row.get("game_id")
                if native is None or (isinstance(native, float) and pd.isna(native)):
                    continue
                key = namespaced_game_key("nfl", native)
            try:
                hs = float(row["home_score"])
                aws = float(row["away_score"])
            except (TypeError, ValueError, KeyError):
                continue
            if pd.isna(hs) or pd.isna(aws):
                continue
            if hs > aws:
                winners[str(key)] = str(row["home_team"]).upper()
            elif aws > hs:
                winners[str(key)] = str(row["away_team"]).upper()
        return winners

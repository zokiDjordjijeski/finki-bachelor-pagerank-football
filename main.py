"""PageRank ranking of European football clubs from heterogeneous match CSVs."""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import seaborn as sns


def _configure_stdio() -> None:
    """Avoid UnicodeEncodeError on Windows consoles (cp1252) for Macedonian text."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_configure_stdio()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("pagerank")

# ---------------------------------------------------------------------------
# Configuration (Избор на Сценарио: 1, 2 или 3)
# ---------------------------------------------------------------------------
SCENARIO = int(input("Изберете сценарио (1 - Хомогени, 2 - Хиерархиски, 3 - Линеарно опаѓање 5м): ").strip() or "3")
MIN_MATCHES = 10
PAGERANK_ALPHA = 0.85
DRAW_WEIGHT = 0.55
LOSS_BACK = 0.10
GOAL_BONUS = 0.05
DATA_DIR = "data"
CUPS_DIR = "cups"
DOWNLOAD_TIMEOUT_SEC = 8
MIN_CSV_BYTES = 80

COMPETITION_COEF: dict[str, float] = {
    "Premier League": 1.0,
    "La Liga": 1.0,
    "Serie A": 1.0,
    "Bundesliga": 1.0,
    "Ligue 1": 1.0,
    "Eredivisie": 0.8,
    "Championship": 0.6,
    "Scottish Premiership": 0.4,
    "National Cup": 0.9,
    "Champions League": 1.5,
    "Europa League": 1.3,
    "Conference League": 1.2,
    "FA Cup": 0.9,
    "Copa del Rey": 0.9,
    "DFB-Pokal": 0.9,
    "Coppa Italia": 0.9,
    "EFL Cup": 0.9,
    "UEFA Champions League": 1.5,
    "UEFA Europa League": 1.3,
    "UEFA Conference League": 1.2,
}

LEAGUE_CODES: dict[str, str] = {
    "e0": "Premier League",
    "e1": "Championship",
    "sp1": "La Liga",
    "i1": "Serie A",
    "d1": "Bundesliga",
    "f1": "Ligue 1",
    "n1": "Eredivisie",
    "sc0": "Scottish Premiership",
}

EUROPEAN_COMPS = frozenset(
    {"Champions League", "Europa League", "Conference League", "UEFA Champions League", "UEFA Europa League", "UEFA Conference League"}
)

# Canonical club names
TEAM_NAME_MAPPING: dict[str, str] = {
    "Paris SG": "PSG",
    "Paris Saint Germain": "PSG",
    "Paris Saint-Germain": "PSG",
    "Bayern München": "Bayern Munich",
    "Bayern Munchen": "Bayern Munich",
    "Manchester Utd": "Manchester United",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Ath Madrid": "Atletico Madrid",
    "Atl. Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Milan": "AC Milan",
    "Roma": "AS Roma",
    "Leverkusen": "Bayer Leverkusen",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Vallecano": "Rayo Vallecano",
    "Sociedad": "Real Sociedad",
    "PSV Eindhoven": "PSV",
}

COLUMN_MAPPING: dict[str, str] = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "homeTeam": "HomeTeam",
    "awayTeam": "AwayTeam",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "Team1": "HomeTeam",
    "Team2": "AwayTeam",
    "Home_Team": "HomeTeam",
    "Away_Team": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "homeScore": "FTHG",
    "awayScore": "FTAG",
    "ft_home": "FTHG",
    "ft_away": "FTAG",
    "HomeGoals": "FTHG",
    "AwayGoals": "FTAG",
    "matchDate": "Date",
    "date": "Date",
    "kickoff_utc": "Date",
    "league_name": "Comp",
}

REQUIRED_COLUMNS = frozenset({"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"})

FOOTBALL_DATA_URLS: dict[str, str] = {
    "E0": "https://football-data.co.uk/mmz4281/2627/E0.csv",
    "E1": "https://football-data.co.uk/mmz4281/2627/E1.csv",
    "SP1": "https://football-data.co.uk/mmz4281/2627/SP1.csv",
    "I1": "https://football-data.co.uk/mmz4281/2627/I1.csv",
    "D1": "https://football-data.co.uk/mmz4281/2627/D1.csv",
    "F1": "https://football-data.co.uk/mmz4281/2627/F1.csv",
    "N1": "https://football-data.co.uk/mmz4281/2627/N1.csv",
    "SC0": "https://football-data.co.uk/mmz4281/2627/SC0.csv",
}


# ---------------------------------------------------------------------------
# Downloads (со паметно кеширање)
# ---------------------------------------------------------------------------
def _looks_like_csv(content: bytes) -> bool:
    if len(content) < MIN_CSV_BYTES:
        return False
    head = content[:400].decode("utf-8", errors="replace").lstrip()
    if head.lower().startswith("<!doctype") or head.lower().startswith("<html"):
        return False
    first_line = head.splitlines()[0] if head.splitlines() else ""
    return "," in first_line


def download_latest_data() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    all_exist = True
    for code in FOOTBALL_DATA_URLS:
        dest = os.path.join(DATA_DIR, f"{code}_26-27.csv")
        if not os.path.exists(dest):
            all_exist = False
            break
        file_age_hours = (time.time() - os.path.getmtime(dest)) / 3600
        if file_age_hours > 24:
            all_exist = False
            break

    if all_exist:
        print("Проверка на локалните фајлови: Ажурирани се (помлади од 24 часа). Прескокнувам симнување.\n")
        return

    print("Проверка на серверите и ажурирање на дневните фајлови...")
    for code, url in FOOTBALL_DATA_URLS.items():
        dest = os.path.join(DATA_DIR, f"{code}_26-27.csv")
        try:
            response = requests.get(
                url,
                timeout=DOWNLOAD_TIMEOUT_SEC,
                headers={"User-Agent": "FINKI-PageRank-Thesis/1.0"},
            )
        except requests.RequestException as exc:
            logger.warning("Download failed for %s (%s): %s", code, url, exc)
            continue

        if response.status_code != 200 or not _looks_like_csv(response.content):
            continue

        with open(dest, "wb") as handle:
            handle.write(response.content)
    print("Проверката заврши.\n")


# ---------------------------------------------------------------------------
# Competition / time weights
# ---------------------------------------------------------------------------
def detect_competition(path: str) -> str | None:
    name = os.path.basename(path).lower()
    if "champions" in name:
        return "Champions League"
    if "europa" in name:
        return "Europa League"
    if "conference" in name:
        return "Conference League"
    if "cup" in name or "copa" in name or name.startswith("fa"):
        return "National Cup"
    for code, comp in LEAGUE_CODES.items():
        if name.startswith(code):
            return comp
    return None


def stage_multiplier(comp: str, match_date: pd.Timestamp) -> float:
    if SCENARIO == 1 or comp not in EUROPEAN_COMPS:
        return 1.0
    if pd.isna(match_date):
        return 1.0

    month = int(match_date.month)
    day = int(match_date.day)
    if month == 2:
        return 1.1
    if month == 3:
        return 1.2
    if month == 4:
        return 1.4
    if month == 5:
        return 1.6 if day < 20 else 1.8
    if month == 6:
        return 1.8
    return 1.0


def time_weight_for(days: int) -> float:
    """Сценарио 3: Линеарно опаѓање за последните 5 месеци (150 дена)."""
    if SCENARIO == 3:
        max_days = 150
        if days > max_days:
            return 0.0
        return max(0.0, 1.0 - (days / max_days))

    months = max(days, 0) / 30.0
    for limit, weight in (
        (1, 1.0),
        (2, 0.9),
        (3, 0.8),
        (4, 0.7),
        (5, 0.6),
        (6, 0.5),
        (7, 0.4),
        (10, 0.3),
    ):
        if months <= limit:
            return weight
    return 0.2


def current_season_start(max_date: pd.Timestamp) -> pd.Timestamp:
    year = int(max_date.year) if int(max_date.month) >= 8 else int(max_date.year) - 1
    return pd.Timestamp(year=year, month=8, day=1)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def _read_csv_safely(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise RuntimeError(f"{os.path.basename(path)}: {exc}") from exc
    raise RuntimeError(f"{os.path.basename(path)}: encoding error")


def _normalize_frame(df: pd.DataFrame, competition: str) -> tuple[pd.DataFrame, int]:
    df = df.rename(columns=COLUMN_MAPPING)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        raise ValueError(f"missing columns {missing}")

    before = len(df)
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
    df["Comp"] = competition
    df["LeagueWeight"] = COMPETITION_COEF.get(competition, 1.0)
    df["SourceKind"] = "european" if competition in EUROPEAN_COMPS else "domestic"
    return df, dropped


def load_data(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))

    cup_file = os.path.join(CUPS_DIR, "matches.csv")
    if os.path.exists(cup_file):
        files.append(cup_file)

    print(f"Пронајдени се вкупно {len(files)} фајлови. Се вчитуваат само тимовите и лигите од фокусот...")

    frames: list[pd.DataFrame] = []

    ALLOWED_COMPETITIONS = frozenset({
        "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
        "Eredivisie", "Championship", "Scottish Premiership",
        "FA Cup", "Copa del Rey", "DFB-Pokal", "Coppa Italia", "EFL Cup",
        "Champions League", "Europa League", "Conference League",
        "UEFA Champions League", "UEFA Europa League", "UEFA Conference League"
    })

    for path in files:
        basename = os.path.basename(path)
        competition = detect_competition(path)
        try:
            raw = _read_csv_safely(path)

            if basename == "matches.csv":
                ONLY_CUPS = frozenset({"FA Cup", "Copa del Rey", "DFB-Pokal", "Coppa Italia", "EFL Cup"})
                raw = raw.rename(columns=COLUMN_MAPPING)
                if "Comp" in raw.columns:
                    for comp_name, group_df in raw.groupby("Comp"):
                        comp_str = str(comp_name).strip()
                        if comp_str in ONLY_CUPS:
                            df_n, _ = _normalize_frame(group_df, comp_str)
                            if not df_n.empty:
                                frames.append(df_n)
                                print(f"{'cups/matches.csv (' + comp_str + ')':<45}{comp_str:<24}{'ok':<14}{COMPETITION_COEF.get(comp_str, 0.9):>7.3f}{len(df_n):>8}")
                continue

            if competition is None or competition not in ALLOWED_COMPETITIONS:
                continue

            df, dropped = _normalize_frame(raw, competition)
            if not df.empty:
                coef = float(df["LeagueWeight"].iloc[0])
                print(f"{basename:<45}{competition:<24}{'ok':<14}{coef:>7.3f}{len(df):>8}")
                frames.append(df)
        except Exception as exc:
            logger.error("Failed to load %s: %s", basename, exc)

    if not frames:
        print("Нема податоци за обработка.")
        sys.exit(1)

    master_df = pd.concat(frames, ignore_index=True)

    master_df["Date"] = pd.to_datetime(master_df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    master_df = master_df.dropna(subset=["Date"])
    return master_df


def report_unmatched_team_names(df: pd.DataFrame, similarity_floor: float = 0.82) -> None:
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
    print("\n--- Валидација на имиња на клубови ---")
    print(f"Уникатни клубови по мапирање: {len(teams)}")


# ---------------------------------------------------------------------------
# Graph + PageRank
# ---------------------------------------------------------------------------
def add_edge(graph: nx.DiGraph, source: str, target: str, weight: float) -> None:
    if graph.has_edge(source, target):
        graph[source][target]["weight"] += weight
    else:
        graph.add_edge(source, target, weight=weight)


def match_edge_weights(goal_diff: int, base: float) -> tuple[float, float]:
    win_weight = (1.0 + GOAL_BONUS * min(goal_diff, 3)) * base
    return win_weight, LOSS_BACK * win_weight


def build_graph(df: pd.DataFrame, max_date: pd.Timestamp) -> tuple[nx.DiGraph, dict[str, int]]:
    graph = nx.DiGraph()
    counts: dict[str, int] = {}
    for row in df.itertuples(index=False):
        home, away = row.HomeTeam, row.AwayTeam
        counts[home] = counts.get(home, 0) + 1
        counts[away] = counts.get(away, 0) + 1

        days_diff = (max_date - row.Date).days if pd.notna(row.Date) else 0
        recency = time_weight_for(int(days_diff))
        if SCENARIO == 3 and recency == 0.0:
            continue

        stage = stage_multiplier(row.Comp, row.Date)
        base = recency * float(row.LeagueWeight) * stage
        if base <= 0:
            continue

        home_goals, away_goals = int(row.FTHG), int(row.FTAG)
        if home_goals == away_goals:
            draw = DRAW_WEIGHT * base
            add_edge(graph, home, away, draw)
            add_edge(graph, away, home, draw)
            continue

        if home_goals > away_goals:
            winner, loser, goal_diff = home, away, home_goals - away_goals
        else:
            winner, loser, goal_diff = away, home, away_goals - home_goals

        to_winner, to_loser = match_edge_weights(goal_diff, base)
        add_edge(graph, loser, winner, to_winner)
        add_edge(graph, winner, loser, to_loser)

    return graph, counts


def compute_ranking(
    graph: nx.DiGraph,
    counts: dict[str, int],
    min_matches: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = nx.pagerank(graph, weight="weight", alpha=PAGERANK_ALPHA)
    ranked = [
        (team, score)
        for team, score in scores.items()
        if counts.get(team, 0) >= min_matches
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# CLI, export, plots
# ---------------------------------------------------------------------------
def prompt_horizon_choice() -> str:
    print("\n" + "=" * 50)
    print(" ИЗБЕРЕТЕ ВРЕМЕНСКИ ПЕРИОД ЗА PAGERANK МОДЕЛОТ:")
    print("=" * 50)
    print(" 1. Последни 3 месеци")
    print(" 2. Последни 6 месеци")
    print(" 3. Последни 12 месеци (Една година)")
    print(" 4. Само тековната сезона (од август)")
    print(" 5. Сите достапни податоци (Без филтер)")
    print("=" * 50)
    return input("Внесете број на опција (1-5): ").strip() or "5"


def apply_time_filter(master: pd.DataFrame, choice: str, max_date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    if choice == "1":
        return master[master["Date"] >= max_date - pd.DateOffset(months=3)], "Последни 3 месеци"
    if choice == "2":
        return master[master["Date"] >= max_date - pd.DateOffset(months=6)], "Последни 6 месеци"
    if choice == "3":
        return master[master["Date"] >= max_date - pd.DateOffset(months=12)], "Последни 12 месеци"
    if choice == "4":
        cutoff = current_season_start(max_date)
        return master[master["Date"] >= cutoff], f"Тековна сезона (од {cutoff.strftime('%d.%m.%Y')})"
    return master, "Без временски филтер (Сите податоци)"


def export_results(ranked: list[tuple[str, float]], counts: dict[str, int]) -> pd.DataFrame:
    results = pd.DataFrame(ranked, columns=["Team", "PageRank_Score"])
    results["Matches_Played"] = results["Team"].map(counts)
    results["PageRank_Score"] = results["PageRank_Score"].round(5)

    excel_filename = f"PageRank_Rezultati_Scenarioto_{SCENARIO}.xlsx"
    results.to_excel(excel_filename, index=False, sheet_name="PageRank Top List")
    print(f"\n[ИНФО] Резултатите се зачувани во Excel: {excel_filename}")
    return results


def save_top15_chart(results: pd.DataFrame) -> None:
    if results.empty:
        return
    top_15 = results.head(15)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="PageRank_Score", y="Team", data=top_15, hue="Team", palette="Blues_r", legend=False)
    plt.title(f"Top 15 Football Teams - PageRank Model (Scenario {SCENARIO})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("PageRank Score", fontsize=12)
    plt.ylabel("Team", fontsize=12)
    plt.tight_layout()

    graph_filename = os.path.abspath(f"PageRank_Top15_Scenarioto_{SCENARIO}.png")
    plt.savefig(graph_filename, dpi=300)
    plt.close()
    print(f"[ИНФО] Визуелниот граф е зачуван како слика: {graph_filename}")


def main() -> None:
    download_latest_data()
    master = load_data(os.path.join(DATA_DIR, "*.csv"))
    report_unmatched_team_names(master)

    if master.empty or "Date" not in master.columns:
        print("Нема доволно податоци.")
        sys.exit(1)

    max_date = pd.Timestamp(master["Date"].max())

    # Сценарио 3 користи фиксно последни 150 дена (без мени), додека Сценарио 1 и 2 бараат избор од мени
    if SCENARIO == 3:
        cutoff = max_date - pd.DateOffset(days=150)
        filtered = master[master["Date"] >= cutoff]
        label = "Сценарио 3: Фиксно последни 150 дена (Линеарно опаѓање)"
    else:
        choice = prompt_horizon_choice()
        filtered, label = apply_time_filter(master, choice, max_date)

    print(f"\n[ИЗБРАНО] {label}")
    print(f"Вкупен број на натпревари по филтрирањето: {len(filtered)}")
    print(f"ПОСЛЕДЕН ДАТУМ ВО АНАЛИЗАТА: {max_date.strftime('%d.%m.%Y')} (Сценарио {SCENARIO})")
    print("=" * 50)

    if filtered.empty:
        print("Нема натпревари во избраниот период.")
        sys.exit(1)

    graph, counts = build_graph(filtered, max_date)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        print("Графот е празен.")
        sys.exit(1)

    ranked = compute_ranking(graph, counts, MIN_MATCHES)
    if not ranked:
        print(f"Нема клубови со најмалку {MIN_MATCHES} натпревари.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"--- РАНГ ЛИСТА (Сценарио {SCENARIO}), мин. {MIN_MATCHES} натпревари ---")
    print("=" * 50)
    for index, (team, score) in enumerate(ranked[:25], 1):
        print(f"{index:2d}. {team:<22}: {score:.5f}  (натпревари: {counts[team]})")

    results = export_results(ranked, counts)
    save_top15_chart(results)


if __name__ == "__main__":
    main()
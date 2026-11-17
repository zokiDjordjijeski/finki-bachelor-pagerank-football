import sys
import os
import glob
import warnings
from datetime import timedelta

import pandas as pd
import networkx as nx

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЈА И ИЗБОР НА СЦЕНАРИО
# ---------------------------------------------------------------------------
# Избери сценарио: 1 (сите лиги рамноправни) или 2 (прогресивни европски коефициенти)
SCENARIO = 1

if SCENARIO == 1:
    COMPETITION_COEF = {
        "Premier League": 1.0,
        "Champions League": 1.0,
        "La Liga": 1.0,
        "Serie A": 1.0,
        "Bundesliga": 1.0,
        "Ligue 1": 1.0,
        "Eredivisie": 1.0,
        "Europa League": 1.0,
        "Conference League": 1.0,
        "Championship": 1.0,
        "Scottish Premiership": 1.0,
    }
else:
    # Сценарио 2: Реална хиерархија (врвни лиги 1.0, послаби/втори лиги со помал пондер)
    COMPETITION_COEF = {
        # Прва класа - Врвни европски лиги
        "Premier League": 1.0,
        "La Liga": 1.0,
        "Serie A": 1.0,
        "Bundesliga": 1.0,
        "Ligue 1": 1.0,
        "Eredivisie": 0.8,

        # Понизок ранг - Чемпионшип и Шкотска со поагресивно намалени коефициенти
        "Championship": 0.6,
        "Scottish Premiership": 0.4,

        # Европски купови со повисок пондер (пред ескалацијата по фази)
        "Champions League": 1.5,
        "Europa League": 1.3,
        "Conference League": 1.2,
    }
# Домашни лиги: код на фајл -> име на натпреварување
LEAGUE_CODES = {
    "e0": "Premier League",
    "e1": "Championship",
    "sp1": "La Liga",
    "i1": "Serie A",
    "d1": "Bundesliga",
    "f1": "Ligue 1",
    "n1": "Eredivisie",
    "sc0": "Scottish Premiership",
}

# Што добива губитникот назад (спречува "dangling" чворови без излезни рабови)
LOSS_BACK = 0.10
GOAL_BONUS = 0.05  # бонус по гол разлика (макс 3 гола)

# Мапирање на имиња на тимови за да се спојат домашните и европските фајлови
TEAM_NAME_MAPPING = {
    "Paris SG": "PSG",  # Го спојува "Paris SG" од француска лига со "PSG" од ЛШ
}

COLUMN_MAPPING = {
    "Home": "HomeTeam", "Away": "AwayTeam",
    "homeTeam": "HomeTeam", "awayTeam": "AwayTeam",
    "Team1": "HomeTeam", "Team2": "AwayTeam",
    "Home_Team": "HomeTeam", "Away_Team": "AwayTeam",
    "HG": "FTHG", "AG": "FTAG",
    "homeScore": "FTHG", "awayScore": "FTAG",
    "HomeGoals": "FTHG", "AwayGoals": "FTAG",
    "matchDate": "Date", "date": "Date",
}


def detect_competition(path):
    """Строго препознавање според името на фајлот (case-insensitive)."""
    name = os.path.basename(path).lower()
    if "champions" in name:
        return "Champions League"
    if "europa" in name:
        return "Europa League"
    if "conference" in name:
        return "Conference League"

    for code, comp in LEAGUE_CODES.items():
        if name.startswith(code):
            return comp
    return None


def stage_multiplier(comp, match_date):
    """Го зголемува коефициентот за елиминациските фази во еврокуповите (Само за Сценарио 2)."""
    if SCENARIO == 1 or comp not in ["Champions League", "Europa League", "Conference League"]:
        return 1.0

    m = match_date.month
    d = match_date.day

    if m == 2:
        return 1.1  # Плејоф за нокаут фаза
    elif m == 3:
        return 1.2  # Осминафинале (1/8 финале)
    elif m == 4:
        return 1.4  # Четвртфинале (1/4 финале)
    elif m == 5:
        if d < 20:
            return 1.6  # Полуфинале
        else:
            return 1.8  # Финале
    elif m == 6:
        return 1.8  # Финале (ако е во јуни)

    return 1.0  # Групна / Лигашка фаза (Септември - Јануари)


def time_weight_for(days):
    months = max(days, 0) / 30.0
    for limit, w in [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7), (5, 0.6), (6, 0.5), (7, 0.4), (10, 0.3)]:
        if months <= limit:
            return w
    return 0.2


def add_edge(G, u, v, w):
    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w)


def load_data(pattern, only=None):
    files = sorted(glob.glob(pattern))
    print(f"Пронајдени се {len(files)} CSV фајлови. (Активно Сценарио: {SCENARIO})")
    frames = []
    print(f"\n{'Фајл':<45}{'Натпреварување':<24}{'Коеф.':>7}{'Редови':>8}")
    for f in files:
        comp = detect_competition(f)
        if comp is None:
            print(f"{os.path.basename(f):<45}{'НЕПРЕПОЗНАЕН - прескокнат':<24}")
            continue
        if only and comp.lower() != only.lower():
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Грешка кај {f}: {e}")
            continue
        df = df.rename(columns=COLUMN_MAPPING)
        need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not need.issubset(df.columns):
            print(f"{os.path.basename(f):<45}{'недостасуваат колони':<24}")
            continue
        df = df[list(need)].copy()
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna()
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
        df["Comp"] = comp
        df["LeagueWeight"] = COMPETITION_COEF[comp]
        print(f"{os.path.basename(f):<45}{comp:<24}{COMPETITION_COEF[comp]:>7.3f}{len(df):>8}")
        frames.append(df)
    if not frames:
        print("Нема податоци за обработка.")
        sys.exit()
    return pd.concat(frames, ignore_index=True)


def check_team_names(master):
    euro = master[master.Comp.isin(["Champions League", "Europa League", "Conference League"])]
    dom = master[~master.Comp.isin(["Champions League", "Europa League", "Conference League"])]
    euro_teams = set(euro.HomeTeam) | set(euro.AwayTeam)
    dom_teams = set(dom.HomeTeam) | set(dom.AwayTeam)
    missing = sorted(euro_teams - dom_teams)
    print(f"\n[ПРОВЕРКА ИМИЊА] {len(missing)} еврокуп-тимови не постојат во домашните фајлови.")
    print(", ".join(missing))


def build_graph(df, max_date):
    G = nx.DiGraph()
    counts = {}
    for row in df.itertuples():
        h, a = row.HomeTeam, row.AwayTeam
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1

        tw = time_weight_for((max_date - row.Date).days)
        sm = stage_multiplier(row.Comp, row.Date)
        base = tw * row.LeagueWeight * sm

        if row.FTHG > row.FTAG:
            winner, loser, gd = h, a, row.FTHG - row.FTAG
        elif row.FTAG > row.FTHG:
            winner, loser, gd = a, h, row.FTAG - row.FTHG
        else:
            w = 0.55 * base
            add_edge(G, h, a, w)
            add_edge(G, a, h, w)
            continue
        w = (1.0 + GOAL_BONUS * min(gd, 3)) * base
        add_edge(G, loser, winner, w)
        add_edge(G, winner, loser, LOSS_BACK * w)
    return G, counts


def points_table(df_league):
    pts, gd = {}, {}
    for r in df_league.itertuples():
        h, a = r.HomeTeam, r.AwayTeam
        pts.setdefault(h, 0);
        pts.setdefault(a, 0)
        gd[h] = gd.get(h, 0) + r.FTHG - r.FTAG
        gd[a] = gd.get(a, 0) + r.FTAG - r.FTHG
        if r.FTHG > r.FTAG:
            pts[h] += 3
        elif r.FTAG > r.FTHG:
            pts[a] += 3
        else:
            pts[h] += 1;
            pts[a] += 1
    return pts


def main():
    choice_arg = sys.argv[1] if len(sys.argv) > 1 else None
    only_code = sys.argv[2].lower() if len(sys.argv) > 2 else None
    only = LEAGUE_CODES.get(only_code) if only_code else None

    master = load_data("data/*.csv", only=only)
    if not only:
        check_team_names(master)
    max_date = master["Date"].max()

    print("\n" + "=" * 50)
    print(f"ПОСЛЕДЕН ДАТУМ ВО БАЗАТА: {max_date.strftime('%d.%m.%Y')} (Сценарио {SCENARIO})")
    print("=" * 50)
    print("1 - 7 дена | 2 - 30 дена | 3 - 90 дена | 4 - 180 дена | 5 - сите")
    choice = choice_arg or input("Избор (1-5): ").strip()

    days_map = {"1": (7, 2), "2": (30, 5), "3": (90, 10), "4": (180, 15)}
    if choice in days_map:
        days, min_matches = days_map[choice]
        df = master[master["Date"] >= max_date - timedelta(days=days)].copy()
        label = f"последни {days} дена"
    else:
        df = master.copy()
        min_matches = 25
        label = "сите податоци"
    if only:
        min_matches = min(min_matches, 10)
    print(f"Анализирање на {len(df)} натпревари ({label})...")

    G, counts = build_graph(df, max_date)
    scores = nx.pagerank(G, weight="weight", alpha=0.85)
    ranked = sorted(
        [(t, s) for t, s in scores.items() if counts.get(t, 0) >= min_matches],
        key=lambda x: x[1], reverse=True,
    )

    print("\n" + "=" * 50)
    print(f"--- РАНГ ЛИСТА (Сценарио {SCENARIO} - {label}), мин. {min_matches} натпревари ---")
    print("=" * 50)
    for i, (t, s) in enumerate(ranked[:25 if not only else 30], 1):
        print(f"{i:2d}. {t:<22}: {s:.5f}  (натпревари: {counts[t]})")

    if only:
        pts = points_table(df)
        common = [t for t, _ in ranked if t in pts]
        if len(common) >= 5:
            pr = pd.Series({t: scores[t] for t in common})
            pt = pd.Series({t: pts[t] for t in common})
            print(f"\n[ВАЛИДАЦИЈА] Spearman корелација PageRank vs бодови: {pr.corr(pt, method='spearman'):.3f}")
            print("\nТабела по бодови (за споредба):")
            for i, (t, p) in enumerate(sorted(pts.items(), key=lambda x: -x[1]), 1):
                print(f"{i:2d}. {t:<22}{p:>4} бода")


if __name__ == "__main__":
    main()
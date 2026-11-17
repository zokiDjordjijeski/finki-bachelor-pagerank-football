import sys
import os
import glob
import warnings
import requests

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. АВТОМАТСКО СИМНУВАЊЕ НА НАЈНОВИТЕ ФАЈЛОВИ ЗА ДОМАШНИТЕ ЛИГИ
# ---------------------------------------------------------------------------
def download_latest_data():
    leagues = {
        "E0": "https://football-data.co.uk/mmz4281/2627/E0.csv",
        "E1": "https://football-data.co.uk/mmz4281/2627/E1.csv",
        "SP1": "https://football-data.co.uk/mmz4281/2627/SP1.csv",
        "I1": "https://football-data.co.uk/mmz4281/2627/I1.csv",
        "D1": "https://football-data.co.uk/mmz4281/2627/D1.csv",
        "F1": "https://football-data.co.uk/mmz4281/2627/F1.csv",
        "N1": "https://football-data.co.uk/mmz4281/2627/N1.csv",
        "SC0": "https://football-data.co.uk/mmz4281/2627/SC0.csv",
    }

    os.makedirs("data", exist_ok=True)
    print("Проверка на локалните фајлови и обид за ажурирање...")
    for code, url in leagues.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                file_path = os.path.join("data", f"{code}_26-27.csv")
                with open(file_path, "wb") as f:
                    f.write(response.content)
        except Exception:
            pass
    print("Проверката заврши.\n")


# ---------------------------------------------------------------------------
# 2. КОНФИГУРАЦИЈА И ИЗБОР НА СЦЕНАРИО
# ---------------------------------------------------------------------------
SCENARIO = 2

COMPETITION_COEF = {
    "Premier League": 1.0, "La Liga": 1.0, "Serie A": 1.0,
    "Bundesliga": 1.0, "Ligue 1": 1.0, "Eredivisie": 0.8,
    "Championship": 0.6, "Scottish Premiership": 0.4,
    "National Cup": 0.9,
    "Champions League": 1.5,
    "Europa League": 1.3,
    "Conference League": 1.2,
}

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

LOSS_BACK = 0.10
GOAL_BONUS = 0.05

TEAM_NAME_MAPPING = {
    "Paris SG": "PSG",
    "Paris Saint Germain": "PSG",
    "Paris Saint-Germain": "PSG",
    "Bayern München": "Bayern Munich",
    "Manchester Utd": "Manchester United",
    "Man United": "Manchester United",
}

COLUMN_MAPPING = {
    "Home": "HomeTeam", "Away": "AwayTeam",
    "homeTeam": "HomeTeam", "awayTeam": "AwayTeam",
    "Team1": "HomeTeam", "Team2": "AwayTeam",
    "Home_Team": "HomeTeam", "Away_Team": "AwayTeam",
    "HG": "FTHG", "AG": "FTAG",
    "homeScore": "FTHG", "awayScore": "FTAG",
    "HomeGoals": "FTHG", "AwayGoals": "FTAG",
    "matchDate": "Date", "date": "Date", "Date": "Date"
}


def detect_competition(path):
    name = os.path.basename(path).lower()
    if "champions" in name:
        return "Champions League"
    if "europa" in name:
        return "Europa League"
    if "conference" in name:
        return "Conference League"
    if "cup" in name or "fa" in name or "copa" in name:
        return "National Cup"

    for code, comp in LEAGUE_CODES.items():
        if name.startswith(code):
            return comp
    return None


def stage_multiplier(comp, match_date):
    if SCENARIO == 1 or comp not in ["Champions League", "Europa League", "Conference League"]:
        return 1.0
    if pd.isna(match_date):
        return 1.0

    m = match_date.month
    d = match_date.day

    if m == 2:
        return 1.1
    elif m == 3:
        return 1.2
    elif m == 4:
        return 1.4
    elif m == 5:
        return 1.6 if d < 20 else 1.8
    elif m == 6:
        return 1.8
    return 1.0


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


def load_data(pattern):
    files = sorted(glob.glob(pattern))
    print(f"Пронајдени се {len(files)} CSV фајлови во папката data. (Активно Сценарио: {SCENARIO})")
    frames = []
    print(f"\n{'Фајл':<45}{'Натпреварување':<24}{'Коеф.':>7}{'Редови':>8}")

    for f in files:
        comp = detect_competition(f)
        if comp is None:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        df = df.rename(columns=COLUMN_MAPPING)

        if "Score" in df.columns and ("FTHG" not in df.columns or df["FTHG"].isna().all()):
            score_split = df["Score"].dropna().str.split("-", expand=True)
            if score_split.shape[1] == 2:
                df["FTHG"] = pd.to_numeric(score_split[0], errors="coerce")
                df["FTAG"] = pd.to_numeric(score_split[1], errors="coerce")

        need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not need.issubset(df.columns):
            continue

        df = df[list(need)].copy()
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna()
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip().replace(TEAM_NAME_MAPPING)
        df["Comp"] = comp
        df["LeagueWeight"] = COMPETITION_COEF.get(comp, 1.0)

        print(f"{os.path.basename(f):<45}{comp:<24}{df['LeagueWeight'].iloc[0]:>7.3f}{len(df):>8}")
        frames.append(df)

    if not frames:
        print("Нема податоци за обработка.")
        sys.exit()
    return pd.concat(frames, ignore_index=True)


def build_graph(df, max_date):
    G = nx.DiGraph()
    counts = {}
    for row in df.itertuples():
        h, a = row.HomeTeam, row.AwayTeam
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1

        days_diff = (max_date - row.Date).days if pd.notna(row.Date) else 0
        tw = time_weight_for(days_diff)
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


def main():
    download_latest_data()
    master = load_data("data/*.csv")

    if master.empty or "Date" not in master.columns:
        print("Нема доволно податоци.")
        sys.exit()

    max_date = master["Date"].max()

    print("\n" + "=" * 50)
    print(" ИЗБЕРЕТЕ ВРЕМЕНСКИ ПЕРИОД ЗА PAGERANK МОДЕЛОТ:")
    print("=" * 50)
    print(" 1. Последни 3 месеци")
    print(" 2. Последни 6 месеци")
    print(" 3. Последни 12 месеци (Една година)")
    print(" 4. Само тековната сезона (од Август 2026 па натаму)")
    print(" 5. Сите достапни податоци (Без филтер)")
    print("=" * 50)

    choice = input("Внесете број на опција (1-5): ").strip()

    if choice == "1":
        cutoff_date = max_date - pd.DateOffset(months=3)
        master = master[master["Date"] >= cutoff_date]
        print("\n[ИЗБРАНО] Филтрирано: Последни 3 месеци")
    elif choice == "2":
        cutoff_date = max_date - pd.DateOffset(months=6)
        master = master[master["Date"] >= cutoff_date]
        print("\n[ИЗБРАНО] Филтрирано: Последни 6 месеци")
    elif choice == "3":
        cutoff_date = max_date - pd.DateOffset(months=12)
        master = master[master["Date"] >= cutoff_date]
        print("\n[ИЗБРАНО] Филтрирано: Последни 12 месеци")
    elif choice == "4":
        cutoff_date = pd.Timestamp("2026-08-01")
        master = master[master["Date"] >= cutoff_date]
        print("\n[ИЗБРАНО] Филтрирано: Тековна сезона (од Август 2026)")
    else:
        print("\n[ИЗБРАНО] Без временски филтер (Сите податоци)")

    print(f"Вкупен број на натпревари по филтрирањето: {len(master)}")
    print(f"ПОСЛЕДЕН ДАТУМ ВО АНАЛИЗАТА: {max_date.strftime('%d.%m.%Y')} (Сценарио {SCENARIO})")
    print("=" * 50)

    df = master.copy()
    min_matches = 10  # Намалено на 10 за да се сигурни дека сите клубови ќе влезат во листата

    G, counts = build_graph(df, max_date)
    scores = nx.pagerank(G, weight="weight", alpha=0.85)
    ranked = sorted(
        [(t, s) for t, s in scores.items() if counts.get(t, 0) >= min_matches],
        key=lambda x: x[1], reverse=True,
    )

    print("\n" + "=" * 50)
    print(f"--- РАНГ ЛИСТА (Сценарио {SCENARIO}), мин. {min_matches} натпревари ---")
    print("=" * 50)
    for i, (t, s) in enumerate(ranked[:25], 1):
        print(f"{i:2d}. {t:<22}: {s:.5f}  (натпревари: {counts[t]})")

    results_df = pd.DataFrame(ranked, columns=["Team", "PageRank_Score"])
    results_df["Matches_Played"] = results_df["Team"].map(counts)
    results_df["PageRank_Score"] = results_df["PageRank_Score"].round(5)

    excel_filename = f"PageRank_Rezultati_Scenarioto_{SCENARIO}.xlsx"
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="PageRank Top List")
    print(f"\n[ИНФО] Резултатите се зачувани во Excel: {excel_filename}")

    # Визуелен граф за Топ 15 (безбедно цртање)
    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        top_15 = results_df.head(15)
        sns.barplot(x="PageRank_Score", y="Team", data=top_15, palette="Blues_r")

        plt.title(f"Top 15 Football Teams - PageRank Model (Scenario {SCENARIO})", fontsize=14, fontweight="bold",
                  pad=15)
        plt.xlabel("PageRank Score", fontsize=12)
        plt.ylabel("Team", fontsize=12)
        plt.tight_layout()

        graph_filename = f"PageRank_Top15_Scenarioto_{SCENARIO}.png"
        plt.savefig(graph_filename, dpi=300)
        plt.close()
        print(f"[ИНФО] Визуелниот граф е зачуван како слика: {graph_filename}")


if __name__ == "__main__":
    main()
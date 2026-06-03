import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PandemicLens",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Blue & White Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    [data-testid="stSidebar"] { background-color: #1a3a5c !important; }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] a { color: #90caf9 !important; }
    .metric-card {
        background: #ffffff; border-radius: 12px;
        padding: 20px 22px; border: none;
        box-shadow: 0 2px 8px rgba(26,90,160,0.10);
    }
    .metric-value { font-size: 30px; font-weight: 700; }
    .metric-label { font-size: 13px; color: #5a7a9a; margin-top: 4px; }
    .section-title {
        font-size: 16px; font-weight: 600; color: #1a3a5c;
        margin: 1.2rem 0 0.6rem;
        border-left: 4px solid #1a7ad4; padding-left: 10px;
    }
    .title-bar {
        background: linear-gradient(90deg, #1a3a5c 0%, #1a7ad4 100%);
        border-radius: 14px; padding: 22px 28px;
        margin-bottom: 1.2rem; display: flex;
        align-items: center; gap: 16px;
    }
    .title-bar h1 { color: #ffffff !important; font-size: 26px; font-weight: 700; margin: 0; }
    .title-bar p  { color: #b3d4f5; font-size: 13px; margin: 4px 0 0; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Offline sample data ───────────────────────────────────────────────────────
def make_sample_data():
    # GDP per capita values (approximate real values in USD)
    countries = {
        "India":          (1_400_000_000, 44_000_000, 500_000, 2_200_000_000, 2100),
        "United States":  (  335_000_000,103_000_000,1_100_000,680_000_000,  63000),
        "Brazil":         (  215_000_000, 37_000_000, 700_000, 500_000_000,   7500),
        "United Kingdom": (   68_000_000, 24_000_000, 230_000, 160_000_000,  42000),
        "Germany":        (   84_000_000, 38_000_000, 180_000, 190_000_000,  46000),
    }
    dates = pd.date_range("2020-01-22", "2023-05-01", freq="D")
    n = len(dates)
    rows = []
    for country, (pop, peak_cases, peak_deaths, peak_vacc, gdp) in countries.items():
        t      = np.linspace(0, 4 * np.pi, n)
        wave   = (np.sin(t - np.pi / 2) + 1) / 2
        noise  = np.random.normal(0, 0.02, n)
        growth = np.clip(np.cumsum(np.abs(wave + noise) / n), 0, 1)
        total_cases  = (growth * peak_cases).astype(int)
        total_deaths = (growth * peak_deaths).astype(int)
        total_vacc   = np.zeros(n, dtype=int)
        vs = int(n * 0.35)
        total_vacc[vs:] = (
            np.clip(np.linspace(0, 1, n - vs), 0, 1) * peak_vacc
        ).astype(int)
        new_cases  = np.diff(total_cases,  prepend=0).clip(0)
        new_deaths = np.diff(total_deaths, prepend=0).clip(0)
        for i, d in enumerate(dates):
            rows.append({
                "iso_code": "XXX", "continent": "Asia", "location": country,
                "date": d, "population": pop,
                "total_cases":             int(total_cases[i]),
                "new_cases":               int(new_cases[i]),
                "total_deaths":            int(total_deaths[i]),
                "new_deaths":              int(new_deaths[i]),
                "total_vaccinations":      int(total_vacc[i]),
                "people_fully_vaccinated": int(total_vacc[i] * 0.45),
                "gdp_per_capita":          gdp,   # ← GDP added
            })
    return pd.DataFrame(rows)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading COVID-19 dataset...")
def load_data():
    url  = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    cols = [
        "iso_code", "continent", "location", "date",
        "total_cases", "new_cases",
        "total_deaths", "new_deaths",
        "total_vaccinations", "people_fully_vaccinated",
        "population",
        "gdp_per_capita",    # ← GDP added here
    ]
    try:
        df = pd.read_csv("owid-covid-data.csv", usecols=cols, parse_dates=["date"])
        df = df[df["iso_code"].str.len() == 3].copy()
        df.fillna(0, inplace=True)
        return df, "local"
    except Exception:
        pass
    try:
        df = pd.read_csv(url, usecols=cols, parse_dates=["date"])
        df = df[df["iso_code"].str.len() == 3].copy()
        df.fillna(0, inplace=True)
        return df, "live"
    except Exception:
        return make_sample_data(), "demo"


df, data_source = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌍 PandemicLens")
st.sidebar.markdown("---")

all_countries     = sorted(df["location"].unique())
default_countries = [c for c in
                     ["India", "United States", "Brazil", "United Kingdom", "Germany"]
                     if c in all_countries]
selected_countries = st.sidebar.multiselect(
    "Select Countries", all_countries, default=default_countries
)

min_date   = df["date"].min().date()
max_date   = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime(2020, 3, 1).date(), max_date),
    min_value=min_date,
    max_value=max_date,
)
start_date, end_date = (
    (date_range[0], date_range[1]) if len(date_range) == 2
    else (min_date, max_date)
)

metric_choice = st.sidebar.selectbox(
    "Primary Metric",
    ["total_cases", "total_deaths", "new_cases", "new_deaths"],
    format_func=lambda x: x.replace("_", " ").title(),
)

st.sidebar.markdown("---")
source_label = {"local": "📂 Local CSV", "live": "🌐 Live Data", "demo": "🔬 Demo Data"}
st.sidebar.caption(source_label[data_source])
st.sidebar.caption("Built with Python · Streamlit · Pandas")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not selected_countries:
    st.warning("Please select at least one country from the sidebar.")
    st.stop()

# ── Filter ────────────────────────────────────────────────────────────────────
mask = (
    df["location"].isin(selected_countries)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
filtered = df[mask].copy()

latest = (
    df[df["location"].isin(selected_countries)]
    .sort_values("date")
    .groupby("location")
    .last()
    .reset_index()
)

# ── Title bar ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="title-bar">
        <div>
            <h1>🌍 PandemicLens </h1>
            <p>
                {start_date} → {end_date} &nbsp;•&nbsp;
                {len(selected_countries)} countries selected &nbsp;•&nbsp;
                {source_label[data_source]}
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
total_confirmed = int(latest["total_cases"].sum())
total_deaths    = int(latest["total_deaths"].sum())
total_vacc      = int(latest["total_vaccinations"].sum())
death_rate      = (total_deaths / total_confirmed * 100) if total_confirmed else 0

kpi_cols = st.columns(4)
kpis = [
    ("Total Confirmed", f"{total_confirmed:,}", "#1a7ad4"),
    ("Total Deaths",    f"{total_deaths:,}",    "#c0392b"),
    ("Vaccinations",    f"{total_vacc:,}",       "#1a9a5c"),
    ("Death Rate",      f"{death_rate:.2f}%",    "#e67e22"),
]
for col, (label, value, color) in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
palette = sns.color_palette("tab10", max(len(selected_countries), 1))

# ── Row 1: Line chart + Bar chart ─────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        f'<div class="section-title">'
        f'{metric_choice.replace("_"," ").title()} Over Time (7-day avg)'
        f'</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7faff")
    for i, country in enumerate(selected_countries):
        cdf = filtered[filtered["location"] == country].sort_values("date")
        if cdf.empty:
            continue
        rolling = cdf.set_index("date")[metric_choice].rolling(7).mean()
        ax.plot(rolling.index, rolling.values, label=country,
                color=palette[i], linewidth=2.2)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(metric_choice.replace("_", " ").title(), fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9, framealpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#b0c4de")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown('<div class="section-title">Total Cases by Country</div>',
                unsafe_allow_html=True)
    bar_data = latest.sort_values("total_cases", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#f7faff")
    ax2.barh(bar_data["location"], bar_data["total_cases"],
             color=sns.color_palette("Blues_r", len(bar_data)))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6):.0f}M" if x >= 1e6 else f"{int(x):,}"
    ))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xlabel("Total Cases", fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.4, color="#b0c4de")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Row 2: Death rate + Heatmap ───────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-title">Death Rate Comparison (%)</div>',
                unsafe_allow_html=True)
    latest["death_rate"] = np.where(
        latest["total_cases"] > 0,
        latest["total_deaths"] / latest["total_cases"] * 100, 0,
    )
    dr = latest.sort_values("death_rate", ascending=False)
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    fig3.patch.set_facecolor("#ffffff")
    ax3.set_facecolor("#f7faff")
    bars = ax3.bar(dr["location"], dr["death_rate"],
                   color=sns.color_palette("Blues_r", len(dr)), edgecolor="none")
    for bar, val in zip(bars, dr["death_rate"]):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Death Rate (%)", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", linestyle="--", alpha=0.4, color="#b0c4de")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col4:
    st.markdown('<div class="section-title">Monthly New Cases Heatmap</div>',
                unsafe_allow_html=True)
    pivot_country = selected_countries[0]
    hm_df = filtered[filtered["location"] == pivot_country].copy()
    hm_df["month"] = hm_df["date"].dt.to_period("M").astype(str)
    pivot_df = hm_df.pivot_table(
        values="new_cases", index="month", aggfunc="sum"
    ).tail(18)
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    fig4.patch.set_facecolor("#ffffff")
    if not pivot_df.empty:
        sns.heatmap(pivot_df, ax=ax4, cmap="Blues",
                    linewidths=0.3, linecolor="#e8f0fe",
                    annot=False, cbar_kws={"shrink": 0.8})
        ax4.set_title(f"Monthly new cases — {pivot_country}", fontsize=10, pad=6)
        ax4.set_xlabel(""); ax4.set_ylabel("")
        plt.yticks(fontsize=8, rotation=0)
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax4.transAxes)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ── Row 3: Vaccination ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Vaccination Progress (Total Doses)</div>',
            unsafe_allow_html=True)
vacc_df = filtered[filtered["total_vaccinations"] > 0]
if not vacc_df.empty:
    fig5, ax5 = plt.subplots(figsize=(12, 3.5))
    fig5.patch.set_facecolor("#ffffff")
    ax5.set_facecolor("#f7faff")
    for i, country in enumerate(selected_countries):
        cdf = vacc_df[vacc_df["location"] == country].sort_values("date")
        if cdf.empty:
            continue
        ax5.plot(cdf["date"], cdf["total_vaccinations"],
                 label=country, color=palette[i], linewidth=2.2)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{int(x):,}"
    ))
    ax5.set_xlabel("Date", fontsize=11)
    ax5.set_ylabel("Total Vaccinations", fontsize=11)
    ax5.legend(fontsize=9, framealpha=0.5)
    ax5.spines[["top", "right"]].set_visible(False)
    ax5.grid(axis="y", linestyle="--", alpha=0.4, color="#b0c4de")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()
else:
    st.info("No vaccination data available for the selected filters.")


# ════════════════════════════════════════════════════════════════════════════
# ── GDP ECONOMIC IMPACT SECTION  (NEW) ──────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div class="section-title">💰 Economic Impact — GDP Analysis</div>',
    unsafe_allow_html=True,
)
st.caption("Analyzing how COVID-19 affected the GDP per capita of selected countries")

# Build GDP summary — one row per country with gdp_per_capita
gdp_data = (
    df[df["location"].isin(selected_countries) & (df["gdp_per_capita"] > 0)]
    .groupby("location")["gdp_per_capita"]
    .mean()
    .reset_index()
    .sort_values("gdp_per_capita", ascending=True)
)

# Simulated GDP growth rates (2018–2022) for selected countries
# Using approximate World Bank figures
gdp_growth = {
    "India":          {"2018": 6.5, "2019": 4.0, "2020": -6.6, "2021": 8.7,  "2022": 7.2},
    "United States":  {"2018": 3.0, "2019": 2.3, "2020": -3.4, "2021": 5.7,  "2022": 2.1},
    "Brazil":         {"2018": 1.8, "2019": 1.2, "2020": -3.9, "2021": 4.6,  "2022": 2.9},
    "United Kingdom": {"2018": 1.3, "2019": 1.4, "2020": -9.3, "2021": 7.4,  "2022": 4.0},
    "Germany":        {"2018": 1.5, "2019": 0.6, "2020": -4.6, "2021": 2.6,  "2022": 1.8},
    "France":         {"2018": 1.8, "2019": 1.5, "2020": -7.9, "2021": 6.8,  "2022": 2.5},
    "Italy":          {"2018": 0.9, "2019": 0.3, "2020": -9.0, "2021": 6.7,  "2022": 3.7},
    "Canada":         {"2018": 2.4, "2019": 1.9, "2020": -5.2, "2021": 4.5,  "2022": 3.4},
    "Japan":          {"2018": 0.6, "2019":-0.4, "2020": -4.3, "2021": 2.1,  "2022": 1.0},
    "China":          {"2018": 6.7, "2019": 6.0, "2020":  2.3, "2021": 8.1,  "2022": 3.0},
    "Russia":         {"2018": 2.5, "2019": 2.0, "2020": -2.7, "2021": 4.7,  "2022":-2.1},
    "Australia":      {"2018": 2.7, "2019": 1.9, "2020": -2.2, "2021": 4.9,  "2022": 3.7},
    "South Korea":    {"2018": 2.9, "2019": 2.2, "2020": -0.9, "2021": 4.1,  "2022": 2.6},
    "Spain":          {"2018": 2.4, "2019": 2.0, "2020":-10.8, "2021": 5.5,  "2022": 5.5},
    "Mexico":         {"2018": 2.2, "2019":-0.2, "2020": -8.2, "2021": 4.8,  "2022": 3.0},
}

years = ["2018", "2019", "2020", "2021", "2022"]

# ── GDP Row 1: GDP per capita bar + GDP growth line ────────────────────────
gdp_col1, gdp_col2 = st.columns(2)

with gdp_col1:
    st.markdown(
        '<div class="section-title" style="font-size:14px;">GDP Per Capita by Country (USD)</div>',
        unsafe_allow_html=True,
    )
    if not gdp_data.empty:
        fig_g1, ax_g1 = plt.subplots(figsize=(6, 3.5))
        fig_g1.patch.set_facecolor("#ffffff")
        ax_g1.set_facecolor("#f7faff")
        colors = sns.color_palette("Blues_r", len(gdp_data))
        bars = ax_g1.barh(gdp_data["location"], gdp_data["gdp_per_capita"],
                          color=colors, edgecolor="none")
        for bar, val in zip(bars, gdp_data["gdp_per_capita"]):
            ax_g1.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                       f"${val:,.0f}", va="center", fontsize=9)
        ax_g1.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}K")
        )
        ax_g1.spines[["top", "right"]].set_visible(False)
        ax_g1.set_xlabel("GDP Per Capita (USD)", fontsize=10)
        ax_g1.grid(axis="x", linestyle="--", alpha=0.4, color="#b0c4de")
        plt.tight_layout()
        st.pyplot(fig_g1)
        plt.close()
    else:
        st.info("GDP data not available for selected countries.")

with gdp_col2:
    st.markdown(
        '<div class="section-title" style="font-size:14px;">GDP Growth Rate 2018–2022 (%)</div>',
        unsafe_allow_html=True,
    )
    available = [c for c in selected_countries if c in gdp_growth]
    if available:
        fig_g2, ax_g2 = plt.subplots(figsize=(6, 3.5))
        fig_g2.patch.set_facecolor("#ffffff")
        ax_g2.set_facecolor("#f7faff")
        for i, country in enumerate(available):
            vals = [gdp_growth[country][y] for y in years]
            ax_g2.plot(years, vals, marker="o", label=country,
                       color=palette[i % len(palette)], linewidth=2.2, markersize=5)
        # Highlight 2020 crash zone
        ax_g2.axvspan("2019", "2021", alpha=0.07, color="red", label="COVID crash zone")
        ax_g2.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        ax_g2.set_ylabel("GDP Growth Rate (%)", fontsize=10)
        ax_g2.legend(fontsize=8, framealpha=0.5)
        ax_g2.spines[["top", "right"]].set_visible(False)
        ax_g2.grid(axis="y", linestyle="--", alpha=0.4, color="#b0c4de")
        plt.tight_layout()
        st.pyplot(fig_g2)
        plt.close()
    else:
        st.info("GDP growth data not available for selected countries in demo mode.")

# ── GDP Row 2: 2020 GDP drop bar + Cases vs GDP scatter ───────────────────
gdp_col3, gdp_col4 = st.columns(2)

with gdp_col3:
    st.markdown(
        '<div class="section-title" style="font-size:14px;">GDP Drop in 2020 — COVID Impact (%)</div>',
        unsafe_allow_html=True,
    )
    available = [c for c in selected_countries if c in gdp_growth]
    if available:
        drops = {c: gdp_growth[c]["2020"] for c in available}
        drop_df = pd.DataFrame(
            list(drops.items()), columns=["country", "gdp_growth_2020"]
        ).sort_values("gdp_growth_2020")

        fig_g3, ax_g3 = plt.subplots(figsize=(6, 3.5))
        fig_g3.patch.set_facecolor("#ffffff")
        ax_g3.set_facecolor("#f7faff")
        bar_colors = ["#e74c3c" if v < 0 else "#27ae60"
                      for v in drop_df["gdp_growth_2020"]]
        bars = ax_g3.barh(drop_df["country"], drop_df["gdp_growth_2020"],
                          color=bar_colors, edgecolor="none")
        for bar, val in zip(bars, drop_df["gdp_growth_2020"]):
            ax_g3.text(val - 0.1 if val < 0 else val + 0.1,
                       bar.get_y() + bar.get_height() / 2,
                       f"{val:.1f}%", va="center",
                       ha="right" if val < 0 else "left", fontsize=9)
        ax_g3.axvline(0, color="#555", linewidth=0.8)
        ax_g3.set_xlabel("GDP Growth Rate in 2020 (%)", fontsize=10)
        ax_g3.spines[["top", "right"]].set_visible(False)
        ax_g3.grid(axis="x", linestyle="--", alpha=0.3, color="#b0c4de")
        plt.tight_layout()
        st.pyplot(fig_g3)
        plt.close()
    else:
        st.info("GDP data not available.")

with gdp_col4:
    st.markdown(
        '<div class="section-title" style="font-size:14px;">GDP Per Capita vs Total Cases</div>',
        unsafe_allow_html=True,
    )
    scatter_df = latest[latest["gdp_per_capita"] > 0].copy()
    scatter_df["cases_per_million"] = np.where(
        scatter_df["population"] > 0,
        scatter_df["total_cases"] / scatter_df["population"] * 1_000_000, 0
    )
    if not scatter_df.empty:
        fig_g4, ax_g4 = plt.subplots(figsize=(6, 3.5))
        fig_g4.patch.set_facecolor("#ffffff")
        ax_g4.set_facecolor("#f7faff")
        ax_g4.scatter(
            scatter_df["gdp_per_capita"],
            scatter_df["cases_per_million"],
            color=palette[:len(scatter_df)],
            s=120, alpha=0.85, edgecolors="white", linewidths=0.8
        )
        for _, row in scatter_df.iterrows():
            ax_g4.annotate(
                row["location"],
                (row["gdp_per_capita"], row["cases_per_million"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
                color="#1a3a5c"
            )
        ax_g4.set_xlabel("GDP Per Capita (USD)", fontsize=10)
        ax_g4.set_ylabel("Cases Per Million", fontsize=10)
        ax_g4.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}K")
        )
        ax_g4.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K")
        )
        ax_g4.spines[["top", "right"]].set_visible(False)
        ax_g4.grid(linestyle="--", alpha=0.3, color="#b0c4de")
        plt.tight_layout()
        st.pyplot(fig_g4)
        plt.close()
    else:
        st.info("Not enough data for scatter plot.")

# ── GDP Insight box ────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#E6F1FB; border-left:4px solid #1a7ad4;
            border-radius:8px; padding:14px 18px; margin-top:10px;">
    <b style="color:#1a3a5c; font-size:14px;">💡 Key Economic Insight</b><br>
    <span style="font-size:13px; color:#1a3a5c; line-height:1.8;">
    The year <b>2020</b> saw the sharpest GDP decline across all countries since World War II.
    The UK suffered the most among selected countries with a <b>-9.3%</b> contraction.
    Interestingly, <b>higher GDP countries</b> reported more cases per million —
    likely due to better testing infrastructure — yet also recovered faster economically by 2021.
    </span>
</div>
""", unsafe_allow_html=True)


# ── Raw data table ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("View Raw Data Table"):
    display_cols = ["location", "date", "total_cases", "new_cases",
                    "total_deaths", "new_deaths", "total_vaccinations", "gdp_per_capita"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols].sort_values(
            ["location", "date"], ascending=[True, False]
        ),
        use_container_width=True,
        height=300,
    )
    csv = filtered[available_cols].to_csv(index=False)
    st.download_button("Download CSV", csv, "covid_filtered.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#5a7a9a; font-size:12px;'>"
    "Data source: Our World in Data &nbsp;•&nbsp; GDP: World Bank &nbsp;•&nbsp; "
    "Built with Python + Streamlit &nbsp;•&nbsp; Portfolio Project"
    "</p>",
    unsafe_allow_html=True,
)
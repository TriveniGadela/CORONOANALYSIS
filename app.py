import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Blue & White Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a3a5c !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8f0fe !important;
    }
    [data-testid="stSidebar"] a {
        color: #90caf9 !important;
    }

    /* KPI cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 22px;
        border: none;
        box-shadow: 0 2px 8px rgba(26,90,160,0.10);
    }
    .metric-value { font-size: 30px; font-weight: 700; }
    .metric-label { font-size: 13px; color: #5a7a9a; margin-top: 4px; }

    /* Section titles */
    .section-title {
        font-size: 16px; font-weight: 600;
        color: #1a3a5c; margin: 1.2rem 0 0.6rem;
        border-left: 4px solid #1a7ad4;
        padding-left: 10px;
    }

    /* Page title area */
    .title-bar {
        background: linear-gradient(90deg, #1a3a5c 0%, #1a7ad4 100%);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .title-bar h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }
    .title-bar p {
        color: #b3d4f5;
        font-size: 13px;
        margin: 4px 0 0;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Offline sample data ───────────────────────────────────────────────────────
def make_sample_data():
    countries = {
        "India":          (1_400_000_000, 44_000_000,  500_000,  2_200_000_000),
        "United States":  (  335_000_000,103_000_000, 1_100_000,   680_000_000),
        "Brazil":         (  215_000_000, 37_000_000,   700_000,   500_000_000),
        "United Kingdom": (   68_000_000, 24_000_000,   230_000,   160_000_000),
        "Germany":        (   84_000_000, 38_000_000,   180_000,   190_000_000),
    }
    dates = pd.date_range("2020-01-22", "2023-05-01", freq="D")
    n = len(dates)
    rows = []
    for country, (pop, peak_cases, peak_deaths, peak_vacc) in countries.items():
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
    ]
    # Try local file first (if user downloaded it manually)
    try:
        df = pd.read_csv("owid-covid-data.csv", usecols=cols, parse_dates=["date"])
        df = df[df["iso_code"].str.len() == 3].copy()
        df.fillna(0, inplace=True)
        return df, "local"
    except Exception:
        pass
    # Try internet
    try:
        df = pd.read_csv(url, usecols=cols, parse_dates=["date"])
        df = df[df["iso_code"].str.len() == 3].copy()
        df.fillna(0, inplace=True)
        return df, "live"
    except Exception:
        return make_sample_data(), "demo"


df, data_source = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌍 COVID-19 Dashboard")
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
            <h1>🌍 COVID-19 Global Analysis Dashboard</h1>
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

# Blue palette for charts
palette = sns.color_palette("Blues_d", max(len(selected_countries), 1))
palette = sns.color_palette("tab10",   max(len(selected_countries), 1))

# ── Row 1: Line chart + Bar chart ─────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        f'<div class="section-title">'
        f'{metric_choice.replace("_"," ").title()} Over Time (7-day avg)'
        f'</div>',
        unsafe_allow_html=True,
    )
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
        ax4.set_xlabel("")
        ax4.set_ylabel("")
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

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("View Raw Data Table"):
    display_cols = ["location", "date", "total_cases", "new_cases",
                    "total_deaths", "new_deaths", "total_vaccinations"]
    st.dataframe(
        filtered[display_cols].sort_values(
            ["location", "date"], ascending=[True, False]
        ),
        use_container_width=True,
        height=300,
    )
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("Download CSV", csv, "covid_filtered.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#5a7a9a; font-size:12px;'>"
    "Data source: Our World in Data &nbsp;•&nbsp; "
    "Built with Python + Streamlit &nbsp;•&nbsp; "
    "Portfolio Project"
    "</p>",
    unsafe_allow_html=True,
)
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
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 18px 20px;
        border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 28px; font-weight: 600; color: #1a1a2e; }
    .metric-label { font-size: 13px; color: #6c757d; margin-top: 2px; }
    .section-title {
        font-size: 16px; font-weight: 600;
        color: #1a1a2e; margin: 1.2rem 0 0.6rem;
        border-left: 4px solid #e74c3c;
        padding-left: 10px;
    }
    footer { visibility: hidden; }
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

        total_vacc = np.zeros(n, dtype=int)
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
                "total_cases":        int(total_cases[i]),
                "new_cases":          int(new_cases[i]),
                "total_deaths":       int(total_deaths[i]),
                "new_deaths":         int(new_deaths[i]),
                "total_vaccinations": int(total_vacc[i]),
                "people_fully_vaccinated": int(total_vacc[i] * 0.45),
            })
    return pd.DataFrame(rows)


# ── Data loading — internet first, sample data fallback ───────────────────────
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
    try:
        df = pd.read_csv(url, usecols=cols, parse_dates=["date"])
        df = df[df["iso_code"].str.len() == 3].copy()
        df.fillna(0, inplace=True)
        return df, True
    except Exception:
        return make_sample_data(), False


df, is_live = load_data()

if not is_live:
    st.warning(
        "⚠️ Could not reach the internet. "
        "The dashboard is running with **demo sample data**. "
        "Connect to Wi-Fi and press **R** to reload for real data.",
        icon="🌐",
    )

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🦠 COVID-19 Dashboard")
st.sidebar.caption("Live data ✓" if is_live else "Demo data (offline)")
st.sidebar.markdown("Data: [Our World in Data](https://ourworldindata.org/covid-cases)")

all_countries     = sorted(df["location"].unique())
default_countries = [c for c in
                     ["India","United States","Brazil","United Kingdom","Germany"]
                     if c in all_countries]
selected_countries = st.sidebar.multiselect(
    "Select Countries", all_countries, default=default_countries
)

min_date  = df["date"].min().date()
max_date  = df["date"].max().date()
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
st.sidebar.caption("Built with Streamlit · Python · Pandas · Matplotlib")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not selected_countries:
    st.warning("Please select at least one country from the sidebar.")
    st.stop()

# ── Filter data ───────────────────────────────────────────────────────────────
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

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🦠 COVID-19 Global Analysis Dashboard")
st.caption(
    f"Showing data from {start_date} to {end_date}  •  "
    f"{len(selected_countries)} countries selected  •  "
    f"{'Live data' if is_live else 'Demo data'}"
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
total_confirmed = int(latest["total_cases"].sum())
total_deaths    = int(latest["total_deaths"].sum())
total_vacc      = int(latest["total_vaccinations"].sum())
death_rate      = (total_deaths / total_confirmed * 100) if total_confirmed else 0

kpi_cols = st.columns(4)
kpis = [
    ("Total Confirmed", f"{total_confirmed:,}", "#e74c3c"),
    ("Total Deaths",    f"{total_deaths:,}",    "#c0392b"),
    ("Vaccinations",    f"{total_vacc:,}",       "#27ae60"),
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
        f'</div>',
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, country in enumerate(selected_countries):
        cdf = filtered[filtered["location"] == country].sort_values("date")
        if cdf.empty:
            continue
        rolling = cdf.set_index("date")[metric_choice].rolling(7).mean()
        ax.plot(rolling.index, rolling.values, label=country,
                color=palette[i], linewidth=2)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(metric_choice.replace("_", " ").title(), fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9, framealpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown('<div class="section-title">Total Cases by Country</div>',
                unsafe_allow_html=True)
    bar_data = latest.sort_values("total_cases", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.barh(bar_data["location"], bar_data["total_cases"],
             color=sns.color_palette("Reds_r", len(bar_data)))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6):.0f}M" if x >= 1e6 else f"{int(x):,}"
    ))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xlabel("Total Cases", fontsize=10)
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
    bars = ax3.bar(dr["location"], dr["death_rate"],
                   color=sns.color_palette("OrRd", len(dr)), edgecolor="none")
    for bar, val in zip(bars, dr["death_rate"]):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Death Rate (%)", fontsize=10)
    ax3.spines[["top", "right"]].set_visible(False)
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
    if not pivot_df.empty:
        sns.heatmap(pivot_df, ax=ax4, cmap="YlOrRd",
                    linewidths=0.3, linecolor="#f0f0f0",
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

# ── Row 3: Vaccination progress ───────────────────────────────────────────────
st.markdown('<div class="section-title">Vaccination Progress (Total Doses)</div>',
            unsafe_allow_html=True)
vacc_df = filtered[filtered["total_vaccinations"] > 0]
if not vacc_df.empty:
    fig5, ax5 = plt.subplots(figsize=(12, 3.5))
    for i, country in enumerate(selected_countries):
        cdf = vacc_df[vacc_df["location"] == country].sort_values("date")
        if cdf.empty:
            continue
        ax5.plot(cdf["date"], cdf["total_vaccinations"], label=country,
                 color=palette[i], linewidth=2)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{int(x):,}"
    ))
    ax5.set_xlabel("Date", fontsize=11)
    ax5.set_ylabel("Total Vaccinations", fontsize=11)
    ax5.legend(fontsize=9, framealpha=0.5)
    ax5.spines[["top", "right"]].set_visible(False)
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
st.caption(
    "Data source: Our World in Data · "
    "Dashboard built with Python + Streamlit · "
    "For educational & portfolio purposes"
)
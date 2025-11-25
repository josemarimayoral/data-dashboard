# app22.py
import re, string, html
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import gdown
import datetime  # para el año dinámico en el footer y los filtros de fecha

# =====================================================================
# CACHE / CONFIG STREAMLIT
# =====================================================================
try:
    cache = st.cache_data
except Exception:
        # Fallback si ejecutas sin Streamlit (por ejemplo en consola)
    def cache(**_):
        def deco(f): return f
        return deco

# =====================================================================
# GOOGLE DRIVE (CSV GRANDES)
# =====================================================================
DRIVE_IDS = {
    "Mentions.csv": "1zBDOB8DEvOlQPOpa7Pwp_QVMbVRPhdQs",
    "Replies.csv":  "1nc_6UIIdTLx4hO6sFw-hSoEPr9HigPWM",
    "Retweets.csv": "1zaBTmtcPcEPgcsWq9BE0mwJ0lgI8Ik_T",
    "Quotes.csv":   "1pumfORtb5L6bAb5XxTpCDJL5T6H1z8XW",
    "Posts.csv":    "1ULxDRVz4XjIDErfKpTJtDa9_fqgUWkdH",
}

@cache(show_spinner=True)
def ensure_csv_local(name: str) -> Path:
    """Si el CSV no existe localmente, lo descarga desde Drive."""
    p = Path(name)
    if p.exists():
        return p
    file_id = DRIVE_IDS.get(name)
    if file_id:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(p), quiet=False)
    return p

def safe_load(name: str) -> pd.DataFrame:
    """Lee CSV desde local (o Drive si hace falta)."""
    p = ensure_csv_local(name)
    return pd.read_csv(p, low_memory=False)

# =====================================================================
# ESTILO GENERAL (CLARO)
# =====================================================================
st.set_page_config(
    page_title="HRH @Alwaleed_Talal Analytics Dashboard",
    layout="wide"
)

st.markdown("""
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --soft:#eef1f7; --muted:#6b7280; --ink:#0f172a;
  --accent:#ef4444; --accent2:#0ea5e9; --rim:#e5e7eb;
}
.block-container{max-width:1400px !important;}
.main, .block-container{background:var(--bg); color:var(--ink);}
[data-testid="stSidebar"]{
  background:var(--panel);
  border-right:1px solid var(--rim);
  color:var(--ink);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span{
  color:var(--ink) !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea{
  background:#ffffff !important;
  color:var(--ink) !important;
  border-radius:8px !important;
  border:1px solid var(--rim) !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton>button {
    background: var(--panel) !important;
    color: var(--ink) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton>button:hover {
    background: var(--soft) !important;
    color: var(--accent2) !important;
}
[data-testid="stSidebar"] .stCheckbox label{
  color:var(--ink) !important;
}
[data-testid="stTabs"] button{
  color:var(--muted);
  background:transparent;
  border:0;
  padding-bottom:0.4rem;
  font-weight:600;
}
[data-testid="stTabs"] button[aria-selected="true"]{
  color:var(--accent);
  border-bottom:2px solid var(--accent);
}
[data-testid="stTabs"] button:hover{
  color:var(--accent2);
  background:var(--soft);
}
.kpi{
  background:var(--panel); border:1px solid var(--rim); border-radius:16px;
  padding:18px; min-height:110px;
  display:flex; flex-direction:column; justify-content:center;
}
.kpi .label{font-size:.85rem; color:var(--muted);}
.kpi .value{font-size:1.6rem; font-weight:800;}
hr{border-top:1px solid var(--rim);}
</style>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_white"

# 🎨 Paleta fija para tipos de interacción
COLOR_MAP = {
    "Posts": "#1f77b4",     # azul
    "Replies": "#2ca02c",   # verde
    "Mentions": "#9467bd",  # morado
    "Retweets": "#d62728",  # rojo
    "Quotes": "#7f7f7f",    # gris neutro
}

# =====================================================================
# UTILIDADES CSV / FECHAS
# =====================================================================
def parse_datetime_any(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.isna().mean() > 0.8:
        as_num = pd.to_numeric(series, errors="coerce")
        try_ms = pd.to_datetime(as_num, unit="ms", utc=True, errors="coerce")
        if try_ms.notna().sum() > s.notna().sum():
            s = try_ms
        else:
            try_s = pd.to_datetime(as_num, unit="s", utc=True, errors="coerce")
            if try_s.notna().sum() > s.notna().sum():
                s = try_s
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        pass
    return s

TWITTER_EPOCH_MS = 1288834974657
def snowflake_to_datetime(snowflake_series: pd.Series) -> pd.Series:
    ids = pd.to_numeric(snowflake_series, errors="coerce").astype("Int64")
    ms = (ids >> 22) + TWITTER_EPOCH_MS
    dt = pd.to_datetime(ms, unit="ms", errors="coerce")
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt

def extract_id_from_link(link_series: pd.Series) -> pd.Series:
    s = link_series.astype(str).str.extract(r"/status/(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["__type"] = label
    low = {c: c.lower() for c in out.columns}

    exact = {
        "time","date","created_at","created at","tweet time","tweet_time",
        "timestamp","tweet timestamp","tweet_timestamp","tweet_created_at",
        "tweet_date","date_utc","time_utc","createdat","createdat"
    }
    tcol = next((c for c in out.columns if low[c] in exact), None)
    if tcol is None:
        tcol = next((c for c in out.columns
                     if re.search(r"(date|time|created|timestamp)", c, re.I)), None)
    out["__time"] = parse_datetime_any(out[tcol]) if tcol else pd.NaT

    ucol = next((c for c in out.columns
                 if re.search(r"(user|username|screen|author|from|profile|handle)", c, re.I)), None)
    xcol = next((c for c in out.columns
                 if re.search(r"(text|tweet|content|full_text|body)", c, re.I)), None)
    lcol = next((c for c in out.columns
                 if re.search(r"(link|url|permalink)", c, re.I)), None)

    out["__user"] = out[ucol] if ucol else None
    out["__text"] = out[xcol] if xcol else None
    out["__link"] = out[lcol] if lcol else None

    idcol = next((c for c in out.columns
                  if re.search(r"(tweet[_-]?id|status[_-]?id|id_str|id$)", c, re.I)), None)
    out["__id"] = out[idcol] if idcol else None
    return out

def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "__time" not in df.columns:
        df["__time"] = pd.NaT
        return df
    t = df["__time"]
    needs_fix = t.isna().all()
    if not needs_fix:
        unique_non_na = t.dropna().unique()
        needs_fix = len(unique_non_na) <= 1
    if needs_fix:
        if "__id" in df.columns and df["__id"].notna().any():
            dt = snowflake_to_datetime(df["__id"])
            if dt.notna().any():
                df = df.copy(); df["__time"] = dt; return df
        link_col = next((c for c in df.columns
                         if re.search(r"(link|url|permalink)", c, re.I)), None)
        if link_col:
            ids = extract_id_from_link(df[link_col])
            dt = snowflake_to_datetime(ids)
            if dt.notna().any():
                df = df.copy(); df["__time"] = dt; return df
    return df

def daily_counts(df, label):
    if df.empty or df["__time"].dropna().empty:
        return pd.DataFrame(columns=["date", "count", "type"])
    d = df.dropna(subset=["__time"]).copy()
    d["date"] = d["__time"].dt.date
    d = d.groupby("date").size().reset_index(name="count")
    d["type"] = label
    return d

def to_daily_series(df):
    if df.empty or df["__time"].dropna().empty:
        return pd.Series(dtype=float)
    d = df.dropna(subset=["__time"]).copy()
    d["date"] = d["__time"].dt.date
    s = d.groupby("date").size()
    s.index = pd.to_datetime(s.index)
    return s

# =====================================================================
# USUARIO / TEXTO / LINKS
# =====================================================================
def display_user_series(df: pd.DataFrame) -> pd.Series:
    preferred = [
        "alias",
        "userName",
        "username",
        "screen_name",
        "screenName",
        "userScreenName",
        "author_username",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in preferred:
        lc = cand.lower()
        if lc in cols_lower:
            return df[cols_lower[lc]].astype(str)
    if "__user" in df.columns:
        return df["__user"].astype(str)
    return pd.Series([""] * len(df), index=df.index)

def clean_link(value):
    if isinstance(value, list):
        return value[0] if value else ""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            arr = eval(value)
            return arr[0] if arr else ""
        except Exception:
            return ""
    return value

def sanitize_tweet_text(text: str) -> str:
    """Limpia HTML suelto tipo <div> que venga en el CSV (normal y escapado)."""
    if not isinstance(text, str):
        return ""
    # Tags <div> reales
    text = re.sub(r"</?div[^>]*>", " ", text, flags=re.IGNORECASE)
    # Versiones escapadas &lt;div&gt; / &lt;/div&gt;
    text = re.sub(r"&lt;/?div[^&]*&gt;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================================
# KPI / DELTAS
# =====================================================================
def kpi_card(title, value, delta=None, invert=False):
    if delta is None:
        delta_html = ""
    else:
        is_up = (delta >= 0)
        if invert:
            is_up = not is_up
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="delta {"up" if is_up else "down"}">{sign}{delta:.1f}%</div>'
    st.markdown(f"""
      <div class="kpi">
        <div class="label">{title}</div>
        <div class="value">{value}</div>
        {delta_html}
      </div>
    """, unsafe_allow_html=True)

def prev_period_delta(series: pd.Series, start_date, end_date):
    if series.empty:
        return None
    mask_curr = ((series.index >= pd.Timestamp(start_date)) &
                 (series.index <= pd.Timestamp(end_date)))
    curr = series.loc[mask_curr].sum()
    period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    prev_start = pd.Timestamp(start_date) - pd.Timedelta(days=period_days)
    prev_end   = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    mask_prev = ((series.index >= prev_start) &
                 (series.index <= prev_end))
    prev = series.loc[mask_prev].sum()
    if prev == 0:
        return None
    return ((curr - prev) / prev) * 100

# =====================================================================
# VISUALIZACIONES BASE
# =====================================================================
def heatmap_hour_week(df, title):
    if df.empty or df["__time"].dropna().empty:
        return None
    d = df.dropna(subset=["__time"]).copy()
    d["weekday"] = d["__time"].dt.day_name()
    d["hour"]    = d["__time"].dt.hour
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d["weekday"] = pd.Categorical(d["weekday"], categories=order, ordered=True)
    fig = px.density_heatmap(
        d, x="hour", y="weekday",
        nbinsx=24,
        color_continuous_scale="Blues",
        title=title,
    )
    return fig

def barh_sorted(df, x, y, title):
    if df is None or df.empty:
        return None
    df2 = df.copy()
    if df2.columns.duplicated().any():
        df2 = df2.loc[:, ~df2.columns.duplicated()]
    if x not in df2.columns or y not in df2.columns:
        return None
    df2 = df2.sort_values(x, ascending=False)
    fig = px.bar(df2, x=x, y=y, orientation="h", title=title)
    fig.update_yaxes(autorange="reversed")
    return fig

def top_users(df, n=20):
    if df.empty:
        return pd.DataFrame(columns=["user", "interactions"])
    tmp = df.copy()
    tmp["__display_user"] = display_user_series(tmp)
    out = (
        tmp.groupby("__display_user", dropna=False)
           .size()
           .reset_index(name="interactions")
           .sort_values("interactions", ascending=False)
           .head(n)
           .rename(columns={"__display_user": "user"})
    )
    return out

# =====================================================================
# TOP INFLUENCERS
# =====================================================================
def compute_influencers(df, n=15):
    if df.empty:
        return pd.DataFrame(columns=["user","interactions","followers"])
    tmp = df.copy()
    tmp["__display_user"] = display_user_series(tmp)

    follower_candidates = [
        "followers_count","userFollowers","followers","follower_count",
        "user_followers","audience_count"
    ]
    follow_col = None
    cols_lower = {c.lower(): c for c in tmp.columns}
    for cand in follower_candidates:
        lc = cand.lower()
        if lc in cols_lower:
            follow_col = cols_lower[lc]
            break

    grp = tmp.groupby("__display_user", dropna=False)
    out = grp.size().reset_index(name="interactions")
    if follow_col:
        out["followers"] = grp[follow_col].max()
        out["followers"] = pd.to_numeric(out["followers"], errors="coerce")
        out = out.sort_values(["followers","interactions"], ascending=[False,False])
    else:
        out["followers"] = pd.NA
        out = out.sort_values("interactions", ascending=False)

    out = out.rename(columns={"__display_user":"user"}).head(n)
    return out

# =====================================================================
# TOP TWEETS POR ENGAGEMENT
# =====================================================================
def best_numeric_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in cols_lower:
            col = cols_lower[lc]
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                return col
            except Exception:
                continue
    return None

def top_tweets(df, n=5):
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    likes_col = best_numeric_col(d, ["favorite_count","like_count","likes","favourites"])
    rt_col    = best_numeric_col(d, ["retweet_count","rt_count","retweets"])
    rep_col   = best_numeric_col(d, ["reply_count","replies"])
    qt_col    = best_numeric_col(d, ["quote_count","quotes"])

    d["engagement"] = 0
    for col in [likes_col, rt_col, rep_col, qt_col]:
        if col:
            d["engagement"] += d[col].fillna(0)

    if d["engagement"].sum() == 0:
        return pd.DataFrame()

    d["time"] = parse_datetime_any(d["__time"])
    d["user"] = display_user_series(d)
    d["link"] = d["__link"].apply(clean_link)
    d["text"] = d["__text"].apply(sanitize_tweet_text)
    d = d.sort_values("engagement", ascending=False)
    return d[["time","user","text","link","engagement"]].head(n)

def render_top_tweets(df, title, n=5):
    st.markdown(f"### {title}")
    tt = top_tweets(df, n=n)
    if tt.empty:
        st.info("Not enough engagement data to rank tweets.")
        return
    for _, row in tt.iterrows():
        time_str = row["time"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row["time"]) else ""
        user = row["user"] or ""
        text = row["text"] or ""
        text_html = html.escape(text).replace("\n","<br>")
        link = row["link"] if isinstance(row["link"], str) else ""
        engagement = int(row["engagement"]) if pd.notna(row["engagement"]) else 0

        link_html = (
            f'<a href="{link}" target="_blank" style="text-decoration:none; font-weight:500;">🔗 View tweet</a>'
            if isinstance(link, str) and link.startswith("http") else ""
        )

        st.markdown(
            f"""
            <div style="
                background: var(--panel, #ffffff);
                border: 1px solid var(--rim, #e5e7eb);
                border-radius: 16px;
                padding: 12px 16px;
                margin-bottom: 10px;
                box-shadow: 0 2px 6px rgba(15,23,42,0.06);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <div style="font-size: 0.9rem; color: var(--muted, #6b7280);">
                        <strong>@{user}</strong>
                        <span style="opacity:.7;"> • {time_str}</span>
                    </div>
                    <div style="font-size:0.85rem; color:var(--muted,#6b7280);">
                        ⭐ {engagement}
                    </div>
                </div>
                <div style="font-size: 0.96rem; margin-bottom: 8px;" dir="auto">
                    {text_html}
                </div>
                <div style="font-size: 0.85rem;">
                    {link_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
# MAPA MUNDO AUDIENCIA
# =====================================================================
def audience_country_counts(df, loc_col):
    if df.empty or not loc_col:
        return pd.DataFrame(columns=["country","count"])
    d = df.copy()
    d["__loc"] = d[loc_col].astype(str).str.strip()
    counts = (
        d["__loc"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("country")
        .reset_index(name="count")
    )
    return counts

def render_world_map(rm, loc_col):
    st.markdown("### Audience geo reach (Replies + Mentions)")
    if not loc_col or rm.empty:
        st.info("No location data available.")
        return
    counts = audience_country_counts(rm, loc_col)
    if counts.empty:
        st.info("Locations do not look like country names (map would be empty).")
        return

    top = counts.head(40)
    q90 = top["count"].quantile(0.90)
    if pd.isna(q90) or q90 <= 0:
        q90 = top["count"].max()

    fig = px.choropleth(
        top,
        locations="country",
        locationmode="country names",
        color="count",
        hover_name="country",
        color_continuous_scale="Blues",
        range_color=[0, q90],
        title="Where the audience is (top locations)",
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# SIDEBAR: DATA + DATE FILTER
# =====================================================================
#st.sidebar.markdown("### Data")
#st.sidebar.caption("CSVs: Posts, Retweets, Replies, Mentions, Quotes\n\n(All data is loaded automatically)")

st.sidebar.subheader("Date Range Filter")
disable_filter = st.sidebar.checkbox(
    "Uncheck to filter by dates since January 2025", value=True
)

# 🔒 Límite inferior: 1 Jan 2025
min_date = datetime.date(2025, 1, 1)
today = datetime.date.today()

# Valores por defecto (últimos 7 días, pero nunca antes de min_date)
default_end = today if today >= min_date else min_date
default_start = default_end - datetime.timedelta(days=7)
if default_start < min_date:
    default_start = min_date

# Si el filtro está desactivado, usamos los defaults sin mostrar inputs
start_date = default_start
end_date = default_end

if not disable_filter:
    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start,
        min_value=min_date,
        max_value=default_end,
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=default_end,
        min_value=min_date,
        max_value=today,
    )

def apply_filter(df):
    if df.empty:
        return df
    df = df.copy()
    df["__time"] = parse_datetime_any(df["__time"])
    if disable_filter:
        # Aunque ignoremos el selector, seguimos cortando a partir de 1 Jan 2025
        return df[df["__time"] >= pd.Timestamp(min_date)]
    s = pd.Timestamp(start_date).tz_localize(None)
    e = pd.Timestamp(end_date).tz_localize(None)
    return df[(df["__time"] >= s) & (df["__time"] <= e)]

# =====================================================================
# DATA LOADING (SIEMPRE TODO)
# =====================================================================
raw_posts   = safe_load("Posts.csv")
raw_rts     = safe_load("Retweets.csv")
raw_quotes  = safe_load("Quotes.csv")
raw_replies = safe_load("Replies.csv")
raw_mentions = safe_load("Mentions.csv")

norm_posts    = ensure_valid_time(normalize(raw_posts,   "Posts"))
norm_rts      = ensure_valid_time(normalize(raw_rts,     "Retweets"))
norm_replies  = ensure_valid_time(normalize(raw_replies, "Replies"))
norm_mentions = ensure_valid_time(normalize(raw_mentions,"Mentions"))
norm_quotes   = ensure_valid_time(normalize(raw_quotes,  "Quotes"))

posts    = apply_filter(norm_posts)
rts      = apply_filter(norm_rts)
replies  = apply_filter(norm_replies)
mentions = apply_filter(norm_mentions)
quotes   = apply_filter(norm_quotes)

# =====================================================================
# HEADER
# =====================================================================
BASE = Path(__file__).parent
col_logo, col_title, col_avatar = st.columns([1, 6, 1])
with col_logo:
    logo_path = BASE / "Audiense-Logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=80)
with col_title:
    st.title("HRH Data Dashboard")
with col_avatar:
    avatar_path = BASE / "HRH avatar.jpg"
    if avatar_path.exists():
        st.image(str(avatar_path), width=64)

# =====================================================================
# TABS
# =====================================================================
tabs = st.tabs(
    ["Summary", "Posts", "Retweets", "Replies", "Mentions", "Quotes", "Insights"]
)

# ---------------------------------------------------------------------
# TAB: SUMMARY
# ---------------------------------------------------------------------
with tabs[0]:
    s_posts    = to_daily_series(posts)
    s_rts      = to_daily_series(rts)
    s_replies  = to_daily_series(replies)
    s_mentions = to_daily_series(mentions)
    s_quotes   = to_daily_series(quotes)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Posts", len(posts), prev_period_delta(s_posts, start_date, end_date))
    with c2:
        kpi_card("Retweets", len(rts), prev_period_delta(s_rts, start_date, end_date))
    with c3:
        kpi_card("Replies", len(replies), prev_period_delta(s_replies, start_date, end_date))
    with c4:
        kpi_card("Mentions", len(mentions), prev_period_delta(s_mentions, start_date, end_date))
    with c5:
        kpi_card("Quotes", len(quotes), prev_period_delta(s_quotes, start_date, end_date))

    frames = []
    for name, df in [
        ("Posts", posts), ("Retweets", rts), ("Replies", replies),
        ("Mentions", mentions), ("Quotes", quotes)
    ]:
        if not df.empty:
            frames.append(daily_counts(df, name))

    if frames:
        trend = pd.concat(frames, ignore_index=True)
        # Quotes en la base del stack
        cat_order = ["Quotes", "Posts", "Retweets", "Replies", "Mentions"]
        fig = px.area(
            trend,
            x="date",
            y="count",
            color="type",
            title="Activity over time (stacked)",
            color_discrete_map=COLOR_MAP,
            category_orders={"type": cat_order},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the current selection.")

    dd = {
        "Retweets": len(rts),
        "Replies": len(replies),
        "Mentions": len(mentions),
        "Quotes": len(quotes)
    }
    pie_df = pd.DataFrame({"type": dd.keys(), "count": dd.values()})
    pie = px.pie(
        pie_df,
        names="type",
        values="count",
        hole=0.55,
        title="Engagement breakdown",
        color="type",
        color_discrete_map=COLOR_MAP,
    )
    st.plotly_chart(pie, use_container_width=True)

    all_df = pd.concat(
        [posts, rts, replies, mentions, quotes],
        ignore_index=True
    )
    hm = heatmap_hour_week(all_df, "When activity happens (hour × weekday)")
    if hm:
        st.plotly_chart(hm, use_container_width=True)

# ---------------------------------------------------------------------
# TAB GENÉRICO POR SECCIÓN
# ---------------------------------------------------------------------
def section_tab(df, label):
    st.subheader(label)
    if df.empty:
        st.info(f"No {label.lower()} for the current selection.")
        return

    st.plotly_chart(
        px.area(
            daily_counts(df, label),
            x="date",
            y="count",
            title=f"{label} by day"
        ),
        use_container_width=True,
    )

    hm = heatmap_hour_week(df, f"{label} timing (hour × weekday)")
    if hm:
        st.plotly_chart(hm, use_container_width=True)

    if label in ("Replies", "Mentions", "Retweets"):
        st.markdown("**Top users (by interactions)**")
        tu = top_users(df, n=15)
        fig_tu = barh_sorted(tu, x="interactions", y="user", title="Most active users")
        if fig_tu:
            st.plotly_chart(fig_tu, use_container_width=True)

with tabs[1]:
    section_tab(posts, "Posts")
with tabs[2]:
    section_tab(rts, "Retweets")
with tabs[3]:
    section_tab(replies, "Replies")
with tabs[4]:
    section_tab(mentions, "Mentions")
with tabs[5]:
    section_tab(quotes, "Quotes")

# ---------------------------------------------------------------------
# TAB: INSIGHTS
# ---------------------------------------------------------------------
with tabs[6]:
    st.subheader("Quick insights (Replies + Mentions)")

    rm = pd.concat([replies, mentions], ignore_index=True)

    if not rm.empty and "__text" in rm.columns:
        URL_RE     = re.compile(r"https?://\S+")
        MENTION_RE = re.compile(r"@\w+")
        HASHTAG_RE = re.compile(r"#\w+")

        def tok(s):
            if not isinstance(s, str):
                return []
            s = URL_RE.sub(" ", s)
            s = MENTION_RE.sub(" ", s)
            s = s.lower().translate(str.maketrans("", "", string.punctuation))
            return [w for w in s.split() if w and not w.isdigit()]

        tags, terms = [], []
        for t in rm["__text"].dropna().tolist():
            tags.extend([h.lower() for h in HASHTAG_RE.findall(t)])
            terms.extend(tok(t))

        col1, col2 = st.columns(2)
        if tags:
            vc = (pd.Series(tags).value_counts().head(20)
                  .reset_index().rename(columns={"index":"hashtag",0:"count"}))
            fig_hash = barh_sorted(vc, x="count", y="hashtag", title="Top hashtags")
            with col1:
                if fig_hash:
                    st.plotly_chart(fig_hash, use_container_width=True)

        if terms:
            vc2 = (pd.Series(terms).value_counts().head(20)
                   .reset_index().rename(columns={"index":"term",0:"count"}))
            fig_terms = barh_sorted(vc2, x="count", y="term", title="Top terms")
            with col2:
                if fig_terms:
                    st.plotly_chart(fig_terms, use_container_width=True)

    else:
        st.info("Add Replies/Mentions to see topics & hashtags.")

    st.markdown("---")
    st.subheader("Top influencers (Replies + Mentions + Retweets)")
    infl_df = compute_influencers(
        pd.concat([replies, mentions, rts], ignore_index=True),
        n=15
    )
    if infl_df.empty:
        st.info("Not enough data to compute influencers.")
    else:
        fig_infl = barh_sorted(
            infl_df.rename(columns={"user":"account"}),
            x="interactions",
            y="account",
            title="Most active accounts interacting with HRH"
        )
        if fig_infl:
            st.plotly_chart(fig_infl, use_container_width=True)
        st.dataframe(infl_df, use_container_width=True, height=360)

    st.markdown("---")
    st.subheader("Top tweets by engagement")
    col_a, col_b = st.columns(2)
    with col_a:
        render_top_tweets(posts, "Top original posts", n=5)
    with col_b:
        render_top_tweets(rm, "Top replies & mentions", n=5)

    st.markdown("---")
    st.subheader("Audience profile (Replies + Mentions)")

    if not rm.empty:
        lang_col = None
        for cand in ["lang", "language", "tweet_lang", "userLang"]:
            if cand in rm.columns:
                lang_col = cand
                break

        loc_col = None
        for cand in ["userLocation", "location", "place", "user_location"]:
            if cand in rm.columns:
                loc_col = cand
                break

        c1, c2 = st.columns(2)

        if lang_col:
            lang_counts = (
                rm[lang_col].fillna("unknown").astype(str)
                .value_counts()
                .rename_axis("language")
                .reset_index(name="count")
            )
            lang_counts = lang_counts[
                lang_counts["language"].str.lower() != "unknown"
            ].head(20)
            fig_lang = barh_sorted(
                lang_counts, x="count", y="language", title="Top languages"
            )
            with c1:
                if fig_lang:
                    st.plotly_chart(fig_lang, use_container_width=True)

        if loc_col:
            loc_counts = (
                rm[loc_col].fillna("unknown").astype(str)
                .value_counts()
                .rename_axis("location")
                .reset_index(name="count")
            )
            loc_counts = loc_counts[
                loc_counts["location"].str.lower() != "unknown"
            ].head(20)
            fig_loc = barh_sorted(
                loc_counts, x="count", y="location", title="Top locations"
            )
            with c2:
                if fig_loc:
                    st.plotly_chart(fig_loc, use_container_width=True)

        render_world_map(rm, loc_col)

    st.caption(
        "Tips: use the date filter on the sidebar or disable it to see the full history (from Jan 2025)."
    )

# =====================================================================
# FOOTER
# =====================================================================
year = datetime.datetime.now().year
st.markdown(
    "<hr style='margin-top:40px; opacity:0.25;'>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div style="text-align:center; color:#6b7280; margin-top:10px; font-size:0.85rem;">
        ✨ © {year} <a href="https://www.audiense.com/" target="_blank" style="color:#6b7280; text-decoration:none;">
        Audiense</a> — Designed and developed by the X Marketing Team
    </div>
    """,
    unsafe_allow_html=True,
)

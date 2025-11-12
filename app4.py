# app.py
import re, string
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# Si usas Streamlit:
try:
    import streamlit as st
    cache = st.cache_data
except Exception:
    # fallback si ejecutas sin Streamlit
    def cache(**_): 
        def deco(f): return f
        return deco

import gdown

# IDs de tus archivos en Google Drive
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
    file_id = DRIVE_IDS[name]
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(p), quiet=False)
    return p

@cache(show_spinner=True)
def load_csv(name: str) -> pd.DataFrame:
    p = ensure_csv_local(name)
    return pd.read_csv(p)

# --- Arranque ligero en Streamlit Cloud ---
st.sidebar.markdown("### Data")
load_big = st.sidebar.button("🔄 Load full data")

def safe_load(name):
    p = ensure_csv_local(name)
    return pd.read_csv(p, low_memory=False)

# Carga primero los pequeños (retweets/quotes/posts)
retweets  = safe_load("Retweets.csv")
quotes    = safe_load("Quotes.csv")
posts     = safe_load("Posts.csv")

# Los grandes solo si el usuario los pide
if load_big:
    mentions = safe_load("Mentions.csv")
    replies  = safe_load("Replies.csv")
else:
    mentions = None
    replies  = None
    st.info("App started in light mode. To load the complete data including Mentions and Replies, click the button on the sidebar.")
# --- fin parche ---

# ------------------------------------------------------

# ------------------------------------------------------------------------------
# CONFIG (tema claro y legible)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="HRH @Alwaleed_Talal Analytics Dashboard", layout="wide")

st.markdown("""
<style>
:root { --bg:#f8fafc; --card:#ffffff; --text:#0f172a; --muted:#475569; --border:#e2e8f0; }
.main, .block-container { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--border); }
h1,h2,h3 { color: var(--text); }
.kpi {
  background: var(--card);
  border:1px solid var(--border);
  border-radius:14px; padding:16px; margin-bottom:12px;
  box-shadow:0 1px 2px rgba(15,23,42,.06);
}
.kpi .title { font-size:.85rem; color:var(--muted); margin-bottom:6px;}
.kpi .value { font-size:1.6rem; font-weight:700;}
.kpi .delta.up { color:#18a957; font-weight:600;}
.kpi .delta.down { color:#e11d48; font-weight:600;}
hr{border:none;border-top:1px solid var(--border);margin:12px 0;}
</style>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_white"

# ------------------------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------------------------
def read_csv_smart(path_or_buffer):
    """Lee CSV con distintos separadores/encodings."""
    tried = []
    for enc in ("utf-8","utf-8-sig","latin1"):
        for sep in (",",";","\t","|"):
            try:
                df = pd.read_csv(path_or_buffer, encoding=enc, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception as e:
                tried.append(f"{enc}/{sep}: {e}")
                try:
                    # si es buffer, volver al inicio
                    path_or_buffer.seek(0)
                except Exception:
                    pass
    return pd.DataFrame()

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
    try: s = s.dt.tz_convert(None)
    except Exception: pass
    return s

TWITTER_EPOCH_MS = 1288834974657
def snowflake_to_datetime(snowflake_series: pd.Series) -> pd.Series:
    ids = pd.to_numeric(snowflake_series, errors="coerce").astype("Int64")
    ms = (ids >> 22) + TWITTER_EPOCH_MS
    dt = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    try: dt = dt.dt.tz_convert(None)
    except Exception: pass
    return dt

def extract_id_from_link(link_series: pd.Series) -> pd.Series:
    s = link_series.astype(str).str.extract(r"/status/(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    out["__type"] = label
    low = {c: c.lower() for c in out.columns}

    # fecha
    exact = {"time","date","created_at","created at","tweet time","tweet_time",
             "timestamp","tweet timestamp","tweet_timestamp","tweet_created_at",
             "tweet_date","date_utc","time_utc","createdat","createdat"}
    tcol = next((c for c in out.columns if low[c] in exact), None)
    if tcol is None:
        tcol = next((c for c in out.columns if re.search(r"(date|time|created|timestamp)", c, re.I)), None)
    out["__time"] = parse_datetime_any(out[tcol]) if tcol else pd.NaT

    # user / text / link
    ucol = next((c for c in out.columns if re.search(r"(user|username|screen|author|from|profile|handle)", c, re.I)), None)
    xcol = next((c for c in out.columns if re.search(r"(text|tweet|content|full_text|body)", c, re.I)), None)
    lcol = next((c for c in out.columns if re.search(r"(link|url|permalink)", c, re.I)), None)
    out["__user"] = out[ucol] if ucol else None
    out["__text"] = out[xcol] if xcol else None
    out["__link"] = out[lcol] if lcol else None

    # tweet id (si existe)
    idcol = next((c for c in out.columns if re.search(r"(tweet[_-]?id|status[_-]?id|id_str|id$)", c, re.I)), None)
    out["__id"] = out[idcol] if idcol else None
    return out

def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "__time" not in df.columns: df["__time"] = pd.NaT; return df
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
        link_col = next((c for c in df.columns if re.search(r"(link|url|permalink)", c, re.I)), None)
        if link_col:
            ids = extract_id_from_link(df[link_col])
            dt = snowflake_to_datetime(ids)
            if dt.notna().any():
                df = df.copy(); df["__time"] = dt; return df
    return df

def daily_counts(df, label):
    if df.empty or df["__time"].dropna().empty:
        return pd.DataFrame(columns=["date","count","type"])
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

def top_users(df, n=20):
    if df.empty: return pd.DataFrame(columns=["user","interactions"])
    return (df.groupby("__user", dropna=False).size()
              .reset_index(name="interactions")
              .sort_values("interactions", ascending=False)
              .head(n)
              .rename(columns={"__user":"user"}))

def recent_table(df, n=60):
    if df.empty: return pd.DataFrame(columns=["time","user","text","link"])
    cols = ["__time","__user","__text","__link"]
    show = df[cols].rename(columns={"__time":"time","__user":"user","__text":"text","__link":"link"})
    return show.sort_values("time", ascending=False).head(n)

def kpi_card(title, value, delta=None, invert=False):
    if delta is None:
        delta_html = ""
    else:
        is_up = (delta >= 0)
        if invert: is_up = not is_up
        cls = "up" if is_up else "down"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="delta {cls}">{sign}{delta:.1f}%</div>'
    st.markdown(f"""
      <div class="kpi">
        <div class="title">{title}</div>
        <div class="value">{value}</div>
        {delta_html}
      </div>
    """, unsafe_allow_html=True)

def prev_period_delta(series: pd.Series, start_date, end_date):
    if series.empty: return None
    mask_curr = (series.index >= pd.Timestamp(start_date)) & (series.index <= pd.Timestamp(end_date))
    curr = series.loc[mask_curr].sum()
    period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    prev_start = pd.Timestamp(start_date) - pd.Timedelta(days=period_days)
    prev_end   = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    mask_prev = (series.index >= prev_start) & (series.index <= prev_end)
    prev = series.loc[mask_prev].sum()
    if prev == 0: return None
    return ((curr - prev) / prev) * 100

# ----------------------------------------------------------------
# DATA LOADING: always use Google Drive (no local option)
# ----------------------------------------------------------------
BASE = Path(__file__).parent  # folder where app.py is

def load_csvs():
    # Force cloud mode by default
    use_local = False

    if use_local:
        files = {
            "Posts": BASE / "Posts.csv",
            "Retweets": BASE / "Retweets.csv",
            "Replies": BASE / "Replies.csv",
            "Mentions": BASE / "Mentions.csv",
            "Quotes": BASE / "Quotes.csv",
        }
        dfs = {k: read_csv_smart(v) for k, v in files.items()}
    else:
        dfs = {
            "Posts": read_csv_smart("Posts.csv"),
            "Retweets": read_csv_smart("Retweets.csv"),
            "Replies": read_csv_smart("Replies.csv"),
            "Mentions": read_csv_smart("Mentions.csv"),
            "Quotes": read_csv_smart("Quotes.csv"),
        }

    return dfs


# Load the data (from Drive by default)
dfs_raw = load_csvs()

# --- DEBUG UI (opcional) ---
DEBUG = False  # cambia a True si quieres ver diagnósticos en el sidebar

if DEBUG:
    # Diagnóstico rápido (antes de mapear)
    st.sidebar.subheader("Diagnostics (before date filter)")
    diag = pd.DataFrame({
        "section": list(dfs_raw.keys()),
        "rows": [len(v) for v in dfs_raw.values()],
        "cols": [v.shape[1] if not v.empty else 0 for v in dfs_raw.values()],
    })
    st.sidebar.dataframe(diag, use_container_width=True, height=220)
# --- fin DEBUG UI ---

# ===== Mapear/normalizar por sección =====
def normalize_all():
    out = {}
    for label, df in dfs_raw.items():
        n = normalize(df, label)

        # Mapeo manual si faltan columnas (solo visible en modo DEBUG)
        if DEBUG:
            with st.sidebar.expander(f"Column mapping • {label}", expanded=False):
                # time
                if n["__time"].isna().all():
                    dt_col = st.selectbox("Datetime column", options=list(df.columns),
                                          index=None, key=f"dt_{label}")
                    if dt_col:
                        n["__time"] = parse_datetime_any(df[dt_col])

                # user
                if "__user" in n and (n["__user"].isna().all() if "__user" in n else True):
                    u_col = st.selectbox("User column",
                                         options=["<skip>"] + list(df.columns),
                                         index=0, key=f"user_{label}")
                    if u_col and u_col != "<skip>":
                        n["__user"] = df[u_col].astype(str)

                # text
                if "__text" in n and (n["__text"].isna().all() if "__text" in n else True):
                    x_col = st.selectbox("Text column",
                                         options=["<skip>"] + list(df.columns),
                                         index=0, key=f"text_{label}")
                    if x_col and x_col != "<skip>":
                        n["__text"] = df[x_col].astype(str)

                # link
                if "__link" in n and (n["__link"].isna().all() if "__link" in n else True):
                    l_col = st.selectbox("Link column (optional)",
                                         options=["<skip>"] + list(df.columns),
                                         index=0, key=f"link_{label}")
                    if l_col and l_col != "<skip>":
                        n["__link"] = df[l_col].astype(str)

        # siempre validar y guardar, con independencia de DEBUG
        n = ensure_valid_time(n)
        out[label] = n

    return out

norm = normalize_all()


# Filtro por fechas
st.sidebar.subheader("Date Range Filter")
disable_filter = st.sidebar.checkbox("Ignore date range (uncheck to filter by dates)", value=True)
default_end = pd.Timestamp.today().normalize()
default_start = default_end - pd.Timedelta(days=7)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date", value=default_end)

def apply_filter(df):
    if df.empty: return df
    df = df.copy()
    df["__time"] = parse_datetime_any(df["__time"])
    if disable_filter: return df
    s = pd.Timestamp(start_date).tz_localize(None)
    e = pd.Timestamp(end_date).tz_localize(None)
    return df[(df["__time"] >= s) & (df["__time"] <= e)]

posts   = apply_filter(norm["Posts"])
rts     = apply_filter(norm["Retweets"])
replies = apply_filter(norm["Replies"])
mentions= apply_filter(norm["Mentions"])
quotes  = apply_filter(norm["Quotes"])

# ------------------------------------------------------------------------------
# ENCABEZADO
# ------------------------------------------------------------------------------
col_logo, col_title, col_avatar = st.columns([1,6,1])
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

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tabs = st.tabs(["Summary","Posts","Retweets","Replies","Mentions","Quotes","Insights"])

# ===== Summary =====
with tabs[0]:
    s_posts    = to_daily_series(posts)
    s_rts      = to_daily_series(rts)
    s_replies  = to_daily_series(replies)
    s_mentions = to_daily_series(mentions)
    s_quotes   = to_daily_series(quotes)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("Posts", len(posts), prev_period_delta(s_posts, start_date, end_date))
    with c2: kpi_card("Retweets", len(rts), prev_period_delta(s_rts, start_date, end_date))
    with c3: kpi_card("Replies", len(replies), prev_period_delta(s_replies, start_date, end_date))
    with c4: kpi_card("Mentions", len(mentions), prev_period_delta(s_mentions, start_date, end_date))
    with c5: kpi_card("Quotes", len(quotes), prev_period_delta(s_quotes, start_date, end_date))

    frames=[]
    for name,df in [("Posts",posts),("Retweets",rts),("Replies",replies),("Mentions",mentions),("Quotes",quotes)]:
        if not df.empty: frames.append(daily_counts(df,name))
    if frames:
        trend = pd.concat(frames, ignore_index=True)
        fig = px.area(trend, x="date", y="count", color="type", title="Activity over time (stacked)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the current selection.")

    dd = {"Retweets":len(rts),"Replies":len(replies),"Mentions":len(mentions),"Quotes":len(quotes)}
    pie = px.pie(pd.DataFrame({"type":dd.keys(),"count":dd.values()}), names="type", values="count", hole=0.55, title="Engagement breakdown")
    st.plotly_chart(pie, use_container_width=True)

# ===== Generic sections =====
def section_tab(df, label):
    st.subheader(label)
    if df.empty:
        st.info(f"No {label.lower()} for the current selection.")
        return
    st.plotly_chart(px.area(daily_counts(df,label), x="date", y="count", title=f"{label} by day"),
                    use_container_width=True)
    if label in ("Replies","Mentions","Retweets"):
        st.markdown("**Top users**")
        st.plotly_chart(px.bar(top_users(df, n=15), x="interactions", y="user", orientation="h",
                               title="Most active users"), use_container_width=True)
    st.markdown("**Recent**")
    st.dataframe(recent_table(df), use_container_width=True, height=360)

with tabs[1]: section_tab(posts, "Posts")
with tabs[2]: section_tab(rts, "Retweets")
with tabs[3]: section_tab(replies, "Replies")
with tabs[4]: section_tab(mentions, "Mentions")
with tabs[5]: section_tab(quotes, "Quotes")

# ===== Insights =====
with tabs[6]:
    st.subheader("Quick insights")
    # Top hashtags / terms en replies+mentions
    rm = pd.concat([replies, mentions], ignore_index=True)
    if "__text" in rm.columns and not rm.empty:
        URL_RE = re.compile(r"https?://\\S+"); MENTION_RE = re.compile(r"@\\w+"); HASHTAG_RE = re.compile(r"#\\w+")
        def tok(s):
            if not isinstance(s,str): return []
            s = URL_RE.sub(" ", s); s = MENTION_RE.sub(" ", s)
            s = s.lower().translate(str.maketrans("", "", string.punctuation))
            return [w for w in s.split() if w and not w.isdigit()]
        tags=[]; terms=[]
        for t in rm["__text"].dropna().tolist():
            tags.extend([h.lower() for h in HASHTAG_RE.findall(t)])
            terms.extend(tok(t))
        if tags:
            vc = pd.Series(tags).value_counts().head(20).reset_index().rename(columns={"index":"hashtag",0:"count"})
            st.plotly_chart(px.bar(vc, x="count", y="hashtag", orientation="h", title="Top hashtags"),
                            use_container_width=True)
        if terms:
            vc2 = pd.Series(terms).value_counts().head(20).reset_index().rename(columns={"index":"term",0:"count"})
            st.plotly_chart(px.bar(vc2, x="count", y="term", orientation="h", title="Top terms"),
                            use_container_width=True)
    else:
        st.info("Add Replies/Mentions to see topics & hashtags.")

st.caption("Tip: activa 'Load from local folder' para leer directamente Posts.csv, Retweets.csv, Replies.csv, Mentions.csv y Quotes.csv desde la misma carpeta del app.py.")
# Start app in light mode; defer big CSVs to a button


import re
import string
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIG & THEME (claro y de alto contraste suave)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="HRH @Alwaleed_Talal Analytics Dashboard", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#f6f7fb; --panel:#ffffff; --soft:#eef1f7; --muted:#6a7280; --ink:#111827;
  --accent:#ef4444; --accent2:#0ea5e9; --rim:#e5e7eb;
}
.block-container{max-width:1400px !important;}
.main, .block-container{background:var(--bg); color:var(--ink);}
[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--rim);}
h1,h2,h3{color:var(--ink);}
hr{border-top:1px solid var(--rim);}

.app-header{
  background:var(--panel); border:1px solid var(--rim);
  border-radius:14px; padding:14px 18px; display:flex; align-items:center; gap:10px;
}
.badge{background:var(--soft); border:1px solid var(--rim); color:var(--ink);
  padding:8px 12px; border-radius:999px; font-weight:600; display:inline-flex; align-items:center; gap:8px;}

.kpi{
  background:var(--panel); border:1px solid var(--rim); border-radius:16px;
  padding:18px; min-height:110px; display:flex; flex-direction:column; justify-content:center;
}
.kpi .label{font-size:.85rem; color:var(--muted);}
.kpi .value{font-size:1.6rem; font-weight:800;}

.card{
  background:var(--panel); border:1px solid var(--rim); border-radius:16px; padding:12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HELPERS: lectura robusta y parseo de fechas
# ------------------------------------------------------------------------------
def read_csv_safe(file):
    # prueba codificaciones y separadores comunes
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        for sep in (",",";","\t","|"):
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                pass
    return pd.DataFrame()

def parse_datetime_any(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if s.isna().mean() > 0.8:
        num = pd.to_numeric(series, errors="coerce")
        s_ms = pd.to_datetime(num, unit="ms", utc=True, errors="coerce")
        s = s_ms if s_ms.notna().sum() > s.notna().sum() else pd.to_datetime(num, unit="s", utc=True, errors="coerce")
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
    s = link_series.astype(str).str.extract(r"/status/(\\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")

# normalización rápida
def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    out["__type"] = label
    low = {c: c.lower() for c in out.columns}
    # fecha
    candidates_exact = {"time","date","created_at","created at","timestamp","createdat","tweet time","tweet_time","tweet_date"}
    tcol = next((c for c in out.columns if low[c] in candidates_exact), None)
    if tcol is None:
        tcol = next((c for c in out.columns if re.search(r"(date|time|created|timestamp)", c, re.I)), None)
    out["__time"] = parse_datetime_any(out[tcol]) if tcol else pd.NaT
    # user / text / link / id
    ucol = next((c for c in out.columns if re.search(r"(user|username|screen|author|from|handle)", c, re.I)), None)
    xcol = next((c for c in out.columns if re.search(r"(text|tweet|content|full_text|body)", c, re.I)), None)
    lcol = next((c for c in out.columns if re.search(r"(link|url|permalink)", c, re.I)), None)
    icol = next((c for c in out.columns if re.search(r"(tweet[_-]?id|status[_-]?id|id_str|^id$)", c, re.I)), None)
    out["__user"] = out[ucol] if ucol else None
    out["__text"] = out[xcol] if xcol else None
    out["__link"] = out[lcol] if lcol else None
    out["__id"]   = out[icol] if icol else None
    return out

def map_columns_ui(df: pd.DataFrame, label: str):
    if df.empty: return df
    st.sidebar.markdown(f"**Column mapping · {label}**")
    # fecha
    if df["__time"].isna().all():
        dt_col = st.sidebar.selectbox(f"Pick datetime column for {label}", options=list(df.columns), index=None, key=f"dt_{label}")
        if dt_col: df = df.copy(); df["__time"] = parse_datetime_any(df[dt_col])
    # user
    if "__user" in df and (df["__user"].isna().all() if "__user" in df else True):
        u_col = st.sidebar.selectbox(f"Pick user column for {label}", options=["<skip>"]+list(df.columns), index=0, key=f"user_{label}")
        if u_col and u_col != "<skip>": df = df.copy(); df["__user"] = df[u_col].astype(str)
    # text
    if "__text" in df and (df["__text"].isna().all() if "__text" in df else True):
        x_col = st.sidebar.selectbox(f"Pick text column for {label}", options=["<skip>"]+list(df.columns), index=0, key=f"text_{label}")
        if x_col and x_col != "<skip>": df = df.copy(); df["__text"] = df[x_col].astype(str)
    # link
    if "__link" in df and (df["__link"].isna().all() if "__link" in df else True):
        l_col = st.sidebar.selectbox(f"Pick link column for {label} (optional)", options=["<skip>"]+list(df.columns), index=0, key=f"link_{label}")
        if l_col and l_col != "<skip>": df = df.copy(); df["__link"] = df[l_col].astype(str)
    return df

def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "__time" not in df.columns: df["__time"] = pd.NaT; return df
    t = df["__time"]
    needs_fix = t.isna().all()
    if not needs_fix:
        non_na = t.dropna().unique()
        needs_fix = len(non_na) <= 1
    if needs_fix:
        if "__id" in df.columns and df["__id"].notna().any():
            dt = snowflake_to_datetime(df["__id"])
            if dt.notna().any(): df = df.copy(); df["__time"] = dt; return df
        link_col = next((c for c in df.columns if re.search(r"(link|url|permalink)", c, re.I)), None)
        if link_col:
            ids = extract_id_from_link(df[link_col]); dt = snowflake_to_datetime(ids)
            if dt.notna().any(): df = df.copy(); df["__time"] = dt; return df
    return df

# agregaciones
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

def plot_heatmap(df, title):
    if df.empty or df["__time"].dropna().empty: 
        return None
    d = df.dropna(subset=["__time"]).copy()
    d["weekday"] = d["__time"].dt.day_name()
    d["hour"]    = d["__time"].dt.hour
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d["weekday"] = pd.Categorical(d["weekday"], categories=order, ordered=True)
    # escala suave
    return px.density_heatmap(d, x="hour", y="weekday", nbinsx=24,
                              color_continuous_scale="Blues", title=title)

# ------------------------------------------------------------------------------
# SIDEBAR · Carga de datos
# ------------------------------------------------------------------------------
st.sidebar.header("Upload CSV files")
mentions_csv = st.sidebar.file_uploader("Mentions.csv", type=["csv"])
sent_csv     = st.sidebar.file_uploader("Posts.csv (sent/from)", type=["csv"])
quotes_csv   = st.sidebar.file_uploader("Quotes.csv", type=["csv"])
rts_csv      = st.sidebar.file_uploader("Retweets.csv (retweets_of)", type=["csv"])
replies_csv  = st.sidebar.file_uploader("Replies.csv (to:)", type=["csv"])

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("Show debug info", value=True)
disable_date_filter = st.sidebar.checkbox("Disable date filter", value=True)

st.sidebar.subheader("Date Range Filter")
start_date = st.sidebar.date_input("Start date")
end_date   = st.sidebar.date_input("End date")

# ------------------------------------------------------------------------------
# LOAD + NORMALIZE + DIAGNOSTICS
# ------------------------------------------------------------------------------
def load_norm(upl, label):
    if upl is None: return pd.DataFrame()
    df = read_csv_safe(upl)
    norm = normalize(df, label)
    norm = map_columns_ui(norm, label)
    norm = ensure_valid_time(norm)
    if debug_mode:
        st.sidebar.write(f"**{label}** → rows: {len(norm)} · cols: {norm.shape[1]}")
    return norm

sent_df     = load_norm(sent_csv, "Posts")
rts_df      = load_norm(rts_csv, "Retweets")
mentions_df = load_norm(mentions_csv, "Mentions")
replies_df  = load_norm(replies_csv, "Replies")
quotes_df   = load_norm(quotes_csv, "Quotes")

if debug_mode:
    diag = pd.DataFrame({
        "section":["Posts","Retweets","Replies","Mentions","Quotes"],
        "rows":[len(sent_df),len(rts_df),len(replies_df),len(mentions_df),len(quotes_df)],
        "cols":[sent_df.shape[1] if not sent_df.empty else 0,
                rts_df.shape[1] if not rts_df.empty else 0,
                replies_df.shape[1] if not replies_df.empty else 0,
                mentions_df.shape[1] if not mentions_df.empty else 0,
                quotes_df.shape[1] if not quotes_df.empty else 0]
    })
    st.sidebar.subheader("Diagnostics (before date filter)")
    st.sidebar.dataframe(diag, height=220, use_container_width=True)

def filt(df):
    if df.empty: return df
    out = df.copy()
    out["__time"] = parse_datetime_any(out["__time"])
    if disable_date_filter:
        return out
    st_ts = pd.Timestamp(start_date).tz_localize(None) if start_date else pd.Timestamp.min
    en_ts = pd.Timestamp(end_date).tz_localize(None) if end_date else pd.Timestamp.max
    return out[(out["__time"] >= st_ts) & (out["__time"] <= en_ts)]

sent_df, rts_df, mentions_df, replies_df, quotes_df = map(
    filt, [sent_df, rts_df, mentions_df, replies_df, quotes_df]
)

# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------
col_logo, col_title, col_avatar = st.columns([1,6,1])
with col_logo:
    st.markdown('<div class="badge">🔶 Audiense</div>', unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="app-header"><span class="badge">HRH @Alwaleed_Talal Analytics Dashboard</span></div>', unsafe_allow_html=True)
with col_avatar:
    st.image("HRH avatar.jpg", caption="", use_column_width=True)

st.title("HRH Data Dashboard")

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Summary","Posts","Retweets","Replies","Mentions","Quotes"])

# ===== Summary =====
with tab1:
    st.header("Weekly Summary")

    s_posts    = to_daily_series(sent_df)
    s_rts      = to_daily_series(rts_df)
    s_replies  = to_daily_series(replies_df)
    s_mentions = to_daily_series(mentions_df)
    s_quotes   = to_daily_series(quotes_df)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f'<div class="kpi"><div class="label">Posts</div><div class="value">{len(sent_df)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><div class="label">Retweets</div><div class="value">{len(rts_df)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi"><div class="label">Replies</div><div class="value">{len(replies_df)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><div class="label">Mentions</div><div class="value">{len(mentions_df)}</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="kpi"><div class="label">Quotes</div><div class="value">{len(quotes_df)}</div></div>', unsafe_allow_html=True)

    # Heatmap de actividad global (audiencia)
    all_df = pd.concat([rts_df, replies_df, mentions_df, quotes_df], ignore_index=True)
    hm_all = plot_heatmap(all_df, "When activity happens (hour × weekday)")
    if hm_all: st.plotly_chart(hm_all, use_container_width=True)
    else: st.info("Upload CSVs or map the date columns to see the heatmap.")

# ===== Posts =====
with tab2:
    st.header("Posts by HRH")
    if sent_df.empty or sent_df["__time"].dropna().empty:
        st.info("Upload Posts.csv and/or map the date column in the sidebar.")
    else:
        hm = plot_heatmap(sent_df, "Posting times (hour × weekday)")
        st.plotly_chart(hm, use_container_width=True)

        daily = daily_counts(sent_df, "Posts")
        if not daily.empty:
            st.plotly_chart(px.bar(daily, x="date", y="count", title="Posts by day"), use_container_width=True)

        # Tabla reciente
        cols = ["__time","__user","__text","__link"]
        show = sent_df[cols].rename(columns={"__time":"time","__user":"user","__text":"text","__link":"link"})
        st.markdown("### Recent posts")
        st.dataframe(show.sort_values("time", ascending=False).head(50), use_container_width=True, height=360)

# ===== Retweets =====
with tab3:
    st.header("Retweets of HRH")
    if rts_df.empty or rts_df["__time"].dropna().empty:
        st.info("Upload Retweets.csv and/or map the date column.")
    else:
        hm = plot_heatmap(rts_df, "Retweets timing (hour × weekday)")
        st.plotly_chart(hm, use_container_width=True)

        st.plotly_chart(px.area(daily_counts(rts_df, "Retweets"), x="date", y="count", title="Retweets by day"),
                        use_container_width=True)

        top = (rts_df.groupby("__user", dropna=False).size()
                  .reset_index(name="retweets")
                  .sort_values("retweets", ascending=False).head(15))
        st.plotly_chart(px.bar(top, x="retweets", y="__user", orientation="h", title="Top retweeters"),
                        use_container_width=True)

# ===== Replies =====
with tab4:
    st.header("Replies to HRH")
    if replies_df.empty or replies_df["__time"].dropna().empty:
        st.info("Upload Replies.csv and/or map the date column.")
    else:
        hm = plot_heatmap(replies_df, "Replies timing (hour × weekday)")
        st.plotly_chart(hm, use_container_width=True)

        st.plotly_chart(px.area(daily_counts(replies_df, "Replies"), x="date", y="count", title="Replies by day"),
                        use_container_width=True)

        top = (replies_df.groupby("__user", dropna=False).size()
                  .reset_index(name="replies")
                  .sort_values("replies", ascending=False).head(15))
        st.plotly_chart(px.bar(top, x="replies", y="__user", orientation="h", title="Most active repliers"),
                        use_container_width=True)

# ===== Mentions =====
with tab5:
    st.header("Mentions of HRH")
    if mentions_df.empty or mentions_df["__time"].dropna().empty:
        st.info("Upload Mentions.csv and/or map the date column.")
    else:
        hm = plot_heatmap(mentions_df, "Mentions timing (hour × weekday)")
        st.plotly_chart(hm, use_container_width=True)

        st.plotly_chart(px.area(daily_counts(mentions_df, "Mentions"), x="date", y="count", title="Mentions by day"),
                        use_container_width=True)

        top = (mentions_df.groupby("__user", dropna=False).size()
                  .reset_index(name="mentions")
                  .sort_values("mentions", ascending=False).head(15))
        st.plotly_chart(px.bar(top, x="mentions", y="__user", orientation="h", title="Top mentioners"),
                        use_container_width=True)

# ===== Quotes =====
with tab6:
    st.header("Quotes / Links to HRH content")
    if quotes_df.empty or quotes_df["__time"].dropna().empty:
        st.info("Upload Quotes.csv and/or map the date column.")
    else:
        hm = plot_heatmap(quotes_df, "Quotes timing (hour × weekday)")
        st.plotly_chart(hm, use_container_width=True)

        st.plotly_chart(px.bar(daily_counts(quotes_df, "Quotes"), x="date", y="count", title="Quotes by day"),
                        use_container_width=True)

st.caption("Tip: usa la barra lateral para mapear columnas (fecha, usuario, texto, link) o desactiva el filtro de fechas si una pestaña aparece vacía.")

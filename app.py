import re
import string
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# -----------------------------------------------------------------------------------
# CONFIG & THEME
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="HRH Data Dashboard", layout="wide")

st.markdown("""
<style>
/* Theme */
.main, .block-container {background: #0f1116; color:#e6e6e6;}
[data-testid="stSidebar"] {background: #0b0d12;}
h1, h2, h3 {color:#f0f2f6;}
/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg, #1b1f2a 0%, #131724 100%);
  border: 1px solid #23283a; border-radius:16px;
  padding:16px; margin-bottom:12px;
}
.kpi-title {font-size:0.85rem; color:#9aa4b2; margin-bottom:6px;}
.kpi-value {font-size:1.6rem; font-weight:700; color:#ffffff;}
.kpi-delta-up {color:#3ddc97; font-weight:600; font-size:0.9rem;}
.kpi-delta-down {color:#ff6b6b; font-weight:600; font-size:0.9rem;}
/* Tables & links */
table {color:#e6e6e6;}
a {color:#4ea1ff;}
</style>
""", unsafe_allow_html=True)

pio.templates.default = "plotly_dark"

# -----------------------------------------------------------------------------------
# HELPERS: read / parse
# -----------------------------------------------------------------------------------
def read_csv_safe(buffer):
    for enc in ("utf-8","utf-8-sig","latin1"):
        for sep in (",",";","\t","|"):
            try:
                df = pd.read_csv(buffer, encoding=enc, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                try: buffer.seek(0)
                except Exception: pass
    return pd.DataFrame()

def parse_datetime_any(series: pd.Series) -> pd.Series:
    """ISO con/sin tz; si falla, epoch ms / epoch s. Devuelve naive."""
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

TWITTER_EPOCH_MS = 1288834974657  # 2010-11-04

def snowflake_to_datetime(snowflake_series: pd.Series) -> pd.Series:
    ids = pd.to_numeric(snowflake_series, errors="coerce").astype("Int64")
    ms = (ids >> 22) + TWITTER_EPOCH_MS
    dt = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt

def extract_id_from_link(link_series: pd.Series) -> pd.Series:
    s = link_series.astype(str).str.extract(r"/status/(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")

# -----------------------------------------------------------------------------------
# Normalización y UI mapping
# -----------------------------------------------------------------------------------
def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["__type"] = label
    low = {c: c.lower() for c in out.columns}

    # Fecha
    time_candidates_exact = [
        "time","date","created_at","created at","tweet time","tweet_time",
        "timestamp","tweet timestamp","tweet_timestamp","tweet_created_at",
        "tweet_date","date_utc","time_utc","createdat"
    ]
    tcol = next((c for c in out.columns if low[c] in time_candidates_exact), None)
    if tcol is None:
        tcol = next((c for c in out.columns if re.search(r"(date|time|created|timestamp)", c, re.I)), None)
    out["__time"] = parse_datetime_any(out[tcol]) if tcol else pd.NaT

    # User / text / link
    ucol = next((c for c in out.columns if re.search(r"(user|username|screen|author|from|profile|handle)", c, re.I)), None)
    xcol = next((c for c in out.columns if re.search(r"(text|tweet|content|full_text|body)", c, re.I)), None)
    lcol = next((c for c in out.columns if re.search(r"(link|url|permalink)", c, re.I)), None)
    out["__user"] = out[ucol] if ucol else None
    out["__text"] = out[xcol] if xcol else None
    out["__link"] = out[lcol] if lcol else None

    # Tweet ID
    idcol = next((c for c in out.columns if re.search(r"(tweet[_-]?id|status[_-]?id|id_str|id$)", c, re.I)), None)
    out["__id"] = out[idcol] if idcol else None
    return out

def map_columns_ui(df: pd.DataFrame, label: str):
    if df.empty: return df
    st.sidebar.markdown(f"**Column mapping for {label}**")
    # Fecha
    if df["__time"].isna().all():
        dt_col = st.sidebar.selectbox(
            f"Pick datetime column for {label}",
            options=list(df.columns), index=None, key=f"dt_{label}"
        )
        if dt_col:
            df = df.copy()
            df["__time"] = parse_datetime_any(df[dt_col])
    # User
    if "__user" in df and (df["__user"].isna().all() if "__user" in df else True):
        u_col = st.sidebar.selectbox(
            f"Pick user column for {label}",
            options=["<skip>"] + list(df.columns), index=0, key=f"user_{label}"
        )
        if u_col and u_col != "<skip>":
            df = df.copy(); df["__user"] = df[u_col].astype(str)
    # Text
    if "__text" in df and (df["__text"].isna().all() if "__text" in df else True):
        x_col = st.sidebar.selectbox(
            f"Pick text column for {label}",
            options=["<skip>"] + list(df.columns), index=0, key=f"text_{label}"
        )
        if x_col and x_col != "<skip>":
            df = df.copy(); df["__text"] = df[xcol].astype(str)
    # Link
    if "__link" in df and (df["__link"].isna().all() if "__link" in df else True):
        l_col = st.sidebar.selectbox(
            f"Pick link column for {label} (optional)",
            options=["<skip>"] + list(df.columns), index=0, key=f"link_{label}"
        )
        if l_col and l_col != "<skip>":
            df = df.copy(); df["__link"] = df[l_col].astype(str)
    return df

def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if "__time" not in df.columns:
        df["__time"] = pd.NaT; return df
    t = df["__time"]
    needs_fix = t.isna().all()
    if not needs_fix:
        unique_non_na = t.dropna().unique()
        needs_fix = len(unique_non_na) <= 1  # p.ej., createdAt del perfil
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

# -----------------------------------------------------------------------------------
# Aggregations / charts helpers
# -----------------------------------------------------------------------------------
def daily_counts(df, label):
    if df.empty or df["__time"].dropna().empty:
        return pd.DataFrame(columns=["date","count","type"])
    d = df.dropna(subset=["__time"]).copy()
    d["date"] = d["__time"].dt.date
    d = d.groupby("date").size().reset_index(name="count")
    d["type"] = label
    return d

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
    if delta is None: delta_html = ""
    else:
        is_up = (delta >= 0)
        if invert: is_up = not is_up
        cls = "kpi-delta-up" if is_up else "kpi-delta-down"
        sign = "+" if delta>=0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.1f}%</div>'
    st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
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
    return ( (curr - prev) / prev ) * 100

def to_daily_series(df):
    if df.empty or df["__time"].dropna().empty: 
        return pd.Series(dtype=float)
    d = df.dropna(subset=["__time"]).copy()
    d["date"] = d["__time"].dt.date
    s = d.groupby("date").size()
    s.index = pd.to_datetime(s.index)
    return s

def heatmap_hour_week(df, title):
    if df.empty or df["__time"].dropna().empty:
        return None
    d = df.dropna(subset=["__time"]).copy()
    d["weekday"] = d["__time"].dt.day_name()
    d["hour"]    = d["__time"].dt.hour
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d["weekday"] = pd.Categorical(d["weekday"], categories=order, ordered=True)
    return px.density_heatmap(d, x="hour", y="weekday", nbinsx=24,
                              color_continuous_scale="Blues", title=title)

def engagement_breakdown(retweets, replies, mentions, quotes, title="Engagement breakdown"):
    vals = {"Retweets":len(retweets),"Replies":len(replies),"Mentions":len(mentions),"Quotes":len(quotes)}
    dd = pd.DataFrame({"type":list(vals.keys()), "count":list(vals.values())})
    return px.pie(dd, names="type", values="count", hole=0.5, title=title)

# -----------------------------------------------------------------------------------
# INSIGHTS HELPERS (topics, hashtags, spikes)
# -----------------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
DEFAULT_STOPWORDS = {
    # genéricos
    "rt","via","amp","https","http","co","tco","www","com",
    # inglés
    "the","and","a","an","of","to","in","on","for","with","at","by","from","or","as","is","are","was","were","be","been","it","this","that","these","those","we","you","they","he","she","i","me","my","our","your","their","them",
    # español
    "de","la","el","los","las","y","o","en","por","para","con","un","una","que","es","son","ha","han","más","muy","ya","como","si","sobre","al","del",
    # árabe básico (añade si quieres)
    "و","في","من","على","الى","عن","مع","هذا","هذه","ذلك","انا","انت","هو","هي"
}
def build_engagement_df(rts_df, replies_df, mentions_df, quotes_df, sent_df):
    frames = []
    for name, df in [("Retweets", rts_df), ("Replies", replies_df), ("Mentions", mentions_df), ("Quotes", quotes_df), ("Posts", sent_df)]:
        if not df.empty:
            d = df[["__time","__text"]].copy()
            d["type"] = name
            frames.append(d)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["__time","__text","type"])

def daily_total_series(all_df):
    if all_df.empty or all_df["__time"].dropna().empty:
        return pd.Series(dtype=float)
    d = all_df.dropna(subset=["__time"]).copy()
    d["date"] = d["__time"].dt.date
    s = d.groupby("date").size()
    s.index = pd.to_datetime(s.index)
    return s

def detect_spikes(series, window=7, z=2.0):
    """días con valor > media_móvil + z*std_móvil"""
    if series.empty or len(series) < window:
        return pd.DataFrame(columns=["date","count","rolling_mean","rolling_std","zscore"])
    s = series.sort_index()
    mean = s.rolling(window).mean()
    std  = s.rolling(window).std().fillna(0)
    zscore = (s - mean) / std.replace(0, np.nan)
    spikes = s[(s > (mean + z*std)) & (~mean.isna())]
    out = pd.DataFrame({
        "date": spikes.index.date,
        "count": spikes.values,
        "rolling_mean": mean.reindex(spikes.index).values,
        "rolling_std": std.reindex(spikes.index).values,
        "zscore": zscore.reindex(spikes.index).values
    })
    return out.sort_values("date", ascending=False)

def tokenize(text):
    if not isinstance(text, str):
        return []
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = t.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w and w not in DEFAULT_STOPWORDS and not w.isdigit()]
    return tokens

def top_terms(df_text, n=20):
    if df_text.empty or "__text" not in df_text.columns:
        return pd.DataFrame(columns=["term","count"])
    all_tokens = []
    for txt in df_text["__text"].dropna().tolist():
        all_tokens.extend(tokenize(txt))
    if not all_tokens:
        return pd.DataFrame(columns=["term","count"])
    vc = pd.Series(all_tokens).value_counts().head(n)
    return vc.reset_index().rename(columns={"index":"term",0:"count"})

def top_hashtags(df_text, n=20):
    if df_text.empty or "__text" not in df_text.columns:
        return pd.DataFrame(columns=["hashtag","count"])
    tags = []
    for txt in df_text["__text"].dropna().tolist():
        tags.extend([h.lower() for h in HASHTAG_RE.findall(txt)])
    if not tags:
        return pd.DataFrame(columns=["hashtag","count"])
    vc = pd.Series(tags).value_counts().head(n)
    return vc.reset_index().rename(columns={"index":"hashtag",0:"count"})

def best_hours_heatmap(df, title):
    if df.empty or df["__time"].dropna().empty:
        return None
    d = df.dropna(subset=["__time"]).copy()
    d["weekday"] = d["__time"].dt.day_name()
    d["hour"]    = d["__time"].dt.hour
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d["weekday"] = pd.Categorical(d["weekday"], categories=order, ordered=True)
    return px.density_heatmap(d, x="hour", y="weekday", nbinsx=24, color_continuous_scale="Viridis", title=title)

# -----------------------------------------------------------------------------------
# UI: inputs
# -----------------------------------------------------------------------------------
st.title("HRH Data Dashboard")

st.sidebar.header("Upload CSV Files (HRH Data)")
mentions_csv = st.sidebar.file_uploader(
    "Mentions", type=["csv"]
)
sent_csv = st.sidebar.file_uploader(
    "Posts", type=["csv"]
)
quotes_csv = st.sidebar.file_uploader(
    "Quotes", type=["csv"]
)
rts_csv = st.sidebar.file_uploader(
    "Retweets", type=["csv"]
)
replies_csv = st.sidebar.file_uploader(
    "Replies", type=["csv"]
)

st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("Show debug info (columns & sample)", value=False)
disable_date_filter = st.sidebar.checkbox("Disable date filter", value=False)

st.sidebar.subheader("Date Range Filter")
default_end = pd.Timestamp.today().normalize()
default_start = default_end - pd.Timedelta(days=7)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date   = st.sidebar.date_input("End date", value=default_end)

# -----------------------------------------------------------------------------------
# Load & normalize
# -----------------------------------------------------------------------------------
def load_norm(upl, label):
    if upl is None: return pd.DataFrame()
    df = read_csv_safe(upl)
    if debug_mode and not df.empty:
        st.sidebar.write(f"**{label} columns**:", list(df.columns))
        st.sidebar.write(df.head(3))
    norm = normalize(df, label)
    norm = map_columns_ui(norm, label)   # mapeo manual si fuese necesario
    norm = ensure_valid_time(norm)       # reconstrucción de fecha desde id/link si procede
    return norm

sent_df     = load_norm(sent_csv, "Posts")
rts_df      = load_norm(rts_csv, "Retweets")
mentions_df = load_norm(mentions_csv, "Mentions")
replies_df  = load_norm(replies_csv, "Replies")
quotes_df   = load_norm(quotes_csv, "Quotes")

# -----------------------------------------------------------------------------------
# Filter
# -----------------------------------------------------------------------------------
def filt(df):
    if df.empty: return df
    out = df.copy()
    out["__time"] = parse_datetime_any(out["__time"])
    if disable_date_filter:
        return out
    start_ts = pd.Timestamp(start_date).tz_localize(None)
    end_ts   = pd.Timestamp(end_date).tz_localize(None)
    return out[(out["__time"] >= start_ts) & (out["__time"] <= end_ts)]

sent_df, rts_df, mentions_df, replies_df, quotes_df = map(
    filt, [sent_df, rts_df, mentions_df, replies_df, quotes_df]
)

# -----------------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Summary","Posts","Retweets","Replies","Mentions","Quotes","Insights"]
)

# ===== Summary =====
with tab1:
    st.subheader("Weekly Summary")

    s_posts    = to_daily_series(sent_df)
    s_rts      = to_daily_series(rts_df)
    s_replies  = to_daily_series(replies_df)
    s_mentions = to_daily_series(mentions_df)
    s_quotes   = to_daily_series(quotes_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Posts", len(sent_df), prev_period_delta(s_posts, start_date, end_date))
    with c2: kpi_card("Retweets", len(rts_df), prev_period_delta(s_rts, start_date, end_date))
    with c3: kpi_card("Replies", len(replies_df), prev_period_delta(s_replies, start_date, end_date))
    with c4: kpi_card("Mentions", len(mentions_df), prev_period_delta(s_mentions, start_date, end_date))
    with c5: kpi_card("Quotes", len(quotes_df), prev_period_delta(s_quotes, start_date, end_date))

    frames = []
    for name, df in [("Posts",sent_df),("Retweets",rts_df),("Replies",replies_df),("Mentions",mentions_df),("Quotes",quotes_df)]:
        if not df.empty: frames.append(daily_counts(df, name))
    if frames:
        trend = pd.concat(frames, ignore_index=True)
        fig = px.area(trend, x="date", y="count", color="type", title="Activity over time (stacked)")
        st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(engagement_breakdown(rts_df, replies_df, mentions_df, quotes_df), use_container_width=True)

    all_df_summary = pd.concat([rts_df, replies_df, mentions_df, quotes_df, sent_df], ignore_index=True)
    fig_hm = heatmap_hour_week(all_df_summary, "When activity happens (hour x weekday)")
    if fig_hm:
        st.plotly_chart(fig_hm, use_container_width=True)

# ===== Posts =====
with tab2:
    st.subheader("Posts by HRH")
    if sent_df.empty or sent_df["__time"].dropna().empty:
        st.info("Upload the posts CSV and/or map the date column in the sidebar.")
    else:
        st.plotly_chart(
            px.bar(daily_counts(sent_df, "Posts"), x="date", y="count", title="Posts by day"),
            use_container_width=True
        )
        fig_hm = heatmap_hour_week(sent_df, "Posting times (hour x weekday)")
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True)

        metric_cols = [c for c in ["__rt","__re","__qt","__lk","__vw"] if c in sent_df.columns]
        if metric_cols:
            tmp = sent_df.copy()
            tmp["eng"] = tmp[metric_cols].fillna(0).sum(axis=1)
            top = tmp.sort_values("eng", ascending=False).head(10)
            show = top[["__time","__text","eng","__link"]].rename(columns={"__time":"time","__text":"text","__link":"link"})
            st.markdown("**Top posts by total interactions**")
            st.dataframe(show, use_container_width=True, height=360)
        else:
            st.markdown("**Recent posts**")
            st.dataframe(recent_table(sent_df), use_container_width=True, height=420)

# ===== Retweets =====
with tab3:
    st.subheader("Retweets of HRH")
    if rts_df.empty:
        st.info("Upload the retweets_of CSV.")
    else:
        st.plotly_chart(
            px.area(daily_counts(rts_df, "Retweets"), x="date", y="count", title="Retweets by day"),
            use_container_width=True
        )
        st.markdown("**Top amplifiers (users retweeting)**")
        st.plotly_chart(
            px.bar(top_users(rts_df, n=15), x="interactions", y="user", orientation="h", title="Top retweeters"),
            use_container_width=True
        )
        st.markdown("**Recent retweets**")
        st.dataframe(recent_table(rts_df), use_container_width=True, height=360)

# ===== Replies =====
with tab4:
    st.subheader("Replies to HRH")
    if replies_df.empty:
        st.info("Upload the replies CSV.")
    else:
        st.plotly_chart(
            px.area(daily_counts(replies_df, "Replies"), x="date", y="count", title="Replies by day"),
            use_container_width=True
        )
        fig_hm = heatmap_hour_week(replies_df, "Replies timing (hour x weekday)")
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown("**Top users by replies**")
        st.plotly_chart(
            px.bar(top_users(replies_df, n=15), x="interactions", y="user", orientation="h", title="Most active repliers"),
            use_container_width=True
        )
        st.markdown("**Recent replies**")
        st.dataframe(recent_table(replies_df), use_container_width=True, height=360)

# ===== Mentions =====
with tab5:
    st.subheader("Mentions of HRH")
    if mentions_df.empty:
        st.info("Upload the mentions CSV.")
    else:
        st.plotly_chart(
            px.area(daily_counts(mentions_df, "Mentions"), x="date", y="count", title="Mentions by day"),
            use_container_width=True
        )
        st.markdown("**Top users mentioning HRH**")
        st.plotly_chart(
            px.bar(top_users(mentions_df, n=15), x="interactions", y="user", orientation="h", title="Top mentioners"),
            use_container_width=True
        )
        st.markdown("**Recent mentions**")
        st.dataframe(recent_table(mentions_df), use_container_width=True, height=360)

# ===== Quotes =====
with tab6:
    st.subheader("Quotes / Links to HRH content")
    if quotes_df.empty:
        st.info("Upload the quotes/links CSV.")
    else:
        st.plotly_chart(
            px.bar(daily_counts(quotes_df, "Quotes"), x="date", y="count", title="Quotes by day"),
            use_container_width=True
        )
        st.markdown("**Recent quotes**")
        st.dataframe(recent_table(quotes_df), use_container_width=True, height=420)

# ===== Insights =====
with tab7:
    st.subheader("Strategic Insights")

    # Construye dataset de engagement total
    all_df = build_engagement_df(rts_df, replies_df, mentions_df, quotes_df, sent_df)

    # Momentum (periodo actual vs anterior)
    s_total = daily_total_series(all_df)
    mom = prev_period_delta(s_total, start_date, end_date)
    c1, c2, c3 = st.columns(3)
    curr_sum = int(s_total.loc[(s_total.index>=pd.Timestamp(start_date)) & (s_total.index<=pd.Timestamp(end_date))].sum() if not s_total.empty else 0)
    with c1: kpi_card("Total interactions (period)", curr_sum, None)
    with c2: kpi_card("Momentum vs prev. period", float(mom) if mom is not None else "n/a")
    with c3: kpi_card("Active days (period)", s_total.loc[(s_total.index>=pd.Timestamp(start_date)) & (s_total.index<=pd.Timestamp(end_date))].astype(bool).sum() if not s_total.empty else 0)

    # Spikes detection
    st.markdown("### Spikes (vs 7-day rolling mean)")
    spikes = detect_spikes(s_total, window=7, z=2.0)
    if not spikes.empty:
        fig_sp = px.bar(spikes.sort_values("date"), x="date", y="count", title="Detected spikes")
        st.plotly_chart(fig_sp, use_container_width=True)
        st.dataframe(spikes, use_container_width=True, height=240)
    else:
        st.info("No spikes detected for the current selection.")

    # Topics & Hashtags (sobre replies + mentions del periodo)
    st.markdown("### Topics and Hashtags (Replies + Mentions)")
    rm_df = pd.concat([replies_df, mentions_df], ignore_index=True)
    if not rm_df.empty:
        rm_df = rm_df[(rm_df["__time"]>=pd.Timestamp(start_date)) & (rm_df["__time"]<=pd.Timestamp(end_date))]
    colA, colB = st.columns(2)
    with colA:
        th = top_hashtags(rm_df, n=20)
        if not th.empty:
            st.plotly_chart(px.bar(th, x="count", y="hashtag", orientation="h", title="Top hashtags"), use_container_width=True)
        else:
            st.info("No hashtags found in replies/mentions for this period.")
    with colB:
        tt = top_terms(rm_df, n=20)
        if not tt.empty:
            st.plotly_chart(px.bar(tt, x="count", y="term", orientation="h", title="Top terms"), use_container_width=True)
        else:
            st.info("No terms found in replies/mentions for this period.")

    # Mejores horas y días para publicar (engagement total)
    st.markdown("### Best times to post (audience activity)")
    fig_hm2 = best_hours_heatmap(all_df, "Engagement timing heatmap")
    if fig_hm2:
        st.plotly_chart(fig_hm2, use_container_width=True)
    else:
        st.info("Not enough timestamps to build the heatmap.")

st.caption("Tip: use the sidebar to map columns, enable debug, or disable the date filter if a tab looks empty.")

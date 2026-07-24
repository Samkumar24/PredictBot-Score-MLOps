import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="PredictBot Score", page_icon="🤖", layout="wide")

# ── mock data ─────────────────────────────────────────────────────────────────
def make_data(n=168):
    now    = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    times  = [now - timedelta(hours=i) for i in range(n-1, -1, -1)]
    base   = 0.81 + 0.04 * np.sin(np.linspace(0, 4 * np.pi, n))
    scores = np.clip(base + np.random.normal(0, 0.018, n), 0.55, 0.99)
    scores[80:84] += 0.09   # simulate a spike
    return pd.DataFrame({"timestamp": times, "bot_score": scores})

df = make_data()

# ── helpers ───────────────────────────────────────────────────────────────────
def color(s):  return "#f85149" if s >= 0.85 else ("#d29922" if s >= 0.70 else "#3fb950")
def label(s):  return "HIGH"    if s >= 0.85 else ("ELEVATED" if s >= 0.70 else "NORMAL")

cur     = float(df["bot_score"].iloc[-1])
prv     = float(df["bot_score"].iloc[-2])
delta   = cur - prv
avg7    = df["bot_score"].mean()
peak    = df["bot_score"].max()
peak_ts = df.loc[df["bot_score"].idxmax(), "timestamp"].strftime("%b %d %H:%M")
hi_cnt  = int((df["bot_score"] >= 0.85).sum())
n       = len(df)
arrow   = "up" if delta > 0 else "off"
now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## 🤖 PredictBot Score")
st.caption(f"Cloudflare Radar · AI Bot Traffic Monitor · {now_str}")
st.divider()

# ── metric cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Current Bot Score", f"{cur:.4f}",
              delta=f"{delta*100:+.2f}%")
with c2:
    st.metric("Status", label(cur))
with c3:
    st.metric("7-Day Average", f"{avg7:.4f}")
with c4:
    st.metric("7-Day Peak", f"{peak:.4f}", f"at {peak_ts}")

# ── alerts ────────────────────────────────────────────────────────────────────
if delta > 0.05:
    st.error(f"🔺 Bot score spiked +{delta:.4f} in the last hour")
if hi_cnt > 0:
    st.warning(f"⚠️ {hi_cnt} hours above 0.85 in the last 7 days")

# ── chart ─────────────────────────────────────────────────────────────────────
st.markdown("#### Bot Score — Last 7 Days (hourly)")
st.line_chart(
    df.set_index("timestamp")[["bot_score"]],
    color=["#58a6ff"],
    height=260,
    use_container_width=True,
)

# ── bottom row ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Recent Predictions")
    recent = df.tail(12).iloc[::-1].copy()
    recent["Time"]   = recent["timestamp"].dt.strftime("%b %d  %H:%M")
    recent["Score"]  = recent["bot_score"].apply(lambda x: f"{x:.5f}")
    recent["Status"] = recent["bot_score"].apply(label)
    st.dataframe(
        recent[["Time", "Score", "Status"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("#### Score Distribution")
    st.dataframe(
        pd.DataFrame({
            "Band" : ["NORMAL (< 0.70)", "ELEVATED (0.70-0.85)", "HIGH (>= 0.85)"],
            "Hours": [
                int((df["bot_score"] < 0.70).sum()),
                int(((df["bot_score"] >= 0.70) & (df["bot_score"] < 0.85)).sum()),
                int((df["bot_score"] >= 0.85).sum()),
            ],
            "%" : [
                f"{(df['bot_score'] < 0.70).mean()*100:.1f}%",
                f"{((df['bot_score'] >= 0.70) & (df['bot_score'] < 0.85)).mean()*100:.1f}%",
                f"{(df['bot_score'] >= 0.85).mean()*100:.1f}%",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Model Info")
    st.info(
        "**Model** predict-bot-champion @ champion  \n"
        "**Source** Mock data — FastAPI not connected  \n"
        f"**Loaded** {n} hourly predictions"
    )

st.divider()
st.caption("PredictBot Score · Powered by Cloudflare Radar · MLflow champion model")
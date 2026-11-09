"""
Graph-Based Circular Fraud Detection — investigator app.

Reads only the small data bundle in ./app_data (made by Week 5 of the notebook):
no model, no search, no big data -> fast, and fits free hosting.
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Circular Fraud Detection", page_icon="🔁", layout="wide")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_data")
MODELS = ["Baseline", "Hybrid (real-time)", "Hybrid (retrospective)"]
COLORS = {"Baseline": "#5B6B82", "Hybrid (real-time)": "#2A9D8F", "Hybrid (retrospective)": "#E8A33D"}
FEATURE_NAMES = {   # plain-language names for the explanation chart
    "log_amount": "size of the amount", "hour_of_day": "hour of the day", "day_of_week": "day of the week",
    "is_self": "sent to its own account", "txn_type_code": "payment format", "currency_mismatch": "currency changed",
    "snd_prev_n": "sender's earlier transfers", "snd_prev_mean_log": "sender's usual amount",
    "amt_vs_snd_mean": "amount vs sender's usual", "snd_hours_since_prev": "hours since sender's last transfer",
    "snd_prev_received_n": "money the sender received before", "rcv_prev_n": "receiver's earlier transfers",
    "rcv_hours_since_prev": "hours since receiver's last transfer", "pair_prev_n": "times this pair traded before",
    "retro_loops": "LOOP: number of loops it is in", "retro_min_cv": "LOOP: how equal the loop's amounts are",
    "retro_min_span_h": "LOOP: how fast the loop closed", "retro_flag": "LOOP: passes the hand-made rule",
}


@st.cache_data
def load():
    a = pd.read_parquet(os.path.join(DATA, "alerts.parquet"))
    c = pd.read_parquet(os.path.join(DATA, "cycles.parquet"))
    e = pd.read_parquet(os.path.join(DATA, "explain.parquet"))
    t = pd.read_csv(os.path.join(DATA, "cycle_table.csv"), index_col="pattern_id")
    m = pd.read_csv(os.path.join(DATA, "metrics.csv"))
    p = pd.read_csv(os.path.join(DATA, "paired_test.csv"))
    s = json.load(open(os.path.join(DATA, "summary.json")))
    return a, c, e, t, m, p, s


def when(h):
    return f"day {int(h // 24) + 1}, {int(h % 24):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def money_order(g):
    g = g.sort_values(["TimeHours", "txn_id"]).reset_index(drop=True)
    by_origin = {}
    for i, o in enumerate(g["Origin_ID"]):
        by_origin.setdefault(o, []).append(i)
    order, used = [0], {0}
    while len(order) < len(g):
        nxt = [i for i in by_origin.get(g.loc[order[-1], "Destination_ID"], []) if i not in used]
        if not nxt:
            return None
        order.append(nxt[0]); used.add(nxt[0])
    return g.loc[order].reset_index(drop=True)


def draw_cycle(ring, highlight=None):
    k = len(ring)
    accts = list(ring["Origin_ID"])
    ang = np.pi / 2 - 2 * np.pi * np.arange(k) / k
    pos = {a: np.array([np.cos(t), np.sin(t)]) for a, t in zip(accts, ang)}
    fig = go.Figure()
    for hop, r in enumerate(ring.itertuples(), start=1):
        p0, p1 = pos[r.Origin_ID], pos[r.Destination_ID]
        dv = (p1 - p0) / np.linalg.norm(p1 - p0)
        a, b = p0 + dv * 0.11, p1 - dv * 0.11
        hot = highlight is not None and r.txn_id == highlight
        fig.add_annotation(x=b[0], y=b[1], ax=a[0], ay=a[1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=5 if hot else 3,
                           arrowcolor="#1B2430" if hot else "#D1495B", text="")
        mid = (a + b) / 2
        out = mid / max(np.linalg.norm(mid), 1e-9) * (0.30 if k > 6 else 0.24)
        fig.add_annotation(x=mid[0] + out[0], y=mid[1] + out[1], showarrow=False,
                           text=f"<b>hop {hop}</b><br>${r.Amount:,.0f}<br><span style='font-size:10px'>{when(r.TimeHours)}</span>",
                           font=dict(size=11, color="#1B2430"), bgcolor="rgba(255,255,255,0.85)")
    fig.add_trace(go.Scatter(x=[pos[a][0] for a in accts], y=[pos[a][1] for a in accts], mode="markers+text",
                             marker=dict(size=40, color=["#1B2430"] + ["#E8A33D"] * (k - 1), line=dict(width=2, color="white")),
                             text=[f"A{i + 1}" for i in range(k)], textfont=dict(color="white", size=13),
                             hovertext=accts, hoverinfo="text", showlegend=False))
    lim = 1.75 if k > 6 else 1.6
    fig.update_xaxes(visible=False, range=[-lim, lim])
    fig.update_yaxes(visible=False, range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=560, margin=dict(t=10, b=10, l=10, r=10), template="plotly_white")
    return fig


def explain_chart(row):
    feats = [c for c in row.index if c not in ("txn_id", "(base value)") and not c.startswith("value: ")]
    contrib = row[feats].astype(float).sort_values(key=np.abs, ascending=False).head(8)[::-1]
    labels = [FEATURE_NAMES.get(f, f) for f in contrib.index]
    fig = go.Figure(go.Bar(x=contrib.values, y=labels, orientation="h",
                           marker_color=["#D1495B" if v > 0 else "#2A9D8F" for v in contrib.values]))
    fig.add_vline(x=0, line_color="#1B2430")
    fig.update_layout(height=340, margin=dict(t=10, b=30, l=10, r=10), template="plotly_white",
                      xaxis_title="← pushes towards normal      pushes towards fraud →")
    return fig


# ------------------------------------------------------------------ "Try it yourself": small loop search
EXAMPLES = {
    "Laundering-like ring (5 accounts)": [
        ("Account A", "Account B", 9800, 1, "10:00"), ("Account B", "Account C", 9750, 1, "18:30"),
        ("Account C", "Account D", 9700, 2, "09:15"), ("Account D", "Account E", 9690, 2, "20:00"),
        ("Account E", "Account A", 9650, 3, "08:45"), ("Account A", "Grocery store", 120, 1, "12:00"),
        ("Account C", "Landlord", 800, 2, "11:00")],
    "Normal business loop (uneven amounts)": [
        ("Shop", "Wholesaler", 2000, 1, "09:00"), ("Wholesaler", "Transport co.", 1200, 2, "15:00"),
        ("Transport co.", "Shop", 700, 4, "07:30"), ("Shop", "Electricity", 150, 2, "10:00")],
    "No loop (one account pays many)": [
        ("Company", "Worker 1", 500, 1, "09:00"), ("Company", "Worker 2", 520, 1, "09:05"),
        ("Company", "Worker 3", 480, 1, "09:10"), ("Worker 1", "Shop", 60, 1, "18:00")],
    "Empty table (enter your own)": [],
}


def to_hours(day, hhmm):
    h, m = str(hhmm).strip().split(":")
    return (int(day) - 1) * 24 + int(h) + int(m) / 60


def find_loops(tx, window_h, min_len, max_len, ratio_lo=0.5, ratio_hi=2.0, limit=50):
    """The project's rules, on a small table: every transfer of a loop happens after its first one and within the
    time window, each is between half and double the previous amount, and no account is visited twice."""
    tx = tx.sort_values("TimeHours").reset_index(drop=True)
    out_of = {}
    for i, r in tx.iterrows():
        out_of.setdefault(r["Origin_ID"], []).append(i)
    loops = []

    def extend(path, seen, start):
        if len(loops) >= limit:
            return
        last = tx.loc[path[-1]]
        for j in out_of.get(last["Destination_ID"], []):
            if j <= path[0] or j in path:
                continue
            r = tx.loc[j]
            if r["TimeHours"] - tx.loc[path[0], "TimeHours"] > window_h:
                continue
            if not (ratio_lo <= r["Amount"] / last["Amount"] <= ratio_hi):
                continue
            if r["Destination_ID"] == start:
                if min_len <= len(path) + 1 <= max_len:
                    loops.append(path + [j])
                continue
            if r["Destination_ID"] in seen or len(path) + 1 >= max_len:
                continue
            extend(path + [j], seen | {r["Destination_ID"]}, start)

    for i, r in tx.iterrows():
        if r["Destination_ID"] == r["Origin_ID"]:
            continue
        extend([i], {r["Origin_ID"], r["Destination_ID"]}, r["Origin_ID"])
    return tx, loops


if not os.path.exists(os.path.join(DATA, "alerts.parquet")):
    st.error("The data bundle is missing: put the `app_data` folder (made by Week 5 of the notebook) next to app.py.")
    st.stop()
alerts, cycles, explain, ctab, metrics, paired, summ = load()
budget = int(summ["alert_budget"])

st.sidebar.title("🔁 Circular Fraud Detection")
page = st.sidebar.radio("Go to", ["Overview", "Try it yourself", "Alert queue", "Investigate a cycle", "How it works"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: IBM AML benchmark (HI-Small), test period · {summ['test_transactions']:,} transactions · "
                   f"alert budget {budget:,} per model. IBM's data is synthetic (AMLworld simulator): real bank "
                   "data with laundering labels is not public.")

# --------------------------------------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Can loops in the transaction graph catch money laundering?")
    st.markdown("Money launderers often move money **in a circle** — A → B → C → A — with each transfer looking "
                "ordinary. This project finds such loops in the transaction graph and gives their scores to an "
                "XGBoost model. Three models were compared, identical except for their features:")
    st.markdown("- **Baseline** — ordinary features only (amount, time, account history)\n"
                "- **Hybrid (real-time)** — + loop features known at the moment of the transfer\n"
                "- **Hybrid (retrospective)** — + loop features found when looking back (monitoring)")
    m = metrics[metrics["dataset"] == summ["dataset"]]
    mean = m.groupby("model")[["cycle txns caught", "cycles caught", "fraud caught", "precision"]].mean()
    c1, c2, c3, c4 = st.columns(4)
    b, r = mean.loc["Baseline"], mean.loc["Hybrid (retrospective)"]
    c1.metric("Cycle transactions caught", f"{r['cycle txns caught']:.0%}", f"{(r['cycle txns caught'] - b['cycle txns caught']) * 100:+.0f} pts vs baseline")
    c2.metric("Laundering cycles caught", f"{r['cycles caught']:.1f} of {int(m['cycles total'].max())}", f"{r['cycles caught'] - b['cycles caught']:+.1f} vs baseline")
    c3.metric("Recall (all laundering)", f"{r['fraud caught']:.0%}", f"{(r['fraud caught'] - b['fraud caught']) * 100:+.1f} pts")
    c4.metric("Precision of the alerts", f"{r['precision']:.0%}", f"{(r['precision'] - b['precision']) * 100:+.1f} pts")
    st.caption("Retrospective hybrid vs baseline, mean over 5 seeds, at the same alert budget.")

    left, right = st.columns(2)
    with left:
        st.subheader("The three models")
        fig = go.Figure()
        for col, lab in [("cycle txns caught", "cycle transactions caught"), ("fraud caught", "all laundering caught")]:
            fig.add_trace(go.Bar(name=lab, x=MODELS, y=[mean.loc[x, col] for x in MODELS],
                                 text=[f"{mean.loc[x, col]:.0%}" for x in MODELS], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=380, template="plotly_white",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Is the gain real? (paired test)")
        p = paired[(paired["dataset"] == summ["dataset"])].copy()
        st.dataframe(p[["comparison", "measure", "better on", "worse on", "sign test p", "mean gain", "95% interval", "verdict"]],
                     hide_index=True, width="stretch")
    st.success("**Finding:** in **monitoring** (looking back), loop features catch clearly more laundering cycles — a "
               "statistically proven gain. In **real time** the gain is small and not proven, because a loop only "
               "becomes visible when its last transfer happens.")

# ------------------------------------------------------------------------------------- Try it yourself
elif page == "Try it yourself":
    st.title("Try it yourself: build transactions, watch the loops appear")
    st.markdown("Enter some money transfers (or start from an example), then press **Find loops**. The app runs the "
                "project's loop detector on your table, draws every loop it finds, and compares it with the "
                "**laundering cycles of IBM's data**.")
    ex = st.selectbox("Start from", list(EXAMPLES))
    start = pd.DataFrame(EXAMPLES[ex], columns=["From", "To", "Amount (USD)", "Day", "Time (HH:MM)"])
    table = st.data_editor(start, num_rows="dynamic", key=f"editor_{ex}", hide_index=True, width="stretch",
                           column_config={"Amount (USD)": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
                                          "Day": st.column_config.NumberColumn(min_value=1, step=1)})
    with st.expander("Detector settings (the same as the project uses for IBM)"):
        c1, c2 = st.columns(2)
        window_h = c1.slider("Time window: a loop must close within (hours)", 1, 168, 96)
        min_len, max_len = c2.slider("Number of transfers in a loop", 2, 10, (2, 10))
        st.caption("Also fixed, as in the project: each transfer must be between half and double the previous one, "
                   "and a loop never visits the same account twice.")
    if st.button("🔍 Find loops", type="primary"):
        t = table.dropna(how="all").copy()
        problems = []
        if len(t) < 2:
            problems.append("Enter at least two transfers.")
        if t[["From", "To", "Amount (USD)", "Day", "Time (HH:MM)"]].isna().any().any():
            problems.append("Every row needs From, To, Amount, Day and Time.")
        try:
            t["TimeHours"] = [to_hours(dd, hh) for dd, hh in zip(t["Day"], t["Time (HH:MM)"])]
        except Exception:
            problems.append("Write times as HH:MM, for example 09:30.")
        if problems:
            for p in problems:
                st.error(p)
        else:
            tx = pd.DataFrame({"Origin_ID": t["From"].astype(str).str.strip(), "Destination_ID": t["To"].astype(str).str.strip(),
                               "Amount": t["Amount (USD)"].astype(float), "TimeHours": t["TimeHours"]})
            tx, found = find_loops(tx, window_h, min_len, max_len)
            if not found:
                st.warning("**No loop found.** Possible reasons: the money never comes back to where it started; it "
                           f"takes longer than {window_h} hours; an amount changes by more than half or double from one "
                           "transfer to the next; or the loop would have to visit the same account twice.")
            else:
                st.success(f"**{len(found)} loop{'s' if len(found) > 1 else ''} found.**")
                ref = ctab[["amount_cv", "span_hours", "hops"]]
                for k, idx in enumerate(found, start=1):
                    ring = tx.loc[idx].reset_index().rename(columns={"index": "txn_id"})
                    a = ring["Amount"].to_numpy()
                    cv, span = float(a.std() / a.mean()), float(ring["TimeHours"].max() - ring["TimeHours"].min())
                    more_equal = float((ref["amount_cv"] >= cv).mean())
                    faster = float((ref["span_hours"] >= span).mean())
                    st.subheader(f"Loop {k}: {len(ring)} transfers")
                    left, right = st.columns([3, 2])
                    with left:
                        names = {acc: f"A{i + 1} = {acc}" for i, acc in enumerate(ring["Origin_ID"])}
                        st.plotly_chart(draw_cycle(ring), width="stretch", key=f"loop_{k}")
                        st.caption("Accounts: " + " · ".join(names.values()))
                    with right:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Amount-CV", f"{cv:.3f}", help="0 = all amounts exactly equal")
                        m2.metric("Time span", f"{span:.0f} h")
                        m3.metric("Accounts", f"{len(ring)}")
                        word = "very equal" if cv < 0.05 else "fairly equal" if cv < 0.15 else "uneven"
                        st.markdown(f"- The amounts are **{word}**: more equal than **{more_equal:.0%}** of IBM's "
                                    f"laundering cycles.\n- It closed in **{span:.0f} hours**: faster than "
                                    f"**{faster:.0%}** of IBM's laundering cycles.")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ref["span_hours"], y=ref["amount_cv"], mode="markers", name="IBM laundering cycles",
                                                 marker=dict(size=9, color="#A0AEC0"), text=[f"{int(h)} hops" for h in ref["hops"]]))
                        fig.add_trace(go.Scatter(x=[span], y=[cv], mode="markers", name="your loop",
                                                 marker=dict(size=18, color="#D1495B", symbol="star")))
                        fig.update_layout(height=300, template="plotly_white", margin=dict(t=10, b=10, l=10, r=10),
                                          xaxis_title="time span (hours)", yaxis_title="amount-CV",
                                          legend=dict(orientation="h", y=-0.3))
                        st.plotly_chart(fig, width="stretch", key=f"cmp_{k}")
            st.info("This shows **what the loop detector sees**. It is not a fraud verdict: the full model also uses "
                    "each account's history (thousands of earlier transfers), which a short table cannot provide. "
                    "Loops also happen in honest business — the scores show how *much* a loop looks like laundering.")

# ------------------------------------------------------------------------------------------ Alert queue
elif page == "Alert queue":
    st.title("Alert queue")
    model = st.selectbox("Rank the alerts by", MODELS, index=2)
    only_cyc = st.checkbox("Show only transactions that belong to a laundering cycle", value=False)
    q = alerts[alerts[f"rank: {model}"] <= budget].sort_values(f"rank: {model}")
    if only_cyc:
        q = q[q["pattern_type"] == "CYCLE"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts shown", f"{len(q):,}")
    c2.metric("Really laundering", f"{int(q['IsFraud'].sum()):,}")
    c3.metric("Part of a cycle", f"{int((q['pattern_type'] == 'CYCLE').sum()):,}")
    show = pd.DataFrame({
        "rank": q[f"rank: {model}"], "transaction": q["txn_id"], "amount (USD)": q["Amount"].round(2),
        "when": q["TimeHours"].map(when), "format": q["txn_type"],
        "truth": np.where(q["IsFraud"] == 1, np.where(q["pattern_type"] == "CYCLE", "laundering (cycle)", "laundering"), "normal"),
        "in a loop": np.where(q["retro_loops"] > 0, "yes", "no"),
        **{f"rank by {m.lower()}": q[f"rank: {m}"] for m in MODELS if m != model}})
    st.dataframe(show.head(500), hide_index=True, width="stretch")
    st.caption(f"Showing up to 500 of the {len(q):,} alerts. Rank 1 = the most suspicious of all "
               f"{summ['test_transactions']:,} test transactions.")

# ---------------------------------------------------------------------------------- Investigate a cycle
elif page == "Investigate a cycle":
    st.title("Investigate a laundering cycle")
    caught = {m: ctab[f"caught: {m}"] for m in MODELS}
    def label(pid):
        r = ctab.loc[pid]
        who = ", ".join(n for n, s in [("baseline", caught["Baseline"][pid]), ("real-time", caught["Hybrid (real-time)"][pid]),
                                       ("retrospective", caught["Hybrid (retrospective)"][pid])] if s) or "no model"
        return f"cycle {pid} · {int(r['hops'])} hops · caught by: {who}"
    order = ctab.sort_values(["hops", "rank: Hybrid (retrospective)"], ascending=[False, True]).index
    pid = st.selectbox("Choose a test cycle", order, format_func=label)
    ring = money_order(cycles[cycles["pattern_id"] == pid])
    r = ctab.loc[pid]
    st.markdown(f"**{int(r['hops'])} accounts, closed in {r['span_hours']:.0f} hours, amount-CV {r['amount_cv']:.3f}** "
                "(0 = all amounts equal).")
    cols = st.columns(3)
    for col, m in zip(cols, MODELS):
        col.metric(m, f"rank {int(r[f'rank: {m}']):,}", "caught" if r[f"caught: {m}"] else "missed",
                   delta_color="normal" if r[f"caught: {m}"] else "inverse")
    if ring is None:
        st.warning("This cycle visits an account twice, so it cannot be drawn as one circle.")
    else:
        # options are hop numbers; the key depends on the cycle, so switching cycles starts fresh
        hop_no = st.select_slider("Pick a transfer to explain", options=list(range(1, len(ring) + 1)),
                                  format_func=lambda h: f"hop {h}", key=f"hop_of_cycle_{pid}")
        hop_txn = int(ring["txn_id"].iloc[hop_no - 1])
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(draw_cycle(ring, highlight=hop_txn), width="stretch")
            st.caption("A1 (dark) is where the money starts and ends. The selected transfer is drawn in black.")
        with right:
            row = explain[explain["txn_id"] == hop_txn]
            hop_info = ring[ring["txn_id"] == hop_txn].iloc[0]
            st.markdown(f"**Hop {hop_no}:** ${hop_info['Amount']:,.0f} on {when(hop_info['TimeHours'])}")
            if len(row):
                st.markdown("Why the **retrospective hybrid** model scored it the way it did:")
                st.plotly_chart(explain_chart(row.iloc[0]), width="stretch")
                st.caption("Red bars push the score towards 'laundering', green towards 'normal'. "
                           "Features starting with LOOP come from the loop detector.")
            else:
                st.info("This transfer happened before the test period (history), so the models did not score it.")
        st.dataframe(pd.DataFrame({"hop": range(1, len(ring) + 1), "from": [f"A{i + 1}" for i in range(len(ring))],
                                   "to": [f"A{(i + 1) % len(ring) + 1}" for i in range(len(ring))],
                                   "amount (USD)": ring["Amount"].round(2), "when": ring["TimeHours"].map(when),
                                   "format": ring["txn_type"],
                                   "test period": np.where(ring["test_period"], "yes", "no (history)")}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------- How it works
else:
    st.title("How it works")
    st.markdown("""
1. **Data.** Three datasets: a small synthetic one with 15 planted rings (to test the method), PaySim
   (a negative control without circular fraud), and the IBM AML benchmark (the main test).
2. **Loop detector.** (Try it on the *Try it yourself* page.) A search written for this project finds money loops in the transaction graph — accounts
   are the points, transfers the arrows — within a time window, and scores each loop: *amount-CV* (how equal the
   amounts are), *time span* and *length*. It found 84% of IBM's test laundering cycles.
3. **No peeking.** The earlier 70% of the data is used to learn, the later 30% only to judge; every feature uses
   only information available at its time, and this is proven by tests.
4. **Models.** Three XGBoost models, identical except for their features, each trained with 5 seeds and compared
   at the same number of alerts, with a paired statistical test over the cycles.
5. **Result.** Looking back (monitoring), loop features catch clearly more laundering cycles — a proven gain.
   In real time the gain is small and not proven.
""")
    st.info("This app only reads a small, pre-computed data bundle; all heavy work happens in the project notebook.")

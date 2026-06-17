"""
Weather-Aware Pace Predictor  --  Streamlit front-end.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import date, time as dtime, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import model as M
import physiology as phys
import storage

# --------------------------------------------------------------------------
# Page setup & light styling
# --------------------------------------------------------------------------

st.set_page_config(page_title="Pace Predictor", page_icon="🏃", layout="wide")

# Use the managed Postgres database in the cloud (set as a Streamlit secret);
# fall back to a local SQLite file when running offline.
try:
    if "DATABASE_URL" in st.secrets:
        os.environ.setdefault("DATABASE_URL", str(st.secrets["DATABASE_URL"]))
except Exception:
    pass

storage.init_db()

st.markdown(
    """
    <style>
      .pace-card {
        border-radius: 16px; padding: 22px 24px; color: #fff;
        box-shadow: 0 6px 20px rgba(0,0,0,.12);
      }
      .pace-card.vo2  { background: linear-gradient(135deg,#ff6a3d,#ff3d77); }
      .pace-card.thr  { background: linear-gradient(135deg,#2b6cff,#21c1c9); }
      .pace-card .lbl { font-size: .85rem; letter-spacing:.08em; text-transform:uppercase; opacity:.9;}
      .pace-card .big { font-size: 3.2rem; font-weight: 800; line-height: 1.05; }
      .pace-card .unit{ font-size: 1.1rem; font-weight: 500; opacity:.9; }
      .pace-card .rng { font-size: .95rem; opacity:.92; margin-top: 4px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Y-axis tick formatter: seconds/km -> "m:ss"
PACE_LABEL = (
    "floor(datum.value/60) + ':' + "
    "(datum.value % 60 < 10 ? '0' : '') + floor(datum.value % 60)"
)


# --------------------------------------------------------------------------
# Model fit (cached on a signature of the user's data)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_fit(user_sig, baseline_pace, baseline_type, lt_fraction, sessions):
    cfg = M.config_from_baseline(baseline_pace, baseline_type, lt_fraction)
    return M.fit(sessions, cfg)


def fit_for_user(user):
    sessions = storage.list_sessions(user["id"])
    # Include time-of-day in the signature so editing a time invalidates cache.
    sig = (user["id"], len(sessions),
           tuple((s["id"], s.get("time")) for s in sessions),
           user["baseline_pace"], user["baseline_type"], user["lt_fraction"])
    return get_fit(sig, user["baseline_pace"], user["baseline_type"],
                   user["lt_fraction"], sessions), sessions


def pace_card(container, kind, title, p):
    cls = "vo2" if kind == "vo2max" else "thr"
    container.markdown(
        f"""
        <div class="pace-card {cls}">
          <div class="lbl">{title}</div>
          <div class="big">{phys.format_pace(p['pace_sec'])}
             <span class="unit">/ km</span></div>
          <div class="rng">likely range {phys.format_pace(p['pace_lo'])} –
             {phys.format_pace(p['pace_hi'])} /km</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar: profile selection / creation
# --------------------------------------------------------------------------

st.sidebar.title("🏃 Pace Predictor")
users = storage.list_users()
names = {u["name"]: u for u in users}

choice = st.sidebar.selectbox(
    "Profile", options=["➕ New profile…"] + list(names.keys()),
    index=1 if users else 0,
)

if choice == "➕ New profile…":
    st.title("Create your profile")
    st.write(
        "One quick calibration. Tell me a pace you can currently hold **in cool "
        "conditions (~10–12 °C)** for one of the two interval types — that becomes "
        "your starting fitness. Everything else is learned from the sessions you log."
    )
    with st.form("new_user"):
        name = st.text_input("Name")
        c1, c2 = st.columns(2)
        b_type = c1.selectbox("Reference session type", ["vo2max", "threshold"],
                              format_func=lambda x: "VO₂max intervals" if x == "vo2max" else "Threshold")
        b_pace = c2.text_input("Pace in cool conditions (m:ss /km)", "4:00")
        lt = st.slider(
            "Your threshold ÷ VO₂max velocity ratio", 0.82, 0.95,
            phys.DEFAULT_LT_FRACTION, 0.005,
            help="Physiological link between the two paces. ~0.90 is typical; "
                 "leave it unless you know yours. The app refines it once you've "
                 "logged both session types.",
        )
        submitted = st.form_submit_button("Create profile", type="primary")
    if submitted:
        if not name.strip():
            st.error("Please enter a name.")
        elif name.strip() in names:
            st.error("That name already exists.")
        else:
            try:
                storage.create_user(name, phys.parse_pace(b_pace), b_type, lt)
                st.success(f"Welcome, {name}! Select your profile in the sidebar.")
                st.rerun()
            except ValueError:
                st.error("Pace must look like 4:00 or 3:45.")
    st.stop()

user = names[choice]
fit, sessions = fit_for_user(user)

st.sidebar.caption(
    f"Baseline: **{phys.format_pace(user['baseline_pace'])}/km** "
    f"({'VO₂max' if user['baseline_type'] == 'vo2max' else 'threshold'}, cool)  \n"
    f"Sessions logged: **{len(sessions)}**"
)

# --------------------------------------------------------------------------
# Main tabs
# --------------------------------------------------------------------------

tab_predict, tab_log, tab_insights, tab_profile = st.tabs(
    ["🔮 Predict", "➕ Log session", "📈 Insights", "⚙️ Profile"]
)


# ---- Weather input widget (shared) ---------------------------------------

def weather_inputs(key: str, default_temp=15.0):
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.number_input("Temperature °C", -10.0, 45.0, default_temp, 0.5, key=f"{key}_t")
    sky = c2.selectbox("Sky", phys.SKY_OPTIONS, index=2, key=f"{key}_s")
    rain = c3.selectbox("Rain", phys.RAIN_OPTIONS, index=0, key=f"{key}_r")
    hum = c4.slider("Humidity %", 10, 100, 55, key=f"{key}_h")
    return phys.Weather(temp_c=temp, sky=sky, rain=rain, humidity=hum)


# ======================  PREDICT  =========================================
with tab_predict:
    st.subheader("What should I run today?")
    cda, cti = st.columns(2)
    d = cda.date_input("Day", value=date.today())
    t = cti.time_input("Time of day", value=dtime(18, 0), step=1800,
                       help="Used for the sun's height → radiant heat "
                            "(Stuttgart). Noon sun loads more than evening sun.")
    w = weather_inputs("pred", default_temp=15.0)
    w.date, w.time = d, t

    pred = M.predict(fit, d, w)
    left, right = st.columns(2)
    pace_card(left, "vo2max", "VO₂max interval pace", pred["vo2max"])
    pace_card(right, "threshold", "Threshold pace", pred["threshold"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Effective temperature", f"{pred['effective_temp']:.0f} °C",
              help="Raw temperature adjusted for sun, humidity and rain.")
    m2.metric("Heat slowdown vs ideal", f"{pred['heat_slowdown_pct']:.1f} %")
    m3.metric("Data personalisation",
              "prior only" if len(sessions) < 4 else f"{len(sessions)} sessions")

    if len(sessions) < 4:
        st.info("Still mostly using your baseline + typical physiology. "
                "Log a handful of sessions and the heat response personalises to you.")

    # Pace-vs-temperature curve at today's fitness.
    temps = np.arange(0, 35.25, 0.5)
    curve = M.heat_curve(fit, d, temps, sky=w.sky, rain=w.rain,
                         humidity=w.humidity, tod=t)
    cdf = pd.DataFrame({
        "Temperature": np.concatenate([temps, temps]),
        "pace": np.concatenate([curve["vo2max"], curve["threshold"]]),
        "Type": ["VO₂max"] * len(temps) + ["Threshold"] * len(temps),
    })
    cdf["Pace"] = cdf["pace"].map(phys.format_pace)

    # Adaptive y-range: hug the data with a little padding instead of zero.
    pmin, pmax = float(cdf["pace"].min()), float(cdf["pace"].max())
    pad = max(5.0, 0.08 * (pmax - pmin))
    ydomain = [pmin - pad, pmax + pad]

    color = alt.Color("Type:N", title=None, scale=alt.Scale(
        domain=["VO₂max", "Threshold"], range=["#ff3d77", "#2b6cff"]))
    xenc = alt.X("Temperature:Q", title="Temperature (°C)",
                 scale=alt.Scale(domain=[float(temps[0]), float(temps[-1])], nice=False))
    yenc = alt.Y("pace:Q", title="pace (min/km)",
                 scale=alt.Scale(domain=ydomain, reverse=True, nice=False),
                 axis=alt.Axis(labelExpr=PACE_LABEL))

    base = alt.Chart(cdf)
    lines = base.mark_line(strokeWidth=3).encode(x=xenc, y=yenc, color=color)

    # Static reference: the temperature you entered above.
    ref = alt.Chart(pd.DataFrame({"Temperature": [float(w.temp_c)]})).mark_rule(
        strokeDash=[5, 4], color="#9aa0a6").encode(x=xenc)

    # Interactive crosshair: follows the cursor and reads off both paces.
    hover = alt.selection_point(nearest=True, on="pointerover",
                                fields=["Temperature"], empty=False)
    selectors = base.mark_point().encode(x=xenc, opacity=alt.value(0)).add_params(hover)
    hl_rule = base.mark_rule(color="#bbbbbb", strokeWidth=1).encode(
        x=xenc).transform_filter(hover)
    hl_pts = lines.mark_point(size=80, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Temperature:Q", title="°C", format=".1f"),
                 alt.Tooltip("Type:N"), alt.Tooltip("Pace:N", title="pace /km")])
    hl_txt = lines.mark_text(align="left", dx=9, dy=-9, fontWeight="bold").encode(
        text=alt.condition(hover, "Pace:N", alt.value("")))

    chart = (ref + lines + selectors + hl_rule + hl_pts + hl_txt).properties(
        height=340, title="Predicted pace across temperatures (at today's fitness)")
    st.altair_chart(chart, width="stretch")
    st.caption("Hover to read off both paces at any temperature. Uses today's "
               "sky/rain/humidity; dashed line = the temperature you entered. "
               "Higher on the chart = faster.")


# ======================  LOG SESSION  =====================================
with tab_log:
    st.subheader("Log an interval session")
    # Plain widgets (not st.form): pressing Enter only commits the field and
    # reruns -- it never triggers the save. Only the button below saves.
    c1, c2, c3, c4 = st.columns(4)
    sd = c1.date_input("Date", value=date.today(), key="log_date")
    stime = c2.time_input("Time", value=dtime(18, 0), step=1800, key="log_time")
    stype = c3.selectbox("Type", ["vo2max", "threshold"],
                         format_func=lambda x: "VO₂max" if x == "vo2max" else "Threshold",
                         key="log_type")
    space = c4.text_input("Average pace (m:ss /km)", key="log_pace")
    lw = weather_inputs("log", default_temp=15.0)
    notes = st.text_input("Notes (optional)", key="log_notes")
    if st.button("Save session", type="primary", key="log_save"):
        try:
            pace_sec = phys.parse_pace(space)
            storage.add_session(user["id"], sd, stype, pace_sec, lw.temp_c,
                                lw.sky, lw.rain, lw.humidity,
                                time_of_day=stime.strftime("%H:%M"), notes=notes)
            # Clear the typed fields so the empty form confirms the save,
            # then rerun and show the confirmation once.
            st.session_state.pop("log_pace", None)
            st.session_state.pop("log_notes", None)
            st.session_state["log_saved_msg"] = (
                f"Saved {('VO₂max' if stype == 'vo2max' else 'threshold')} "
                f"session at {phys.format_pace(pace_sec)}/km.")
            st.rerun()
        except (ValueError, ZeroDivisionError):
            st.error("Enter the pace as m:ss, e.g. 3:45.")

    if "log_saved_msg" in st.session_state:
        st.success(st.session_state.pop("log_saved_msg"))

    st.divider()
    st.markdown("##### Recent sessions")
    if sessions:
        df = pd.DataFrame([{
            "Date": s["date"], "Time": s["time"] or "—", "Type": s["session_type"],
            "Pace": phys.format_pace(s["pace_sec"]),
            "°C": s["temp_c"], "Sky": s["sky"], "Rain": s["rain"],
            "RH%": int(s["humidity"]), "Notes": s["notes"], "_id": s["id"],
        } for s in reversed(sessions)])
        st.dataframe(df.drop(columns="_id"), width='stretch', hide_index=True)

        n_missing = sum(1 for s in sessions if not s["time"])
        if n_missing:
            st.caption(f"⏱️ {n_missing} session(s) have no time of day yet — "
                       "add it below so the sun/radiation effect applies to them.")

        def _label(s):
            return (f"{s['date']} · {s['session_type']} · "
                    f"{phys.format_pace(s['pace_sec'])} · "
                    f"{s['time'] or 'no time'} (#{s['id']})")

        by_label = {_label(s): s for s in reversed(sessions)}

        with st.expander("✏️ Edit a session", expanded=bool(n_missing)):
            pick = st.selectbox("Session", list(by_label.keys()), key="edit_pick")
            s = by_label[pick]
            sid = s["id"]  # widget keys include the id so the fields re-fill
                           # with the chosen session's values when you switch.
            e1, e2, e3, e4 = st.columns(4)
            ed = e1.date_input("Date", value=s["date"], key=f"ed_date_{sid}")
            et = e2.time_input("Time", step=1800, key=f"ed_time_{sid}",
                               value=phys.parse_time(s["time"]) or dtime(18, 0))
            etype = e3.selectbox(
                "Type", ["vo2max", "threshold"],
                index=0 if s["session_type"] == "vo2max" else 1,
                format_func=lambda x: "VO₂max" if x == "vo2max" else "Threshold",
                key=f"ed_type_{sid}")
            epace = e4.text_input("Pace (m:ss /km)", key=f"ed_pace_{sid}",
                                  value=phys.format_pace(s["pace_sec"]))
            w1, w2, w3, w4 = st.columns(4)
            etemp = w1.number_input("Temperature °C", -10.0, 45.0,
                                    float(s["temp_c"]), 0.5, key=f"ed_temp_{sid}")
            esky = w2.selectbox("Sky", phys.SKY_OPTIONS, key=f"ed_sky_{sid}",
                                index=phys.SKY_OPTIONS.index(s["sky"])
                                if s["sky"] in phys.SKY_OPTIONS else 2)
            erain = w3.selectbox("Rain", phys.RAIN_OPTIONS, key=f"ed_rain_{sid}",
                                 index=phys.RAIN_OPTIONS.index(s["rain"])
                                 if s["rain"] in phys.RAIN_OPTIONS else 0)
            ehum = w4.slider("Humidity %", 10, 100, int(s["humidity"]),
                             key=f"ed_hum_{sid}")
            enotes = st.text_input("Notes", value=s["notes"], key=f"ed_notes_{sid}")
            if st.button("Save changes", type="primary", key=f"ed_save_{sid}"):
                try:
                    storage.update_session(sid, ed, etype, phys.parse_pace(epace),
                                           etemp, esky, erain, ehum,
                                           et.strftime("%H:%M"), enotes)
                    st.session_state.pop("edit_pick", None)  # label may have changed
                    st.session_state["log_saved_msg"] = f"Updated the {ed} session."
                    st.rerun()
                except (ValueError, ZeroDivisionError):
                    st.error("Pace must look like 3:45.")

        with st.expander("🗑️ Delete a session"):
            to_del = st.selectbox("Session", ["—"] + list(by_label.keys()),
                                  key="del_pick")
            if to_del != "—" and st.button("Delete selected", type="secondary"):
                storage.delete_session(by_label[to_del]["id"])
                st.rerun()
    else:
        st.caption("No sessions yet.")


# ======================  INSIGHTS  ========================================
with tab_insights:
    st.subheader("Your model")
    if len(sessions) < 2:
        st.info("Log at least a couple of sessions to see your fitness trend and "
                "heat response.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Threshold ÷ VO₂max", f"{M.learned_lt_fraction(fit):.3f}",
                  help="Estimated from your data once both session types exist; "
                       "otherwise held at your profile value.")
        k2.metric("Slowdown at 25 °C", f"{M.heat_sensitivity_at(fit, 25):.1f} %")
        k3.metric("Session noise (±1σ)", f"{100*(np.exp(fit.sigma)-1):.1f} %",
                  help="Run-to-run scatter the model can't explain by weather/fitness.")

        # Fitness trajectory (ideal-conditions vVO2max pace) + normalized points.
        days, pace, lo, hi = M.fitness_trajectory(fit)
        base = sessions[0]["date"]  # == fit.day0
        traj = pd.DataFrame({
            "Date": [base + timedelta(days=int(x)) for x in days],
            "pace": pace, "lo": lo, "hi": hi})
        ndays, npace = M.weather_normalized_paces(fit, sessions)
        pts = pd.DataFrame({
            "Date": [base + timedelta(days=int(x)) for x in ndays],
            "pace": npace,
            "Type": [("VO₂max" if s["session_type"] == "vo2max" else "Threshold-adj")
                     for s in sessions]})

        band = alt.Chart(traj).mark_area(opacity=0.18, color="#ff3d77").encode(
            x="Date:T",
            y=alt.Y("lo:Q", title="ideal-weather VO₂max pace (min/km)",
                    scale=alt.Scale(reverse=True, zero=False),
                    axis=alt.Axis(labelExpr=PACE_LABEL)),
            y2="hi:Q")
        fline = alt.Chart(traj).mark_line(strokeWidth=3, color="#ff3d77").encode(
            x="Date:T", y=alt.Y("pace:Q", scale=alt.Scale(reverse=True, zero=False)))
        scat = alt.Chart(pts).mark_circle(size=70, opacity=0.65).encode(
            x="Date:T",
            y=alt.Y("pace:Q", scale=alt.Scale(reverse=True, zero=False)),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["VO₂max", "Threshold-adj"], range=["#ff3d77", "#2b6cff"])),
            tooltip=["Date:T", "Type:N"])
        st.altair_chart((band + fline + scat).properties(
            height=340, title="Fitness over time (weather removed)"),
            width='stretch')
        st.caption("Line = estimated fitness (faster = higher). Dots = your "
                   "sessions with the estimated weather effect subtracted, so they "
                   "should scatter around the line regardless of the day's weather.")


# ======================  PROFILE  =========================================
with tab_profile:
    st.subheader("Profile settings")
    with st.form("edit_profile"):
        c1, c2 = st.columns(2)
        b_pace = c1.text_input("Baseline pace (cool, m:ss /km)",
                               phys.format_pace(user["baseline_pace"]))
        b_type = c2.selectbox("Baseline type", ["vo2max", "threshold"],
                              index=0 if user["baseline_type"] == "vo2max" else 1,
                              format_func=lambda x: "VO₂max" if x == "vo2max" else "Threshold")
        lt = st.slider("Threshold ÷ VO₂max ratio", 0.82, 0.95,
                       float(user["lt_fraction"]), 0.005)
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        try:
            storage.update_user(user["id"], phys.parse_pace(b_pace), b_type, lt)
            st.success("Updated.")
            st.rerun()
        except ValueError:
            st.error("Pace must look like 4:00.")

    st.divider()
    with st.expander("Danger zone"):
        st.write(f"Delete **{user['name']}** and all their sessions.")
        if st.button("Delete this profile", type="secondary"):
            storage.delete_user(user["id"])
            st.rerun()

# Weather-Aware Pace Predictor

A small personal app that predicts your **threshold pace** and **VO₂max interval
pace** for a given day, accounting for the weather. You log your interval
sessions (pace + weather + type); for a new day you enter the forecast and it
tells you the pace to aim for — for *both* paces, even if you've only ever
logged one type.

## Quick start

```bash
./run.sh
```

That creates a virtualenv on first run and launches the app in your browser
(`http://localhost:8501`). Or manually:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/streamlit run app.py
```

## How to use it

1. **Create a profile**: just pick a name + password — no fitness numbers
   needed. Next time you **log in** by name + password. Profiles are private and
   not listed; each person's heat response and fitness are learned separately.
2. **Log session** after each interval workout: date, time, type, average pace,
   and the weather (temperature, sky, rain, humidity). Your very first session
   is what the model takes its starting point from — log one before predicting.
3. **Predict**: pick a day + time, enter the weather, and read off the two paces
   with a likely range, plus a pace-vs-temperature curve.
4. **Insights**: see your fitness trend (with the weather effect removed) and
   your personal heat sensitivity.

Passwords are stored only as salted **PBKDF2-HMAC-SHA256** hashes. Data lives in
SQLite locally, or Postgres when `DATABASE_URL` is set (deployment).

## The model (short version)

Performance is modelled in **log-velocity**, where fitness and the weather
penalty act additively:

```
log v  =  fitness(date)                # slow latent random walk
        + offset·1[threshold]          # physiological threshold↔VO₂max link
        + heat_penalty(weather)        # learned per-user heat sensitivity
        + noise
```

* **`fitness(date)`** is a latent state that drifts slowly (a random walk in
  calendar time). It's there only to stop slow seasonal fitness gains from being
  mistaken for a weather effect — you don't predict it, but ignoring it would
  bias everything. Because it's constrained to change slowly, the model can
  separate it from the *fast* day-to-day weather swings. Its starting level is
  read straight off your logged sessions (weather-corrected back to ideal
  conditions) — you're never asked for a baseline pace.
* **One latent fitness drives both paces** through a fixed physiological ratio
  (vLT ≈ 0.90·vVO₂max, refined from your data), so threshold-only logging still
  yields a VO₂max prediction.
* **Weather → effective temperature** (sun/humidity/rain folded in with fixed
  physiological constants); the model only learns your overall heat sensitivity
  (two coefficients), which keeps it estimable on ~50 sessions.
* **Solar radiation by time of day**: the sun's elevation is computed from the
  date and clock time at a fixed location (**Stuttgart, Germany**) and added as
  radiant heat load, scaled by cloud cover — so a noon run loads more than an
  evening run at the same air temperature. Time of day is optional (older
  sessions fall back to a mid-day estimate) and can be edited per session.

This is your state-space model written in its batch (smoother) form. Because
every term is linear-Gaussian, inference is the exact posterior of a Bayesian
linear regression — no EM or particle filter, robust on little data, and it
returns full uncertainty. See `model.py` for the details.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `model.py` | Bayesian dynamic linear model (fit + predict) |
| `physiology.py` | pace↔velocity, weather→effective-temp, threshold↔VO₂max |
| `storage.py` | SQLite persistence (users + sessions) |

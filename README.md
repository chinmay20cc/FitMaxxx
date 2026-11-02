# FitMaxxx — Personal Fitness Companion

FitMaxxx is a student-built fitness analytics prototype. It combines smartphone activity recognition, simulated wearable data, a personalized coaching layer, and local session history in one Streamlit experience.

## Product scope

This release focuses on five pieces:

1. **FitMaxxx identity and visual refresh** — warm, editorial design with a forest-green, soft-lime and off-white palette.
2. **Dashboard** — daily overview, movement trends, wearable studio, activity lab and model lab.
3. **Personal coach** — uses the user's goal and aggregate activity metrics. Claude is optional; a local fallback works without an API key.
4. **History and personalization** — profile preferences and session summaries are stored in a local SQLite database.
5. **GitHub and deployment readiness** — requirements, tests and a GitHub Actions workflow.

## Wearables are simulated in this release

The Wearable Studio provides demo profiles for Apple Watch, Galaxy Watch, Garmin and Fitbit-style devices. All numbers are generated locally for demonstration. No watch, account, Health Connect, HealthKit, Bluetooth or vendor API is connected. This keeps the current project easy to run and demonstrate. A mobile companion app and real integrations can be a later phase.

## Machine-learning models

The existing project models remain separate:

- SVM benchmark trained on the 561-feature UCI HAR dataset.
- Random Forest trained on 44 statistical features from raw sensor signals.
- Orientation-robust Random Forest trained on 42 gravity-relative features.

The Activity Lab can analyse the built-in demo session, a Sensor Logger ZIP, or two CSV files. The Wearable Studio is a separate simulator and does not claim to be a live hardware connection.

## AI coach

The coach receives profile fields and aggregate metrics only. Raw sensor arrays and uploaded files are not sent to Anthropic. Without an API key, FitMaxxx uses its local personalized insight generator. To enable Claude locally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then add your key:

```toml
ANTHROPIC_API_KEY = "your-key"
ANTHROPIC_MODEL = "claude-haiku-5-5"
```

The Streamlit UI does not display setup instructions or secret names. Keep `secrets.toml` private and never commit it. The default model ID is Claude Haiku 5.5; check the provider's current model catalogue before changing it.

## Run locally

Use Python 3.12 in a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-train.txt
streamlit run app/streamlit_app.py
```

The project root should contain the trained models and existing UCI dataset files:

```text
models/best_model.joblib
models/raw_model.joblib
models/robust_model.joblib
data/UCI HAR Dataset/
app/sample_data/
```

If the demo session files are not present, generate them using the existing project script `python src/build_demo_sample.py`.

## Tests

```bash
python -m pytest tests/ -v
```

GitHub Actions runs the test suite on pushes and pull requests.

## Local history and privacy

FitMaxxx stores profile preferences and session summaries in `data/fitmaxxx.sqlite3` when running locally. Raw phone recordings are not written to the history database. The entire `data/` folder is ignored by Git.

For a public Streamlit demo, set `FITMAXXX_PUBLIC_DEMO = "1"` in Streamlit secrets. This keeps profile and session history in each browser session instead of a shared SQLite file. Session-only history resets when that browser session ends.

## Important limitations

- Wearable readings are simulated, not fetched from physical devices.
- Activity predictions are experimental and depend on phone placement and sensor quality.
- The six-class UCI HAR dataset does not include running.
- Calorie values and the movement score are estimates, not validated measurements.
- FitMaxxx is a general wellness prototype, not a medical product.

## Later roadmap

A mobile companion app can later replace the simulator with authorized Health Connect / HealthKit or vendor data adapters. That is intentionally outside this release.

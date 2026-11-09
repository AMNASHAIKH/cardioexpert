# CardioExpert – Heart Disease Expert System

Rule-based expert system (78 IF-THEN rules) that detects **Coronary Artery Disease, Heart Failure
and Cardiac Arrhythmia**, gives an urgency decision and recommendations. UI built with Streamlit.

## Files
| File | Purpose |
|---|---|
| `engine.py` | Knowledge base (78 rules) + forward-chaining inference engine with certainty factors |
| `app.py` | Streamlit user interface |
| `samples.py` | Default values and 5 demo patients |
| `test_engine.py` | Tests the engine on the demo patients |
| `requirements.txt` | Python packages for deployment |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
python test_engine.py      # optional: run tests
```

## Make it live (Streamlit Community Cloud – free)
1. Create a public GitHub repository and upload all files in this folder (keep `app.py` at the top level).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repository, branch `main`, main file `app.py`.
4. Click **Deploy**. After 1–2 minutes you get a link like `https://<name>.streamlit.app`.
5. Open the link on your phone (mobile data) to confirm it works over the internet.

*Educational project only – not a medical device.*

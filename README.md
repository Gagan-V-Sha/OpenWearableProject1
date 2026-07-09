# OpenWearable

A **Human-Centered AI (HCAI)** backend that turns wearable data into simple, explainable daily recovery guidance.

## The problem

Smartwatches track sleep, heart rate, steps, and activity — but most people don't want raw numbers. They want one clear answer:

> *Should I rest today, take it easy, or train hard?*

A black-box model isn't enough. Users need to **understand and trust** the recommendation.

## What we built

OpenWearable reads **7 days of Fitbit-style data** and returns:

| Recommendation | Meaning |
|----------------|---------|
| **Rest Day** | Body needs recovery |
| **Light Activity** | Okay for easy exercise |
| **Intensive Training** | Well recovered — train hard |

Each result includes a **recovery score (0–100)** and an explanation of **why**, not just a label.

## How it works

1. **Rule engine** — transparent scoring from sleep, resting HR, HRV, steps, and activity.
2. **ML surrogate (XGBoost)** — learns the rules (~88% agreement) for fast predictions while staying checkable.
3. **SHAP** — shows which features moved the score up or down.
4. **DiCE** — suggests small lifestyle changes ("what if you slept more?").
5. **Fairness audit** — checks whether recommendations differ unfairly across groups.

## Data

Built on the public **LifeSnaps** dataset:

- **71 users**, **~7,410 daily logs**
- Fitbit measurements + **SEMA** mood surveys (how tired/rested people felt)

## Why it's HCAI

- Humans can **read the rules**
- Humans can **see the evidence**
- Humans can **challenge the output** with what-if and counterfactual tools

The goal isn't just prediction — it's **recovery guidance people can trust**.

## Tech stack

- Python, FastAPI, pandas, scikit-learn, XGBoost, SHAP, DiCE
- Deployed on Vercel

<<<<<<< HEAD
```

=======
## Project structure

```
api.py              # FastAPI server
pipeline.py         # Full data + training pipeline
rule_engine.py      # Recovery scoring rules
data/
  lifesnaps/        # Raw LifeSnaps data
  processed/        # Processed CSVs for the API
models/             # Trained models
vercel/             # Deployment config
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Main API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/users` | List users |
| `GET /api/dashboard/{user_id}` | Score, metrics, trends |
| `POST /api/whatif` | Slider counterfactual |
| `GET /api/suggestions/{user_id}` | DiCE suggestions |
| `POST /api/ask` | Plain-language Q&A |

Set `GEMINI_API_KEY` in `.env` (see `vercel/environment.example`) for the Ask feature.

---

**OpenWearable** — wearable recovery intelligence you can explain, check, and trust.
>>>>>>> 2d9df60 (Update project files)

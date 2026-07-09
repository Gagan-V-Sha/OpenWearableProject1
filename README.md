OpenWearable
A Human-Centered AI (HCAI) backend that turns wearable data into simple, explainable daily recovery guidance.

The problem
Smartwatches track sleep, heart rate, steps, and activity — but most people don’t want raw numbers. They want one clear answer:

Should I rest today, take it easy, or train hard?

A black-box model isn’t enough. Users need to understand and trust the recommendation.

What we built
OpenWearable reads 7 days of Fitbit-style data and returns:

Recommendation	Meaning
Rest Day
Body needs recovery
Light Activity
Okay for easy exercise
Intensive Training
Well recovered — train hard
Each result includes a recovery score (0–100) and an explanation of why, not just a label.

How it works
Rule engine — transparent scoring from sleep, resting HR, HRV, steps, and activity.
ML surrogate (XGBoost) — learns the rules (~88% agreement) for fast predictions while staying checkable.
SHAP — shows which features moved the score up or down.
DiCE — suggests small lifestyle changes (“what if you slept more?”).
Fairness audit — checks whether recommendations differ unfairly across groups.
Data
Built on the public LifeSnaps dataset:

71 users, ~7,410 daily logs
Fitbit measurements + SEMA mood surveys (how tired/rested people felt)
Why it’s HCAI
Humans can read the rules
Humans can see the evidence
Humans can challenge the output with what-if and counterfactual tools
The goal isn’t just prediction — it’s recovery guidance people can trust.

Tech stack
Python, FastAPI, pandas, scikit-learn, XGBoost, SHAP, DiCE
Deployed on Vercel

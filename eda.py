

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
PLOTS_DIR = DATA / "eda_plots"

DAILY_PATH = PROCESSED / "combined_daily.csv"
DEMO_PATH = PROCESSED / "user_demographics.csv"
PROFILE_PATH = PROCESSED / "profiles_7day.csv"

def setup_directories():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {PLOTS_DIR}")

def run_demographic_eda():
    print("\n" + "="*50)
    print("1. DEMOGRAPHICS & FAIRNESS BASELINE")
    print("="*50)
    if not DEMO_PATH.exists():
        print("Demographics file not found.")
        return

    df = pd.read_csv(DEMO_PATH)
    print(f"Total Users: {len(df)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if 'gender' in df.columns:
        sns.countplot(data=df, x='gender', ax=axes[0], order=df['gender'].value_counts().index)
        axes[0].set_title('Gender Distribution')
        print(df['gender'].value_counts(normalize=True).map('{:.1%}'.format).to_string())

    if 'age_group' in df.columns:
        sns.countplot(data=df, x='age_group', ax=axes[1], order=df['age_group'].value_counts().index)
        axes[1].set_title('Age Group Distribution')
        print(df['age_group'].value_counts(normalize=True).map('{:.1%}'.format).to_string())

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'demographics_distribution.png', dpi=300)
    plt.close()

def run_daily_eda():
    print("\n" + "="*50)
    print("2. RAW DAILY PHYSIOLOGICAL DATA")
    print("="*50)
    if not DAILY_PATH.exists():
        print("Daily dataset not found.")
        return

    df = pd.read_csv(DAILY_PATH)
    print(f"Total Daily Logs: {len(df)}")

    features = ['sleep_hours', 'resting_hr', 'steps', 'sleep_efficiency', 'active_minutes', 'rmssd']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        if feature in df.columns:
            sns.histplot(df[feature].dropna(), bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'daily_features_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Daily Features Correlation Matrix')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'daily_features_correlation.png', dpi=300)
    plt.close()

def run_profile_eda():
    print("\n" + "="*50)
    print("3. MACHINE LEARNING 7-DAY PROFILES")
    print("="*50)
    if not PROFILE_PATH.exists():
        print("Profile dataset not found.")
        return

    df = pd.read_csv(PROFILE_PATH)
    print(f"Total ML-Ready Profiles: {len(df)}")

    if 'recommendation' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='recommendation', order=df['recommendation'].value_counts().index)
        plt.title('Distribution of Target Labels (Recommendations)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'target_labels_distribution.png', dpi=300)
        plt.close()

    features = ['sleep_change_pct', 'hr_elevation_bpm', 'training_load_ratio', 'rmssd_avg_7d']
    if all(f in df.columns for f in features + ['recommendation']):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, feature in enumerate(features):
            sns.boxplot(data=df, x='recommendation', y=feature, ax=axes[i], showfliers=False)
            axes[i].set_title(f'{feature} by Recommendation')
            axes[i].set_xlabel('')

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'features_by_recommendation.png', dpi=300)
        plt.close()

def main():
    setup_directories()
    run_demographic_eda()
    run_daily_eda()
    run_profile_eda()
    print("\nEDA plots have been saved .")

if __name__ == "__main__":
    main()

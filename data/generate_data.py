"""
Data generation script for Customer Churn / Credit Risk classification dataset.
Creates a realistic tabular dataset with 250+ entries and realistic feature distributions.
"""

import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples: int = 260, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # Generate realistic features
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    credit_scores = np.random.normal(loc=650, scale=80, size=n_samples).clip(350, 850).astype(int)
    ages = np.random.normal(loc=38, scale=10, size=n_samples).clip(18, 70).astype(int)
    tenure_years = np.random.randint(0, 11, size=n_samples)
    balances = np.random.exponential(scale=50000, size=n_samples).clip(0, 250000).round(2)
    num_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.45, 0.40, 0.12, 0.03])
    has_credit_card = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    estimated_salary = np.random.uniform(25000, 180000, size=n_samples).round(2)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.25, 0.25, 0.15])
    
    # Calculate churn probability with realistic business signals:
    # Older age, lower satisfaction, inactive status, low/extreme products increase churn
    log_odds = (
        -1.8
        - 0.003 * (credit_scores - 650)
        + 0.045 * (ages - 38)
        - 0.08 * tenure_years
        + 0.000005 * (balances - 50000)
        + 0.35 * (num_products == 1)
        + 1.20 * (num_products >= 3)
        - 0.90 * is_active_member
        - 0.45 * (satisfaction_score - 3)
        + np.random.normal(0, 0.4, size=n_samples)
    )
    
    probabilities = 1 / (1 + np.exp(-log_odds))
    churned = (np.random.uniform(0, 1, size=n_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "credit_score": credit_scores,
        "age": ages,
        "tenure": tenure_years,
        "balance": balances,
        "num_products": num_products,
        "has_credit_card": has_credit_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        "satisfaction_score": satisfaction_score,
        "churn": churned
    })
    
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "customer_churn_data.csv")
    df = generate_dataset(n_samples=260, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records successfully saved to {output_path}")
    print(f"Churn distribution:\n{df['churn'].value_counts(normalize=True).round(3)}")

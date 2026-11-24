"""
Synthetic Dataset Generator for Housing Price Valuation
Generates a realistic tabular dataset with 250 records (>100 entries)
combining linear, non-linear, and interaction effects for regression benchmarking.
"""

import os
import numpy as np
import pandas as pd

def generate_housing_data(n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Base property characteristics
    square_footage = np.random.normal(loc=2300, scale=650, size=n_samples).clip(900, 4800).astype(int)
    
    # Bedrooms correlated with square footage
    bedrooms = np.clip(np.round(square_footage / 650 + np.random.normal(0, 0.5, size=n_samples)), 1, 6).astype(int)
    
    # Bathrooms correlated with bedrooms
    bathrooms = np.clip(np.round(bedrooms * 0.75 + np.random.choice([0, 0.5, 1.0], size=n_samples, p=[0.2, 0.5, 0.3]), 1), 1.0, 5.0)
    
    # Lot size
    lot_size_sqft = (square_footage * np.random.uniform(1.3, 3.5, size=n_samples) + np.random.normal(500, 200, size=n_samples)).astype(int).clip(1200, 15000)
    
    # Age of property in years
    property_age = np.random.randint(0, 50, size=n_samples)
    
    # Distance to city center (km)
    distance_to_center = np.round(np.random.uniform(1.5, 28.0, size=n_samples), 1)
    
    # Neighborhood Quality Score (1 to 10)
    # Slightly negatively correlated with distance to center
    base_score = 9.5 - (distance_to_center / 30.0) * 4.0 + np.random.normal(0, 0.8, size=n_samples)
    neighborhood_quality = np.clip(np.round(base_score, 1), 1.0, 10.0)
    
    # Amenities (Garage & Pool)
    garage_prob = np.clip(0.4 + (square_footage - 1000) / 4000.0, 0.2, 0.95)
    has_garage = (np.random.uniform(0, 1, size=n_samples) < garage_prob).astype(int)
    
    pool_prob = np.clip(0.1 + (neighborhood_quality / 25.0) + (lot_size_sqft / 30000.0), 0.05, 0.7)
    has_pool = (np.random.uniform(0, 1, size=n_samples) < pool_prob).astype(int)
    
    # Annual property tax in USD
    annual_tax = np.round(1800 + square_footage * 1.8 + neighborhood_quality * 280 + np.random.normal(0, 300, size=n_samples), 2)
    
    # Target: Property Price in USD
    # Combining:
    # - Linear contributions: square footage ($135/sqft), bathrooms ($15k each), garage ($22k), pool ($35k)
    # - Non-linear interactions: neighborhood score exponential factor, distance decay
    # - Property age depreciation (diminishing decay)
    # - Tax synergy & realistic random market fluctuation noise
    base_price = 85000.0
    linear_comp = (
        square_footage * 135.0 +
        bedrooms * 8000.0 +
        bathrooms * 16000.0 +
        (lot_size_sqft * 7.5) +
        has_garage * 24000.0 +
        has_pool * 38000.0 +
        annual_tax * 14.0
    )
    
    # Non-linear multiplier based on neighborhood quality and distance
    location_multiplier = 1.0 + (neighborhood_quality / 10.0) * 0.45 - (distance_to_center / 30.0) * 0.25
    
    # Age depreciation: depreciates faster initially, leveling off for older historic homes
    age_discount = 1.0 - (1.0 - np.exp(-property_age / 25.0)) * 0.22
    
    # Interaction: Luxury synergy (pool + high neighborhood quality)
    synergy_bonus = (has_pool * (neighborhood_quality >= 8.0)) * 25000.0
    
    # Realistic gaussian market noise
    noise = np.random.normal(0, 14000, size=n_samples)
    
    price = (base_price + linear_comp) * location_multiplier * age_discount + synergy_bonus + noise
    price = np.round(price, 2)
    
    df = pd.DataFrame({
        "square_footage": square_footage,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size_sqft": lot_size_sqft,
        "property_age_years": property_age,
        "distance_to_center_km": distance_to_center,
        "neighborhood_quality_score": neighborhood_quality,
        "has_garage": has_garage,
        "has_pool": has_pool,
        "annual_tax": annual_tax,
        "price": price
    })
    
    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "housing_prices_data.csv")
    
    df = generate_housing_data(n_samples=250, seed=42)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records successfully.")
    print(f"File saved to: {output_path}")
    print("\nDataset Summary Preview:")
    print(df.head())
    print("\nDataset Descriptive Statistics:")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]])

if __name__ == "__main__":
    main()

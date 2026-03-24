import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("debris-catalogue.csv")

print(f"Loaded {len(df)} debris objects\n")
print(f"Columns available: {list(df.columns)}\n")

# Show the key orbital elements for the first 10 objects
cols = ['NORAD_CAT_ID', 'OBJECT_NAME', 'SEMIMAJOR_AXIS', 
        'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
        'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'PERIAPSIS', 
        'APOAPSIS', 'RCS_SIZE', 'COUNTRY_CODE']

# Only show columns that actually exist in the CSV
available = [c for c in cols if c in df.columns]

print("First 10 objects:")
print(df[available].head(10).to_string(index=False))

# Basic statistics
print(f"\n{'='*50}")
print("Summary Statistics")
print(f"{'='*50}")

if 'PERIAPSIS' in df.columns:
    print(f"  Perigee altitude:  {df['PERIAPSIS'].min():.1f} – {df['PERIAPSIS'].max():.1f} km")

if 'APOAPSIS' in df.columns:
    print(f"  Apogee altitude:   {df['APOAPSIS'].min():.1f} – {df['APOAPSIS'].max():.1f} km")

if 'INCLINATION' in df.columns:
    print(f"  Inclination:       {df['INCLINATION'].min():.2f} – {df['INCLINATION'].max():.2f}°")

if 'RA_OF_ASC_NODE' in df.columns:
    print(f"  RAAN:              {df['RA_OF_ASC_NODE'].min():.2f} – {df['RA_OF_ASC_NODE'].max():.2f}°")

if 'ECCENTRICITY' in df.columns:
    print(f"  Eccentricity:      {df['ECCENTRICITY'].min():.6f} – {df['ECCENTRICITY'].max():.6f}")

if 'RCS_SIZE' in df.columns:
    print(f"\n  RCS size breakdown:")
    print(df['RCS_SIZE'].value_counts().to_string())

if 'COUNTRY_CODE' in df.columns:
    print(f"\n  Top 5 countries of origin:")
    print(df['COUNTRY_CODE'].value_counts().head(5).to_string())
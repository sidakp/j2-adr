import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──
df = pd.read_csv("plots/debris-catalogue-full.csv")
R_E = 6371.0  # km

# Derived columns
df['ALTITUDE'] = df['SEMIMAJOR_AXIS'] - R_E  # km

# Group debris by source (extract from OBJECT_NAME)
def classify_source(name):
    name = name.upper()
    if 'FENGYUN' in name:
        return 'Fengyun-1C'
    elif 'COSMOS' in name or 'KOSMOS' in name:
        return 'Cosmos 2251'
    elif 'PSLV' in name:
        return 'PSLV upper stage'
    elif 'IRIDIUM' in name:
        return 'Iridium 33'
    else:
        return 'Other'

df['SOURCE'] = df['OBJECT_NAME'].apply(classify_source)

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

# Colour map per source
source_colours = {
    'Fengyun-1C': '#e63946',
    'Cosmos 2251': '#457b9d',
    'PSLV upper stage': '#2a9d8f',
    'Iridium 33': '#e9c46a',
    'Other': '#adb5bd',
}

# ── Figure: RAAN vs Altitude ──
fig, ax = plt.subplots(figsize=(8, 5))

for source, colour in source_colours.items():
    mask = df['SOURCE'] == source
    count = mask.sum()
    if count == 0:
        continue
    ax.scatter(
        df.loc[mask, 'RA_OF_ASC_NODE'],
        df.loc[mask, 'ALTITUDE'],
        c=colour, label=f'{source} ({count})',
        s=35, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3
    )

ax.set_xlabel('Right Ascension of Ascending Node (°)')
ax.set_ylabel('Altitude (km)')
ax.set_title('SSO Debris Catalogue: Spatial Distribution')
ax.set_xlim(-5, 365)
ax.set_ylim(500, 800)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

fig.tight_layout()
fig.savefig('plots/images/debris_raan_vs_altitude.png', bbox_inches='tight')
print("Saved debris_raan_vs_altitude.png")

# ── Figure: Inclination vs Altitude ──
fig2, ax2 = plt.subplots(figsize=(8, 5))

for source, colour in source_colours.items():
    mask = df['SOURCE'] == source
    if mask.sum() == 0:
        continue
    ax2.scatter(
        df.loc[mask, 'INCLINATION'],
        df.loc[mask, 'ALTITUDE'],
        c=colour, label=f'{source} ({mask.sum()})',
        s=35, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3
    )

ax2.set_xlabel('Inclination (°)')
ax2.set_ylabel('Altitude (km)')
ax2.set_title('SSO Debris Catalogue: Inclination vs Altitude')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

fig2.tight_layout()
fig2.savefig('plots/images/debris_inc_vs_altitude.png', bbox_inches='tight')
print("Saved debris_inc_vs_altitude.png")

plt.show()
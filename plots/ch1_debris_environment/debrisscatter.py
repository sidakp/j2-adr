import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("plots/ch1_debris_environment/debris-catalogue-full.csv")
R_E = 6371.0  # km
df['ALTITUDE'] = df['SEMIMAJOR_AXIS'] - R_E

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})

point_colour = '#3b5d7e'
n_total = len(df)


# Figure 1.2: RAAN vs Altitude
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    df['RA_OF_ASC_NODE'], df['ALTITUDE'],
    c=point_colour, s=30, edgecolors='k',
    linewidths=0.3, alpha=0.8, zorder=3,
    label=f'Debris ({n_total})',
)
ax.set_xlabel('Right Ascension of Ascending Node ($^\\circ$)')
ax.set_ylabel('Altitude (km)')
ax.set_xlim(-5, 365)
ax.set_ylim(500, 800)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
fig.tight_layout()
fig.savefig('plots/ch1_debris_environment/images/debris_raan_vs_altitude.png',
            bbox_inches='tight')
print("Saved debris_raan_vs_altitude.png")


# Figure 1.1: Inclination vs Altitude
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(
    df['INCLINATION'], df['ALTITUDE'],
    c=point_colour, s=30, edgecolors='k',
    linewidths=0.3, alpha=0.8, zorder=3,
    label=f'Debris ({n_total})',
)
ax2.set_xlabel('Inclination ($^\\circ$)')
ax2.set_ylabel('Altitude (km)')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
fig2.tight_layout()
fig2.savefig('plots/ch1_debris_environment/images/debris_inc_vs_altitude.png',
             bbox_inches='tight')
print("Saved debris_inc_vs_altitude.png")

plt.show()

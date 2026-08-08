import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ==========================================
# 1. DATA PREPARATION
# ==========================================
# Fresh Weight (biomass): still only available for the original 4 concentrations
data_ciech_fw = """NaCl_mM,FW
0,3.74
200,9.73
400,10.10
1000,0.50"""
df_fw = pd.read_csv(io.StringIO(data_ciech_fw))

# Stiffness: now available for all 6 concentrations (600/800 mM added 2026-08-08,
# from Force_Curve_Analysis_Ciechocinek_600_800__080726.xlsx, "Full Table" sheet)
data_ciech_stiffness = """NaCl_mM,Stiffness
0,1.790
200,1.278
400,1.304
600,1.003
800,0.605
1000,0.357"""
df_stiff = pd.read_csv(io.StringIO(data_ciech_stiffness))

# Combined, ordered by concentration; FW is NaN (pending) for 600 & 800 mM
df = pd.merge(df_stiff, df_fw, on='NaCl_mM', how='outer').sort_values('NaCl_mM').reset_index(drop=True)

# Correlation matrix from Table S2 (Using the exact variables we use in the app)
corr_data = """Variable,E stiffness,Pectin HM-HG,Cellulose,S/G,H/G,S,G,H,Lignin-Total yield,FW
E stiffness,1.0,0.556,0.950,0.239,-0.002,-0.080,-0.602,-0.889,-0.349,0.477
Pectin HM-HG,0.556,1.0,0.708,0.907,-0.825,0.744,0.275,-0.755,0.535,0.897
Cellulose,0.950,0.708,1.0,0.367,-0.186,0.061,-0.484,-0.987,-0.216,0.519
S/G,0.239,0.907,0.367,1.0,-0.960,0.948,0.626,-0.412,0.824,0.944
H/G,-0.002,-0.825,-0.186,-0.960,1.0,-0.981,-0.768,0.268,-0.912,-0.813
S,-0.080,0.744,0.061,0.948,-0.981,1.0,0.840,0.411,0.961,0.818
G,-0.602,0.275,-0.484,0.626,-0.768,0.840,1.0,0.958,-0.146,0.410
H,-0.889,-0.755,-0.987,-0.412,0.268,0.411,0.958,1.0,0.074,-0.516
Lignin-Total yield,-0.349,0.535,-0.216,0.824,-0.912,0.961,-0.146,0.074,1.0,0.646
FW,0.477,0.897,0.519,0.944,-0.813,0.818,0.410,-0.516,0.646,1.0"""
df_corr = pd.read_csv(io.StringIO(corr_data), index_col=0)

# ==========================================
# 2. PLOTTING FIGURE A: BAR CHART
# ==========================================
fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(df['NaCl_mM']))
width = 0.35

# FW bars: real values where available, 0-height placeholder + "pending" label where missing
fw_vals = df['FW'].fillna(0)
bars1 = ax1.bar(x - width/2, fw_vals, width, label='Fresh Biomass (g)', color='#4caf50')
for xi, (val, is_missing) in enumerate(zip(df['FW'], df['FW'].isna())):
    if is_missing:
        ax1.text(xi - width/2, 0.15, 'pending', rotation=90, ha='center', va='bottom',
                  fontsize=7, color='#4caf50', style='italic')

ax1.set_ylabel('Fresh Biomass (g)', color='#4caf50')
ax1.tick_params(axis='y', labelcolor='#4caf50')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, df['Stiffness'], width, label='Cell Wall Stiffness (MPa)', color='#2196f3')
ax2.set_ylabel('Cell Wall Stiffness (MPa)', color='#2196f3')
ax2.tick_params(axis='y', labelcolor='#2196f3')

ax1.set_xlabel('NaCl Concentration (mM)')
ax1.set_xticks(x)
ax1.set_xticklabels(df['NaCl_mM'])
ax1.set_title('Empirical Data: Biomass vs. Stiffness under Salinity Stress')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig1.tight_layout()
fig1.savefig('Figure_Biomass_Stiffness_BarChart.png', dpi=300) # High res for paper
plt.close()

# ==========================================
# 3. PLOTTING FIGURE B: HEATMAP
# ==========================================
# Use clearer, non-abbreviated labels for the axes (per reviewer request):
#   'FW' -> 'Fresh weight', 'E stiffness' -> 'Cell wall stiffness',
#   'Lignin-Total yield' -> 'Lignin (total yield)'
label_map = {
    'FW': 'Fresh weight',
    'E stiffness': 'Cell wall stiffness',
    'Lignin-Total yield': 'Lignin (total yield)',
}
df_corr_display = df_corr.rename(index=label_map, columns=label_map)

fig2, ax3 = plt.subplots(figsize=(8, 6))
# Use a diverging colormap (Red for negative, Blue/Green for positive)
sns.heatmap(df_corr_display, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
            linewidths=0.5, linecolor='white', ax=ax3, vmin=-1, vmax=1)
# No title: the descriptive caption (Figure 4) already explains the figure.
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

fig2.tight_layout()
fig2.savefig('Figure_Correlation_Heatmap.png', dpi=300) # High res for paper
plt.close()

print("✅ Figures generated successfully!")
print("1. Figure_Biomass_Stiffness_BarChart.png")
print("2. Figure_Correlation_Heatmap.png")
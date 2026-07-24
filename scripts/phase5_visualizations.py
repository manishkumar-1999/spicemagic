"""
Phase 5: Comprehensive Visualizations of Spice-Disease Database

Creates publication-quality charts:
1. Top positive/negative spice-disease pairs (bar chart)
2. Most-studied spices with sentiment breakdown (stacked bar)
3. Most-studied diseases
4. Sentiment distribution (pie chart)
5. Disease category analysis
6. Time series research trends
7. Spice vs disease category heatmap
8. Confidence distribution
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  Phase 5: Creating Visualizations")
print("=" * 70)

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# Load database
print("\n[Loading data...]")
try:
    df = pd.read_csv('final_database_v2.csv')
except FileNotFoundError:
    df = pd.read_csv('final_database.csv')
print(f"  Loaded {len(df):,} rows")

df['label'] = df['Association Nature'].astype(str).str.strip().str.lower()
df['Spice_norm'] = df['Spice'].astype(str).str.strip()
df['Disease_norm'] = df['Disease'].astype(str).str.strip()

# Color scheme
COLORS = {
    'positive': '#27AE60',   # Green
    'negative': '#E74C3C',   # Red
    'neutral':  '#F39C12',   # Orange/Yellow
}

# ============================================================
# Chart 1: Sentiment Distribution (Pie Chart)
# ============================================================
print("\n[Chart 1] Sentiment distribution pie chart...")
label_counts = df['label'].value_counts()

fig, ax = plt.subplots(figsize=(9, 8))
labels_with_counts = [f'{lbl.capitalize()}\n{cnt:,} ({cnt/len(df)*100:.1f}%)'
                       for lbl, cnt in label_counts.items()]
colors_list = [COLORS[lbl] for lbl in label_counts.index]

wedges, texts = ax.pie(label_counts.values, labels=labels_with_counts,
                        colors=colors_list, startangle=90,
                        textprops={'fontsize': 13, 'fontweight': 'bold'},
                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})

ax.set_title(f'Spice-Disease Association Sentiment Distribution\n({len(df):,} total associations)',
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizations/01_sentiment_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_sentiment_pie.png")

# ============================================================
# Chart 2: Top 15 Positive Spice-Disease Pairs
# ============================================================
print("\n[Chart 2] Top positive spice-disease pairs...")
positive = df[df['label'] == 'positive']
top_pos = (positive.groupby(['Spice_norm', 'Disease_norm']).size()
           .reset_index(name='Count').sort_values('Count', ascending=False).head(15))

fig, ax = plt.subplots(figsize=(12, 8))
pairs = [f"{r['Spice_norm']} → {r['Disease_norm']}" for _, r in top_pos.iterrows()]
counts = top_pos['Count'].values

bars = ax.barh(range(len(pairs)), counts, color=COLORS['positive'],
                edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels(pairs, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Positive Spice-Disease Associations',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')

# Add values on bars
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/02_top_positive.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_top_positive.png")

# ============================================================
# Chart 3: Top 15 Negative Spice-Disease Pairs
# ============================================================
print("\n[Chart 3] Top negative spice-disease pairs...")
negative = df[df['label'] == 'negative']
top_neg = (negative.groupby(['Spice_norm', 'Disease_norm']).size()
           .reset_index(name='Count').sort_values('Count', ascending=False).head(15))

fig, ax = plt.subplots(figsize=(12, 8))
pairs = [f"{r['Spice_norm']} → {r['Disease_norm']}" for _, r in top_neg.iterrows()]
counts = top_neg['Count'].values

bars = ax.barh(range(len(pairs)), counts, color=COLORS['negative'],
                edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels(pairs, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Negative Spice-Disease Associations',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')

for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/03_top_negative.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_top_negative.png")

# ============================================================
# Chart 4: Most-Studied Spices (Stacked Bar)
# ============================================================
print("\n[Chart 4] Most-studied spices with sentiment breakdown...")
spice_stats = df.groupby('Spice_norm').agg(
    Total=('label', 'count'),
    Positive=('label', lambda x: (x == 'positive').sum()),
    Neutral=('label', lambda x: (x == 'neutral').sum()),
    Negative=('label', lambda x: (x == 'negative').sum()),
).reset_index()
top_spices = spice_stats.sort_values('Total', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 8))
x = np.arange(len(top_spices))
ax.barh(x, top_spices['Positive'], color=COLORS['positive'], label='Positive', edgecolor='black', alpha=0.85)
ax.barh(x, top_spices['Neutral'], left=top_spices['Positive'], color=COLORS['neutral'],
        label='Neutral', edgecolor='black', alpha=0.85)
ax.barh(x, top_spices['Negative'], left=top_spices['Positive']+top_spices['Neutral'],
        color=COLORS['negative'], label='Negative', edgecolor='black', alpha=0.85)

ax.set_yticks(x)
ax.set_yticklabels(top_spices['Spice_norm'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most-Studied Spices (Sentiment Breakdown)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='x')

# Add total labels
for i, (_, row) in enumerate(top_spices.iterrows()):
    ax.text(row['Total'] + 50, i, f'{row["Total"]:,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/04_top_spices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_top_spices.png")

# ============================================================
# Chart 5: Most-Studied Diseases (Stacked Bar)
# ============================================================
print("\n[Chart 5] Most-studied diseases with sentiment breakdown...")
disease_stats = df.groupby('Disease_norm').agg(
    Total=('label', 'count'),
    Positive=('label', lambda x: (x == 'positive').sum()),
    Neutral=('label', lambda x: (x == 'neutral').sum()),
    Negative=('label', lambda x: (x == 'negative').sum()),
).reset_index()
top_diseases = disease_stats.sort_values('Total', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 8))
x = np.arange(len(top_diseases))
ax.barh(x, top_diseases['Positive'], color=COLORS['positive'], label='Positive', edgecolor='black', alpha=0.85)
ax.barh(x, top_diseases['Neutral'], left=top_diseases['Positive'], color=COLORS['neutral'],
        label='Neutral', edgecolor='black', alpha=0.85)
ax.barh(x, top_diseases['Negative'], left=top_diseases['Positive']+top_diseases['Neutral'],
        color=COLORS['negative'], label='Negative', edgecolor='black', alpha=0.85)

ax.set_yticks(x)
ax.set_yticklabels(top_diseases['Disease_norm'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most-Studied Diseases (Sentiment Breakdown)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='x')

for i, (_, row) in enumerate(top_diseases.iterrows()):
    ax.text(row['Total'] + 50, i, f'{row["Total"]:,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/05_top_diseases.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_top_diseases.png")

# ============================================================
# Chart 6: Most Beneficial Spices (Top by Positive %)
# ============================================================
print("\n[Chart 6] Most beneficial spices (highest positive %)...")
beneficial = spice_stats[spice_stats['Total'] >= 500].copy()
beneficial['Pos %'] = beneficial['Positive'] / beneficial['Total'] * 100
beneficial = beneficial.sort_values('Pos %', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(beneficial)), beneficial['Pos %'],
                color=COLORS['positive'], edgecolor='black', alpha=0.85)
ax.set_yticks(range(len(beneficial)))
ax.set_yticklabels(beneficial['Spice_norm'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('% Positive Associations', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Beneficial Spices (Min 500 papers)\nHighest % of Positive Associations',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=50, color='black', linestyle='--', alpha=0.4)

for bar, (_, row) in zip(bars, beneficial.iterrows()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{row["Pos %"]:.1f}% (n={row["Total"]:,})',
            va='center', fontsize=9, fontweight='bold')

plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('visualizations/06_most_beneficial.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_most_beneficial.png")

# ============================================================
# Chart 7: Disease Categories
# ============================================================
print("\n[Chart 7] Disease category analysis...")
if 'Disease Category' in df.columns:
    df['Category_norm'] = df['Disease Category'].astype(str).str.strip().fillna('Unknown')
    df_cat = df[df['Category_norm'] != 'nan']
    category_stats = df_cat.groupby('Category_norm').agg(
        Total=('label', 'count'),
        Positive=('label', lambda x: (x == 'positive').sum()),
        Neutral=('label', lambda x: (x == 'neutral').sum()),
        Negative=('label', lambda x: (x == 'negative').sum()),
    ).reset_index().sort_values('Total', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(len(category_stats))
    ax.barh(x, category_stats['Positive'], color=COLORS['positive'], label='Positive', edgecolor='black', alpha=0.85)
    ax.barh(x, category_stats['Neutral'], left=category_stats['Positive'],
            color=COLORS['neutral'], label='Neutral', edgecolor='black', alpha=0.85)
    ax.barh(x, category_stats['Negative'],
            left=category_stats['Positive']+category_stats['Neutral'],
            color=COLORS['negative'], label='Negative', edgecolor='black', alpha=0.85)

    ax.set_yticks(x)
    labels = [c[:45] + '...' if len(c) > 45 else c for c in category_stats['Category_norm']]
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Disease Categories (Sentiment Breakdown)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (_, row) in enumerate(category_stats.iterrows()):
        ax.text(row['Total'] + 100, i, f'{row["Total"]:,}',
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/07_disease_categories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_disease_categories.png")

# ============================================================
# Chart 8: Time Series Research Trends
# ============================================================
print("\n[Chart 8] Time series research trends...")
if 'Publication Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Publication Date'], errors='coerce').dt.year
    year_data = df.dropna(subset=['Year'])
    year_data['Year'] = year_data['Year'].astype(int)
    year_data = year_data[(year_data['Year'] >= 1970) & (year_data['Year'] <= 2025)]

    time_stats = year_data.groupby('Year').agg(
        Total=('label', 'count'),
        Positive=('label', lambda x: (x == 'positive').sum()),
        Neutral=('label', lambda x: (x == 'neutral').sum()),
        Negative=('label', lambda x: (x == 'negative').sum()),
    ).reset_index().sort_values('Year')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Total papers per year (line)
    ax1.plot(time_stats['Year'], time_stats['Total'], color='#2E75B6', linewidth=2.5, marker='o', markersize=4)
    ax1.fill_between(time_stats['Year'], time_stats['Total'], alpha=0.3, color='#2E75B6')
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax1.set_title('Spice-Disease Research Growth Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1970, 2026)

    # Panel 2: Sentiment over time (stacked area)
    ax2.stackplot(time_stats['Year'],
                  time_stats['Positive'], time_stats['Neutral'], time_stats['Negative'],
                  labels=['Positive', 'Neutral', 'Negative'],
                  colors=[COLORS['positive'], COLORS['neutral'], COLORS['negative']],
                  alpha=0.85)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1970, 2026)

    plt.tight_layout()
    plt.savefig('visualizations/08_time_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 08_time_trends.png")

# ============================================================
# Chart 9: Spice × Disease Category Heatmap
# ============================================================
print("\n[Chart 9] Spice × Disease Category heatmap...")
if 'Category_norm' in df.columns:
    # Get top 15 spices and top 15 categories
    top15_spices = spice_stats.sort_values('Total', ascending=False).head(15)['Spice_norm'].tolist()
    top15_cats = df_cat.groupby('Category_norm').size().sort_values(ascending=False).head(15).index.tolist()

    # Create matrix
    subset = df_cat[df_cat['Spice_norm'].isin(top15_spices) & df_cat['Category_norm'].isin(top15_cats)]
    heatmap_data = subset[subset['label'] == 'positive'].groupby(
        ['Spice_norm', 'Category_norm']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(top15_spices)
    heatmap_data = heatmap_data.reindex(columns=top15_cats)

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(heatmap_data.values, cmap='YlGn', aspect='auto')

    # Ticks
    ax.set_xticks(range(len(top15_cats)))
    labels_x = [c[:25] + '...' if len(c) > 25 else c for c in top15_cats]
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(top15_spices)))
    ax.set_yticklabels(top15_spices, fontsize=10)

    # Values in cells
    for i in range(len(top15_spices)):
        for j in range(len(top15_cats)):
            val = heatmap_data.values[i, j]
            if val > 0:
                color = 'white' if val > heatmap_data.values.max() * 0.5 else 'black'
                ax.text(j, i, f'{int(val)}', ha='center', va='center',
                        color=color, fontsize=8, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Positive Papers', fontsize=11, fontweight='bold')

    ax.set_title('Positive Associations Heatmap: Spice × Disease Category',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spice', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/09_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 09_heatmap.png")

# ============================================================
# Chart 10: Confidence Distribution
# ============================================================
print("\n[Chart 10] Confidence distribution...")
if 'max_confidence' in df.columns:
    fig, ax = plt.subplots(figsize=(11, 6))

    # Histogram with per-class colors
    for label, color in COLORS.items():
        subset = df[df['label'] == label]
        ax.hist(subset['max_confidence'], bins=40, alpha=0.6, color=color,
                label=f'{label.capitalize()} (n={len(subset):,})', edgecolor='black')

    ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Model Prediction Confidence by Class',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=0.75, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0.76, ax.get_ylim()[1]*0.9, 'Threshold: 0.75', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('visualizations/10_confidence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 10_confidence.png")

# ============================================================
# Master Summary Chart
# ============================================================
print("\n[Master Chart] Overall summary figure...")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Spice-Disease Association Database — Overview',
             fontsize=16, fontweight='bold')

# Sentiment pie
ax = axes[0, 0]
ax.pie(label_counts.values, labels=[l.capitalize() for l in label_counts.index],
       colors=[COLORS[l] for l in label_counts.index],
       autopct='%1.1f%%', startangle=90,
       textprops={'fontsize': 11, 'fontweight': 'bold'},
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title(f'Sentiment Distribution\n({len(df):,} associations)', fontsize=12, fontweight='bold')

# Top 10 spices
ax = axes[0, 1]
top10_spices = top_spices.head(10)
ax.barh(range(10), top10_spices['Total'], color='#2E75B6', edgecolor='black', alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_spices['Spice_norm'], fontsize=10)
ax.invert_yaxis()
ax.set_title('Top 10 Most-Studied Spices', fontsize=12, fontweight='bold')
ax.set_xlabel('Papers')
ax.grid(True, alpha=0.3, axis='x')

# Top 10 diseases
ax = axes[1, 0]
top10_diseases = top_diseases.head(10)
ax.barh(range(10), top10_diseases['Total'], color='#8E44AD', edgecolor='black', alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_diseases['Disease_norm'], fontsize=10)
ax.invert_yaxis()
ax.set_title('Top 10 Most-Studied Diseases', fontsize=12, fontweight='bold')
ax.set_xlabel('Papers')
ax.grid(True, alpha=0.3, axis='x')

# Time trend
ax = axes[1, 1]
if 'time_stats' in dir():
    recent = time_stats[time_stats['Year'] >= 2000]
    ax.plot(recent['Year'], recent['Total'], color='#E67E22', linewidth=2.5, marker='o')
    ax.fill_between(recent['Year'], recent['Total'], alpha=0.3, color='#E67E22')
    ax.set_title('Research Growth (2000+)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Papers')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/00_master_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 00_master_summary.png")

print(f"\n{'='*70}")
print(f"  ✓ ALL VISUALIZATIONS CREATED")
print(f"{'='*70}")
print(f"  📁 Location: visualizations/ folder")
print(f"\n  Charts created:")
print(f"    00_master_summary.png    — 4-panel overview")
print(f"    01_sentiment_pie.png     — Overall sentiment distribution")
print(f"    02_top_positive.png      — Top positive spice-disease pairs")
print(f"    03_top_negative.png      — Top negative pairs")
print(f"    04_top_spices.png        — Most-studied spices")
print(f"    05_top_diseases.png      — Most-studied diseases")
print(f"    06_most_beneficial.png   — Highest positive % spices")
print(f"    07_disease_categories.png — Category analysis")
print(f"    08_time_trends.png       — Research growth over time")
print(f"    09_heatmap.png           — Spice × Category heatmap")
print(f"    10_confidence.png        — Prediction confidence distribution")
print(f"\n  Ready for report inclusion!")
print(f"{'='*70}")

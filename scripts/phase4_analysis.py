"""
Phase 4: Comprehensive Analysis of Spice-Disease Database
Extracts insights from final_database_v2.csv (228,750 high-confidence labels)

Outputs:
- insights.xlsx (multiple sheets with formatted tables)
- Individual CSV files for detailed analysis
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
warnings.filterwarnings('ignore')

print("=" * 70)
print("  Phase 4: Analysis of Spice-Disease Database")
print("=" * 70)

# Load database
print("\n[Step 1] Loading final_database_v2.csv...")
db_file = 'final_database_v2.csv'
try:
    df = pd.read_csv(db_file)
    print(f"  Loaded {len(df):,} rows from {db_file}")
except FileNotFoundError:
    print(f"  {db_file} not found. Trying final_database.csv...")
    df = pd.read_csv('final_database.csv')
    print(f"  Loaded {len(df):,} rows from final_database.csv")

# Normalize labels
df['label'] = df['Association Nature'].astype(str).str.strip().str.lower()
df['Spice_norm'] = df['Spice'].astype(str).str.strip()
df['Disease_norm'] = df['Disease'].astype(str).str.strip()

print(f"\n  Label distribution:")
for label, count in df['label'].value_counts().items():
    print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")

# ============================================================
# Analysis 1: Top-N Positive Spice-Disease Pairs
# ============================================================
print("\n[Analysis 1] Top-N positive spice-disease pairs...")
positive = df[df['label'] == 'positive']
top_positive = (positive.groupby(['Spice_norm', 'Disease_norm'])
                .size()
                .reset_index(name='Count'))
top_positive['Rank'] = top_positive['Count'].rank(method='dense', ascending=False).astype(int)
top_positive = top_positive.sort_values('Count', ascending=False).head(50)
print(f"  Top pair: {top_positive.iloc[0]['Spice_norm']} → {top_positive.iloc[0]['Disease_norm']} ({top_positive.iloc[0]['Count']:,} papers)")

# ============================================================
# Analysis 2: Top-N Negative Pairs
# ============================================================
print("\n[Analysis 2] Top-N negative spice-disease pairs...")
negative = df[df['label'] == 'negative']
top_negative = (negative.groupby(['Spice_norm', 'Disease_norm'])
                .size()
                .reset_index(name='Count'))
top_negative['Rank'] = top_negative['Count'].rank(method='dense', ascending=False).astype(int)
top_negative = top_negative.sort_values('Count', ascending=False).head(50)
if len(top_negative) > 0:
    print(f"  Top pair: {top_negative.iloc[0]['Spice_norm']} → {top_negative.iloc[0]['Disease_norm']} ({top_negative.iloc[0]['Count']:,} papers)")

# ============================================================
# Analysis 3: Top-N Neutral Pairs
# ============================================================
print("\n[Analysis 3] Top-N neutral pairs...")
neutral = df[df['label'] == 'neutral']
top_neutral = (neutral.groupby(['Spice_norm', 'Disease_norm'])
                .size()
                .reset_index(name='Count'))
top_neutral['Rank'] = top_neutral['Count'].rank(method='dense', ascending=False).astype(int)
top_neutral = top_neutral.sort_values('Count', ascending=False).head(50)

# ============================================================
# Analysis 4: Spice Profiles (Most Studied Spices)
# ============================================================
print("\n[Analysis 4] Spice profiles...")
spice_stats = df.groupby('Spice_norm').agg(
    Total=('label', 'count'),
    Positive=('label', lambda x: (x == 'positive').sum()),
    Neutral=('label', lambda x: (x == 'neutral').sum()),
    Negative=('label', lambda x: (x == 'negative').sum()),
).reset_index()

# Percentages
spice_stats['Pos %'] = (spice_stats['Positive'] / spice_stats['Total'] * 100).round(1)
spice_stats['Neu %'] = (spice_stats['Neutral'] / spice_stats['Total'] * 100).round(1)
spice_stats['Neg %'] = (spice_stats['Negative'] / spice_stats['Total'] * 100).round(1)

# Most studied
top_spices = spice_stats.sort_values('Total', ascending=False).head(30)
print(f"  Most studied spice: {top_spices.iloc[0]['Spice_norm']} ({top_spices.iloc[0]['Total']:,} associations)")

# Most beneficial (highest positive % with minimum 500 papers)
beneficial_spices = spice_stats[spice_stats['Total'] >= 500].sort_values('Pos %', ascending=False).head(20)
if len(beneficial_spices) > 0:
    print(f"  Most beneficial: {beneficial_spices.iloc[0]['Spice_norm']} ({beneficial_spices.iloc[0]['Pos %']}% positive)")

# ============================================================
# Analysis 5: Disease Profiles
# ============================================================
print("\n[Analysis 5] Disease profiles...")
disease_stats = df.groupby('Disease_norm').agg(
    Total=('label', 'count'),
    Positive=('label', lambda x: (x == 'positive').sum()),
    Neutral=('label', lambda x: (x == 'neutral').sum()),
    Negative=('label', lambda x: (x == 'negative').sum()),
).reset_index()

disease_stats['Pos %'] = (disease_stats['Positive'] / disease_stats['Total'] * 100).round(1)
disease_stats['Neu %'] = (disease_stats['Neutral'] / disease_stats['Total'] * 100).round(1)
disease_stats['Neg %'] = (disease_stats['Negative'] / disease_stats['Total'] * 100).round(1)

top_diseases = disease_stats.sort_values('Total', ascending=False).head(30)
print(f"  Most studied disease: {top_diseases.iloc[0]['Disease_norm']} ({top_diseases.iloc[0]['Total']:,} associations)")

# ============================================================
# Analysis 6: Disease Category Analysis
# ============================================================
print("\n[Analysis 6] Disease category analysis...")
if 'Disease Category' in df.columns:
    df['Category_norm'] = df['Disease Category'].astype(str).str.strip().fillna('Unknown')
    category_stats = df.groupby('Category_norm').agg(
        Total=('label', 'count'),
        Positive=('label', lambda x: (x == 'positive').sum()),
        Neutral=('label', lambda x: (x == 'neutral').sum()),
        Negative=('label', lambda x: (x == 'negative').sum()),
    ).reset_index()
    category_stats['Pos %'] = (category_stats['Positive'] / category_stats['Total'] * 100).round(1)
    category_stats = category_stats.sort_values('Total', ascending=False).head(30)
    print(f"  Most studied category: {category_stats.iloc[0]['Category_norm']} ({category_stats.iloc[0]['Total']:,})")
else:
    category_stats = None

# ============================================================
# Analysis 7: Time Trends
# ============================================================
print("\n[Analysis 7] Time trends...")
time_stats = None
if 'Publication Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Publication Date'], errors='coerce').dt.year
    year_data = df.dropna(subset=['Year'])
    year_data['Year'] = year_data['Year'].astype(int)
    year_data = year_data[year_data['Year'] >= 1970]

    time_stats = year_data.groupby('Year').agg(
        Total=('label', 'count'),
        Positive=('label', lambda x: (x == 'positive').sum()),
        Neutral=('label', lambda x: (x == 'neutral').sum()),
        Negative=('label', lambda x: (x == 'negative').sum()),
    ).reset_index().sort_values('Year')
    time_stats['Pos %'] = (time_stats['Positive'] / time_stats['Total'] * 100).round(1)

    if len(time_stats) > 0:
        recent = time_stats[time_stats['Year'] >= 2015]
        if len(recent) > 0:
            print(f"  Papers 2015+: {recent['Total'].sum():,}")
            print(f"  Peak year: {time_stats.loc[time_stats['Total'].idxmax(), 'Year']} ({time_stats['Total'].max():,} papers)")

# ============================================================
# Analysis 8: Overall Statistics Summary
# ============================================================
print("\n[Analysis 8] Overall statistics...")
summary_stats = {
    'Total Associations': len(df),
    'Unique Spices': df['Spice_norm'].nunique(),
    'Unique Diseases': df['Disease_norm'].nunique(),
    'Positive Associations': (df['label'] == 'positive').sum(),
    'Neutral Associations': (df['label'] == 'neutral').sum(),
    'Negative Associations': (df['label'] == 'negative').sum(),
    'Mean Confidence': df['max_confidence'].mean().round(3) if 'max_confidence' in df.columns else 'N/A',
    'Unique Papers': df['PubMed ID'].nunique() if 'PubMed ID' in df.columns else 'N/A',
}

for key, val in summary_stats.items():
    if isinstance(val, (int, float)):
        print(f"  {key}: {val:,}" if isinstance(val, int) else f"  {key}: {val}")
    else:
        print(f"  {key}: {val}")

# ============================================================
# Save to Excel
# ============================================================
print("\n[Saving to insights.xlsx]")

wb = Workbook()

HEADER_FILL = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
HEADER_FONT = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
DATA_FONT   = Font(name='Calibri', size=11)
POS_FILL = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
NEG_FILL = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
NEU_FILL = PatternFill(start_color='FFEB9C', end_color='FFEB9C', fill_type='solid')
THIN_BORDER = Border(
    left=Side(style='thin', color='808080'),
    right=Side(style='thin', color='808080'),
    top=Side(style='thin', color='808080'),
    bottom=Side(style='thin', color='808080'),
)

def add_sheet(wb, name, data, fill=None, widths=None):
    """Add a formatted sheet."""
    if wb.active.title == 'Sheet':
        ws = wb.active
        ws.title = name
    else:
        ws = wb.create_sheet(name)

    if isinstance(data, pd.DataFrame):
        data = [data.columns.tolist()] + data.values.tolist()

    for row in data:
        ws.append(row)

    # Header formatting
    ws.row_dimensions[1].height = 30
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = THIN_BORDER

    # Data formatting
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font = DATA_FONT
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal='center', vertical='center')
            if fill and cell.row > 1:
                cell.fill = fill

    # Column widths
    if widths:
        for col_letter, width in widths.items():
            ws.column_dimensions[col_letter].width = width
    else:
        ws.column_dimensions['A'].width = 8
        for col_letter in 'BCDEFGHIJK':
            ws.column_dimensions[col_letter].width = 18

    ws.freeze_panes = 'A2'
    return ws

# Sheet 1: Summary
summary_df = pd.DataFrame({
    'Metric': list(summary_stats.keys()),
    'Value': [str(v) if not isinstance(v, (int, float)) else v for v in summary_stats.values()]
})
add_sheet(wb, '1. Summary', summary_df, widths={'A': 30, 'B': 25})

# Sheet 2: Top Positive Pairs
top_pos_display = top_positive[['Rank', 'Spice_norm', 'Disease_norm', 'Count']].head(50)
top_pos_display.columns = ['Rank', 'Spice', 'Disease', 'Papers']
add_sheet(wb, '2. Top Positive', top_pos_display, fill=POS_FILL,
          widths={'A': 8, 'B': 25, 'C': 40, 'D': 12})

# Sheet 3: Top Negative Pairs
top_neg_display = top_negative[['Rank', 'Spice_norm', 'Disease_norm', 'Count']].head(50)
top_neg_display.columns = ['Rank', 'Spice', 'Disease', 'Papers']
add_sheet(wb, '3. Top Negative', top_neg_display, fill=NEG_FILL,
          widths={'A': 8, 'B': 25, 'C': 40, 'D': 12})

# Sheet 4: Top Neutral Pairs
top_neu_display = top_neutral[['Rank', 'Spice_norm', 'Disease_norm', 'Count']].head(50)
top_neu_display.columns = ['Rank', 'Spice', 'Disease', 'Papers']
add_sheet(wb, '4. Top Neutral', top_neu_display, fill=NEU_FILL,
          widths={'A': 8, 'B': 25, 'C': 40, 'D': 12})

# Sheet 5: Most Studied Spices
spice_display = top_spices[['Spice_norm', 'Total', 'Positive', 'Neutral', 'Negative',
                             'Pos %', 'Neu %', 'Neg %']].reset_index(drop=True)
spice_display.columns = ['Spice', 'Total Papers', 'Positive', 'Neutral', 'Negative',
                         'Pos %', 'Neu %', 'Neg %']
add_sheet(wb, '5. Top Spices', spice_display,
          widths={'A': 25, 'B': 12, 'C': 11, 'D': 11, 'E': 11, 'F': 10, 'G': 10, 'H': 10})

# Sheet 6: Most Beneficial Spices
if len(beneficial_spices) > 0:
    ben_display = beneficial_spices[['Spice_norm', 'Total', 'Positive', 'Pos %']].reset_index(drop=True)
    ben_display.columns = ['Spice', 'Total Papers', 'Positive Count', 'Positive %']
    add_sheet(wb, '6. Most Beneficial', ben_display, fill=POS_FILL,
              widths={'A': 25, 'B': 12, 'C': 15, 'D': 12})

# Sheet 7: Most Studied Diseases
disease_display = top_diseases[['Disease_norm', 'Total', 'Positive', 'Neutral', 'Negative',
                                 'Pos %', 'Neu %', 'Neg %']].reset_index(drop=True)
disease_display.columns = ['Disease', 'Total Papers', 'Positive', 'Neutral', 'Negative',
                           'Pos %', 'Neu %', 'Neg %']
add_sheet(wb, '7. Top Diseases', disease_display,
          widths={'A': 35, 'B': 12, 'C': 11, 'D': 11, 'E': 11, 'F': 10, 'G': 10, 'H': 10})

# Sheet 8: Disease Categories
if category_stats is not None:
    cat_display = category_stats[['Category_norm', 'Total', 'Positive', 'Neutral', 'Negative', 'Pos %']].reset_index(drop=True)
    cat_display.columns = ['Category', 'Total Papers', 'Positive', 'Neutral', 'Negative', 'Pos %']
    add_sheet(wb, '8. Disease Categories', cat_display,
              widths={'A': 30, 'B': 12, 'C': 11, 'D': 11, 'E': 11, 'F': 10})

# Sheet 9: Time Trends
if time_stats is not None:
    time_display = time_stats[['Year', 'Total', 'Positive', 'Neutral', 'Negative', 'Pos %']].reset_index(drop=True)
    time_display.columns = ['Year', 'Total Papers', 'Positive', 'Neutral', 'Negative', 'Pos %']
    add_sheet(wb, '9. Time Trends', time_display,
              widths={'A': 10, 'B': 12, 'C': 11, 'D': 11, 'E': 11, 'F': 10})

wb.save('insights.xlsx')
print("  ✓ Saved insights.xlsx (9 sheets)")

# Also save CSV files
top_positive.to_csv('top_positive_pairs.csv', index=False)
top_negative.to_csv('top_negative_pairs.csv', index=False)
top_neutral.to_csv('top_neutral_pairs.csv', index=False)
spice_stats.sort_values('Total', ascending=False).to_csv('spice_stats.csv', index=False)
disease_stats.sort_values('Total', ascending=False).to_csv('disease_stats.csv', index=False)
if time_stats is not None:
    time_stats.to_csv('time_trends.csv', index=False)
print("  ✓ Saved individual CSV files")

# ============================================================
# Print highlights
# ============================================================
print("\n" + "="*70)
print("  KEY HIGHLIGHTS FROM ANALYSIS")
print("="*70)

print(f"\n  📊 Database Overview:")
print(f"    Total labeled associations: {len(df):,}")
print(f"    Unique spices: {df['Spice_norm'].nunique():,}")
print(f"    Unique diseases: {df['Disease_norm'].nunique():,}")
print(f"    Unique papers: {df['PubMed ID'].nunique():,}" if 'PubMed ID' in df.columns else "")

print(f"\n  🥇 Top 5 Positive Spice-Disease Associations:")
for i, row in top_positive.head(5).iterrows():
    print(f"    {row['Rank']}. {row['Spice_norm']} → {row['Disease_norm']}: {row['Count']:,} papers")

print(f"\n  🔴 Top 5 Negative Spice-Disease Associations:")
for i, row in top_negative.head(5).iterrows():
    print(f"    {row['Rank']}. {row['Spice_norm']} → {row['Disease_norm']}: {row['Count']:,} papers")

print(f"\n  🌿 Top 5 Most-Studied Spices:")
for i, row in top_spices.head(5).iterrows():
    print(f"    {row['Spice_norm']}: {row['Total']:,} total ({row['Pos %']}% positive)")

print(f"\n  🏥 Top 5 Most-Studied Diseases:")
for i, row in top_diseases.head(5).iterrows():
    print(f"    {row['Disease_norm']}: {row['Total']:,} total ({row['Pos %']}% positive)")

print(f"\n{'='*70}")
print(f"  ✓ ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"  ✓ insights.xlsx        — 9 sheets of formatted analysis")
print(f"  ✓ top_positive_pairs.csv, top_negative_pairs.csv, top_neutral_pairs.csv")
print(f"  ✓ spice_stats.csv, disease_stats.csv, time_trends.csv")
print(f"\n  Next: Phase 5 — Visualizations from these insights")
print(f"{'='*70}")

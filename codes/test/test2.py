import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
import statsmodels.api as sm
from pathlib import Path
import json


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('time_based_analysis')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_dataset(file_path, logger):
    """Load the processed dataset."""
    try:
        df = pd.read_excel(file_path, dtype={"Schadennummer": str})
        df.columns = df.columns.str.strip()
        logger.info(f"Successfully loaded data from: {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns only
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise


def calculate_visit_metrics(df, output_folder, logger):
    """
    Calculate various visit timing metrics for each patient.

    Returns:
      A patient-level DataFrame with metrics:
        - Heilungsdauer (days to last visit)
        - Anzahl_Besuche (number of visits)
        - Tage_bis_Erstbesuch (days until first visit)
        - Durchschnitt_Tage_zwischen_Besuchen, Standardabweichung_Besuchsabstände,
          Max_Lücke_zwischen_Besuchen, Min_Lücke_zwischen_Besuchen
        - Besuchsfrequenz_pro_Monat and category for frequency
        - Early intervention flags for 30, 60, and 90 days.
    """
    logger.info("Calculating visit timing metrics...")

    required_columns = ['Schadennummer', 'Unfalldatum', 'Besuchsdatum', 'Days_Since_Accident']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    for date_col in ['Unfalldatum', 'Besuchsdatum']:
        if date_col in df.columns and pd.api.types.is_object_dtype(df[date_col]):
            logger.info(f"Converting {date_col} to datetime")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    patient_metrics = []
    unique_patients = df['Schadennummer'].unique()
    logger.info(f"Analyzing visit patterns for {len(unique_patients)} patients")

    for patient_id in unique_patients:
        try:
            patient_visits = df[df['Schadennummer'] == patient_id].copy()
            patient_visits = patient_visits.sort_values('Besuchsdatum')
            visit_count = len(patient_visits)
            accident_date = patient_visits['Unfalldatum'].iloc[0]
            healing_duration = patient_visits['Days_Since_Accident'].max()
            first_visit_date = patient_visits['Besuchsdatum'].iloc[0]
            days_to_first_visit = (first_visit_date - accident_date).days

            if visit_count > 1:
                visit_dates = patient_visits['Besuchsdatum'].tolist()
                visit_gaps = [(visit_dates[i] - visit_dates[i - 1]).days for i in range(1, len(visit_dates))]
                avg_days_between = np.mean(visit_gaps)
                std_days_between = np.std(visit_gaps)
                max_gap = max(visit_gaps)
                min_gap = min(visit_gaps)
            else:
                avg_days_between = std_days_between = max_gap = min_gap = np.nan

            visit_frequency = (visit_count / healing_duration) * 30 if healing_duration > 0 else np.nan

            early_30 = days_to_first_visit <= 30
            early_60 = days_to_first_visit <= 60
            early_90 = days_to_first_visit <= 90

            if visit_count <= 3:
                freq_category = "Low (≤3)"
            elif visit_count <= 6:
                freq_category = "Medium (4-6)"
            else:
                freq_category = "High (>6)"

            patient_metrics.append({
                'Schadennummer': patient_id,
                'Heilungsdauer': healing_duration,
                'Besuchsanzahl': visit_count,
                'Tage_bis_Erstbesuch': days_to_first_visit,
                'Durchschnitt_Tage_zwischen_Besuchen': avg_days_between,
                'Standardabweichung_Besuchsabstände': std_days_between,
                'Max_Lücke_zwischen_Besuchen': max_gap,
                'Min_Lücke_zwischen_Besuchen': min_gap,
                'Besuchsfrequenz_pro_Monat': visit_frequency,
                'Frühe_Intervention_30': early_30,
                'Frühe_Intervention_60': early_60,
                'Frühe_Intervention_90': early_90,
                'Besuchsfrequenz_Kategorie': freq_category
            })
            logger.info(f"Patient {patient_id}: {visit_count} visits, first visit at {days_to_first_visit} days")
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}", exc_info=True)

    patient_df = pd.DataFrame(patient_metrics)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "patient_visit_metrics.xlsx")
    patient_df.to_excel(output_path, index=False)
    logger.info(f"Saved patient visit metrics to {output_path}")
    return patient_df


def analyze_first_visit_impact(patient_df, plots_folder, logger):
    """
    Analyze the impact of first visit timing on healing duration.
    Generates a scatter plot with regression and a box plot comparing early vs. delayed groups.
    """
    logger.info("Analyzing impact of first visit timing...")
    required = ['Heilungsdauer', 'Tage_bis_Erstbesuch',
                'Frühe_Intervention_30', 'Frühe_Intervention_60', 'Frühe_Intervention_90']
    missing = [col for col in required if col not in patient_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    corr, p_val = stats.pearsonr(patient_df['Tage_bis_Erstbesuch'], patient_df['Heilungsdauer'])
    logger.info(f"Correlation between days to first visit and healing duration: r={corr:.3f}, p={p_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(patient_df['Tage_bis_Erstbesuch'], patient_df['Heilungsdauer'],
                alpha=0.7, s=50, color="#3498db", edgecolor="black")
    x = patient_df['Tage_bis_Erstbesuch']
    y = patient_df['Heilungsdauer']
    m, b = np.polyfit(x, y, 1)
    plt.plot(np.array([x.min(), x.max()]), m * np.array([x.min(), x.max()]) + b,
             'r--', label=f'Trend: y = {m:.2f}x + {b:.2f}')
    plt.text(0.05, 0.95, f"Correlation: r = {corr:.3f}\np-value: {p_val:.4f}",
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Relationship Between First Visit Timing and Healing Duration", fontsize=14, fontweight="bold")
    plt.xlabel("Days to First Visit After Accident", fontsize=12)
    plt.ylabel("Healing Duration (Days)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    os.makedirs(plots_folder, exist_ok=True)
    output_path = os.path.join(plots_folder, "first_visit_timing_impact.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved first visit timing impact plot to {output_path}")

    # Compare early vs. delayed intervention at thresholds 30, 60, and 90 days.
    early_results = {}
    for threshold, col in [(30, 'Frühe_Intervention_30'), (60, 'Frühe_Intervention_60'), (90, 'Frühe_Intervention_90')]:
        early_group = patient_df[patient_df[col]]['Heilungsdauer']
        delayed_group = patient_df[~patient_df[col]]['Heilungsdauer']
        early_count = len(early_group)
        delayed_count = len(delayed_group)
        logger.info(f"Threshold {threshold} days: {early_count} early, {delayed_count} delayed")
        if early_count >= 3 and delayed_count >= 3:
            early_mean = early_group.mean()
            delayed_mean = delayed_group.mean()
            mean_diff = early_mean - delayed_mean
            if early_count >= 8 and delayed_count >= 8:
                early_normal = stats.shapiro(early_group)[1] > 0.05
                delayed_normal = stats.shapiro(delayed_group)[1] > 0.05
                if early_normal and delayed_normal:
                    t_stat, p_val_group = stats.ttest_ind(early_group, delayed_group, equal_var=False)
                    test_name = "t-test"
                else:
                    u_stat, p_val_group = stats.mannwhitneyu(early_group, delayed_group, alternative='two-sided')
                    test_name = "Mann-Whitney U"
            else:
                u_stat, p_val_group = stats.mannwhitneyu(early_group, delayed_group, alternative='two-sided')
                test_name = "Mann-Whitney U"

            pooled_std = np.sqrt(((early_count - 1) * early_group.std() ** 2 +
                                  (delayed_count - 1) * delayed_group.std() ** 2) /
                                 (early_count + delayed_count - 2))
            if pooled_std > 0:
                cohen_d = abs(mean_diff) / pooled_std
                if cohen_d < 0.2:
                    effect_interp = "negligible"
                elif cohen_d < 0.5:
                    effect_interp = "small"
                elif cohen_d < 0.8:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
            else:
                cohen_d = None
                effect_interp = "not calculable"

            logger.info(
                f"Threshold {threshold} days: {test_name} p={p_val_group:.4f}, Cohen's d={cohen_d:.2f} ({effect_interp})")
            early_results[threshold] = {
                'early_mean': early_mean,
                'delayed_mean': delayed_group.mean(),
                'mean_difference': mean_diff,
                'test_type': test_name,
                'p_value': p_val_group,
                'significant': p_val_group < 0.05,
                'cohens_d': cohen_d,
                'effect_size_interpretation': effect_interp,
                'early_count': early_count,
                'delayed_count': delayed_count
            }
            # Create box plot for this threshold
            plt.figure(figsize=(10, 6))
            data_to_plot = [early_group, delayed_group]
            bp = plt.boxplot(data_to_plot, patch_artist=True, labels=[
                f'Early (≤{threshold} days)\nn={early_count}',
                f'Delayed (>{threshold} days)\nn={delayed_count}'
            ])
            colors = ['lightblue', 'lightcoral']
            for box, color in zip(bp['boxes'], colors):
                box.set(facecolor=color)
            for i, group in enumerate(data_to_plot):
                plt.scatter(np.repeat(i + 1, len(group)), group, alpha=0.7, s=30, color='navy', zorder=3)
            plt.title(f"Healing Duration by Early vs. Delayed Intervention ({threshold} days)", fontsize=14,
                      fontweight="bold")
            plt.ylabel("Healing Duration (Days)", fontsize=12)
            plt.grid(axis='y', alpha=0.3, linestyle="--")
            stats_text = f"Mean diff: {abs(mean_diff):.1f} days\n{test_name}: p = {p_val_group:.4f}"
            if cohen_d is not None:
                stats_text += f"\nEffect size (d): {cohen_d:.2f} ({effect_interp})"
            plt.text(0.5, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=12,
                     ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            output_box = os.path.join(plots_folder, f"early_intervention_{threshold}_days.png")
            plt.tight_layout()
            plt.savefig(output_box, dpi=300)
            plt.close()
            logger.info(f"Saved early intervention box plot for threshold {threshold} days to {output_box}")
        else:
            logger.warning(f"Not enough data for threshold {threshold} days comparison")
    return {
        'correlation': {'r': corr, 'p_value': p_val, 'significant': p_val < 0.05},
        'early_intervention': early_results,
        'regression': {'slope': m, 'intercept': b}
    }


def analyze_visit_frequency(patient_df, plots_folder, logger):
    """
    Analyze the impact of visit frequency on healing duration.
    Generates scatter plots and box plots for visit count and monthly frequency.
    """
    logger.info("Analyzing impact of visit frequency...")
    required = ['Heilungsdauer', 'Besuchsanzahl', 'Besuchsfrequenz_pro_Monat', 'Besuchsfrequenz_Kategorie']
    missing = [col for col in required if col not in patient_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    count_corr, count_p = stats.pearsonr(patient_df['Besuchsanzahl'], patient_df['Heilungsdauer'])
    logger.info(f"Correlation between visit count and healing duration: r={count_corr:.3f}, p={count_p:.4f}")

    freq_data = patient_df.dropna(subset=['Besuchsfrequenz_pro_Monat', 'Heilungsdauer'])
    if len(freq_data) >= 5:
        freq_corr, freq_p = stats.pearsonr(freq_data['Besuchsfrequenz_pro_Monat'], freq_data['Heilungsdauer'])
        logger.info(
            f"Correlation between visit frequency per month and healing duration: r={freq_corr:.3f}, p={freq_p:.4f}")
    else:
        logger.warning("Not enough data to calculate correlation for visit frequency")
        freq_corr, freq_p = None, None

    # Scatter plot for visit count
    plt.figure(figsize=(10, 6))
    plt.scatter(patient_df['Besuchsanzahl'], patient_df['Heilungsdauer'],
                alpha=0.7, s=50, color="#e74c3c", edgecolor="black")
    x = patient_df['Besuchsanzahl']
    y = patient_df['Heilungsdauer']
    count_m, count_b = np.polyfit(x, y, 1)
    plt.plot(np.array([x.min(), x.max()]), count_m * np.array([x.min(), x.max()]) + count_b,
             'r--', label=f'Trend: y = {count_m:.2f}x + {count_b:.2f}')
    plt.text(0.05, 0.95, f"Correlation: r = {count_corr:.3f}\np-value: {count_p:.4f}",
             transform=plt.gca().transAxes, fontsize=12, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Relationship Between Visit Count and Healing Duration", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Visits", fontsize=12)
    plt.ylabel("Healing Duration (Days)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    output_count = os.path.join(plots_folder, "visit_count_impact.png")
    plt.tight_layout()
    plt.savefig(output_count, dpi=300)
    plt.close()
    logger.info(f"Saved visit count impact plot to {output_count}")

    # Scatter plot for visit frequency (if data exists)
    if freq_corr is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(freq_data['Besuchsfrequenz_pro_Monat'], freq_data['Heilungsdauer'],
                    alpha=0.7, s=50, color="#2ecc71", edgecolor="black")
        x_freq = freq_data['Besuchsfrequenz_pro_Monat']
        y_freq = freq_data['Heilungsdauer']
        freq_m, freq_b = np.polyfit(x_freq, y_freq, 1)
        plt.plot(np.array([x_freq.min(), x_freq.max()]),
                 freq_m * np.array([x_freq.min(), x_freq.max()]) + freq_b,
                 'g--', label=f'Trend: y = {freq_m:.2f}x + {freq_b:.2f}')
        plt.text(0.05, 0.95, f"Correlation: r = {freq_corr:.3f}\np-value: {freq_p:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title("Relationship Between Visit Frequency and Healing Duration", fontsize=14, fontweight="bold")
        plt.xlabel("Visit Frequency (Visits per Month)", fontsize=12)
        plt.ylabel("Healing Duration (Days)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        output_freq = os.path.join(plots_folder, "visit_frequency_impact.png")
        plt.tight_layout()
        plt.savefig(output_freq, dpi=300)
        plt.close()
        logger.info(f"Saved visit frequency impact plot to {output_freq}")

    # Analyze visit frequency categories with box plot
    category_counts = patient_df['Besuchsfrequenz_Kategorie'].value_counts()
    logger.info(f"Visit frequency category counts: {category_counts.to_dict()}")
    valid_categories = [cat for cat, count in category_counts.items() if count >= 3]
    if len(valid_categories) >= 2:
        cat_data = patient_df[patient_df['Besuchsfrequenz_Kategorie'].isin(valid_categories)]
        cat_stats = cat_data.groupby('Besuchsfrequenz_Kategorie')['Heilungsdauer'].agg(
            ['count', 'mean', 'median', 'std']).reset_index()
        category_order = [cat for cat in ['Low (≤3)', 'Medium (4-6)', 'High (>6)'] if cat in valid_categories]
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Besuchsfrequenz_Kategorie', y='Heilungsdauer', data=cat_data, order=category_order)
        sns.stripplot(x='Besuchsfrequenz_Kategorie', y='Heilungsdauer', data=cat_data, order=category_order,
                      color='black', alpha=0.5, jitter=True)
        for i, cat in enumerate(category_order):
            stat_row = cat_stats[cat_stats['Besuchsfrequenz_Kategorie'] == cat]
            plt.text(i, cat_data['Heilungsdauer'].max() * 0.9,
                     f"n = {stat_row['count'].values[0]}\nMean: {stat_row['mean'].values[0]:.1f}\nMedian: {stat_row['median'].values[0]:.1f}",
                     ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title("Healing Duration by Visit Frequency Category", fontsize=14, fontweight="bold")
        plt.xlabel("Visit Frequency Category", fontsize=12)
        plt.ylabel("Healing Duration (Days)", fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle="--")
        output_cat = os.path.join(plots_folder, "visit_frequency_categories.png")
        plt.tight_layout()
        plt.savefig(output_cat, dpi=300)
        plt.close()
        logger.info(f"Saved visit frequency categories plot to {output_cat}")
        category_comparison = {
            'test_type': "ANOVA" if len(valid_categories) >= 3 else "t-test",
            'p_value': None,  # p_value from the test (to be computed if needed)
            'significant': None,
            'categories': category_order,
            'statistics': cat_stats.to_dict()
        }
    else:
        logger.warning("Not enough data across visit frequency categories for comparison")
        category_comparison = None

    return {
        'visit_count_correlation': {
            'r': count_corr,
            'p_value': count_p,
            'significant': count_p < 0.05,
            'slope': count_m,
            'intercept': count_b
        },
        'visit_frequency_correlation': {
            'r': freq_corr,
            'p_value': freq_p,
            'significant': freq_p < 0.05 if freq_p is not None else None,
            'slope': freq_m if freq_corr is not None else None,
            'intercept': freq_b if freq_corr is not None else None
        } if freq_corr is not None else None,
        'category_comparison': category_comparison
    }


def analyze_visit_spacing(patient_df, plots_folder, logger):
    """
    Analyze the impact of visit spacing on healing duration.
    Generates scatter plots for average spacing, maximum gap, and variability in spacing.
    """
    logger.info("Analyzing impact of visit spacing...")
    required = ['Heilungsdauer', 'Durchschnitt_Tage_zwischen_Besuchen',
                'Standardabweichung_Besuchsabstände', 'Max_Lücke_zwischen_Besuchen']
    missing = [col for col in required if col not in patient_df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    spacing_df = patient_df.dropna(subset=['Durchschnitt_Tage_zwischen_Besuchen'])
    if len(spacing_df) < 5:
        logger.warning("Not enough patients with multiple visits for spacing analysis")
        return None

    avg_corr, avg_p = stats.pearsonr(spacing_df['Durchschnitt_Tage_zwischen_Besuchen'], spacing_df['Heilungsdauer'])
    logger.info(f"Correlation between average visit spacing and healing duration: r={avg_corr:.3f}, p={avg_p:.4f}")

    std_df = spacing_df.dropna(subset=['Standardabweichung_Besuchsabstände'])
    if len(std_df) >= 5:
        std_corr, std_p = stats.pearsonr(std_df['Standardabweichung_Besuchsabstände'], std_df['Heilungsdauer'])
        logger.info(f"Correlation between spacing variability and healing duration: r={std_corr:.3f}, p={std_p:.4f}")
    else:
        std_corr, std_p = None, None
        logger.warning("Not enough data for spacing variability correlation")

    max_corr, max_p = stats.pearsonr(spacing_df['Max_Lücke_zwischen_Besuchen'], spacing_df['Heilungsdauer'])
    logger.info(f"Correlation between maximum gap and healing duration: r={max_corr:.3f}, p={max_p:.4f}")

    # Scatter plot for average spacing
    plt.figure(figsize=(10, 6))
    plt.scatter(spacing_df['Durchschnitt_Tage_zwischen_Besuchen'], spacing_df['Heilungsdauer'],
                alpha=0.7, s=50, color="#9b59b6", edgecolor="black")
    avg_m, avg_b = np.polyfit(spacing_df['Durchschnitt_Tage_zwischen_Besuchen'], spacing_df['Heilungsdauer'], 1)
    x_range = np.array([spacing_df['Durchschnitt_Tage_zwischen_Besuchen'].min(),
                        spacing_df['Durchschnitt_Tage_zwischen_Besuchen'].max()])
    plt.plot(x_range, avg_m * x_range + avg_b, '--', color='purple', label=f'Trend: y = {avg_m:.2f}x + {avg_b:.2f}')
    plt.text(0.05, 0.95, f"Correlation: r = {avg_corr:.3f}\np-value: {avg_p:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Relationship Between Average Visit Spacing and Healing Duration", fontsize=14, fontweight="bold")
    plt.xlabel("Average Days Between Visits", fontsize=12)
    plt.ylabel("Healing Duration (Days)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    output_avg = os.path.join(plots_folder, "average_visit_spacing_impact.png")
    plt.tight_layout()
    plt.savefig(output_avg, dpi=300)
    plt.close()
    logger.info(f"Saved average visit spacing impact plot to {output_avg}")

    # Scatter plot for maximum gap
    plt.figure(figsize=(10, 6))
    plt.scatter(spacing_df['Max_Lücke_zwischen_Besuchen'], spacing_df['Heilungsdauer'],
                alpha=0.7, s=50, color="#f39c12", edgecolor="black")
    max_m, max_b = np.polyfit(spacing_df['Max_Lücke_zwischen_Besuchen'], spacing_df['Heilungsdauer'], 1)
    x_range_max = np.array(
        [spacing_df['Max_Lücke_zwischen_Besuchen'].min(), spacing_df['Max_Lücke_zwischen_Besuchen'].max()])
    plt.plot(x_range_max, max_m * x_range_max + max_b, '--', color='orange',
             label=f'Trend: y = {max_m:.2f}x + {max_b:.2f}')
    plt.text(0.05, 0.95, f"Correlation: r = {max_corr:.3f}\np-value: {max_p:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Relationship Between Maximum Visit Gap and Healing Duration", fontsize=14, fontweight="bold")
    plt.xlabel("Maximum Gap Between Visits (Days)", fontsize=12)
    plt.ylabel("Healing Duration (Days)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    output_max = os.path.join(plots_folder, "maximum_visit_gap_impact.png")
    plt.tight_layout()
    plt.savefig(output_max, dpi=300)
    plt.close()
    logger.info(f"Saved maximum visit gap impact plot to {output_max}")

    # Scatter plot for spacing variability (if available)
    if std_corr is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(std_df['Standardabweichung_Besuchsabstände'], std_df['Heilungsdauer'],
                    alpha=0.7, s=50, color="#16a085", edgecolor="black")
        std_m, std_b = np.polyfit(std_df['Standardabweichung_Besuchsabstände'], std_df['Heilungsdauer'], 1)
        x_range_std = np.array(
            [std_df['Standardabweichung_Besuchsabstände'].min(), std_df['Standardabweichung_Besuchsabstände'].max()])
        plt.plot(x_range_std, std_m * x_range_std + std_b, '--', color='teal',
                 label=f'Trend: y = {std_m:.2f}x + {std_b:.2f}')
        plt.text(0.05, 0.95, f"Correlation: r = {std_corr:.3f}\np-value: {std_p:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title("Relationship Between Visit Spacing Variability and Healing Duration", fontsize=14, fontweight="bold")
        plt.xlabel("Std. Deviation of Visit Spacing (Days)", fontsize=12)
        plt.ylabel("Healing Duration (Days)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle="--")
        output_std = os.path.join(plots_folder, "visit_spacing_variability_impact.png")
        plt.tight_layout()
        plt.savefig(output_std, dpi=300)
        plt.close()
        logger.info(f"Saved visit spacing variability impact plot to {output_std}")

    return {
        'average_spacing': {
            'r': avg_corr,
            'p_value': avg_p,
            'significant': avg_p < 0.05,
            'slope': avg_m,
            'intercept': avg_b
        },
        'spacing_variability': {
            'r': std_corr,
            'p_value': std_p,
            'significant': std_p < 0.05 if std_p is not None else None,
            'slope': std_m if std_corr is not None else None,
            'intercept': std_b if std_corr is not None else None
        } if std_corr is not None else None,
        'maximum_gap': {
            'r': max_corr,
            'p_value': max_p,
            'significant': max_p < 0.05,
            'slope': max_m,
            'intercept': max_b
        }
    }


def generate_patient_timeline_plots(df, patient_df, output_folder, logger, max_patients=8):
    """
    Generate timeline plots showing visit patterns for selected patients.
    """
    logger.info("Generating patient timeline plots...")
    required = ['Schadennummer', 'Unfalldatum', 'Besuchsdatum', 'Days_Since_Accident']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns for timeline plots: {missing}")
        return False

    if len(patient_df) > max_patients:
        sorted_patients = patient_df.sort_values('Heilungsdauer')
        short_patients = sorted_patients.iloc[:2]['Schadennummer'].tolist()
        long_patients = sorted_patients.iloc[-2:]['Schadennummer'].tolist()
        middle_start = len(sorted_patients) // 2 - (max_patients - 4) // 2
        middle_end = middle_start + (max_patients - 4)
        middle_patients = sorted_patients.iloc[middle_start:middle_end]['Schadennummer'].tolist()
        selected_patients = short_patients + middle_patients + long_patients
    else:
        selected_patients = patient_df['Schadennummer'].tolist()

    logger.info(f"Selected {len(selected_patients)} patients for timeline plots")
    for patient_id in selected_patients:
        try:
            patient_visits = df[df['Schadennummer'] == patient_id].copy()
            patient_visits = patient_visits.sort_values('Besuchsdatum')
            healing_duration = patient_visits['Days_Since_Accident'].max()
            plt.figure(figsize=(12, 6))
            visit_days = patient_visits['Days_Since_Accident'].tolist()
            plt.scatter(visit_days, [1] * len(visit_days), s=100, color='blue', zorder=3)
            for i, day in enumerate(visit_days):
                plt.text(day, 1.05, f"Visit {i + 1}", ha='center', va='bottom', fontsize=9)
            plt.plot([0, healing_duration], [1, 1], 'k-', alpha=0.3)
            tick_positions = list(range(0, int(healing_duration) + 60, 60))
            plt.xticks(tick_positions)
            if len(visit_days) > 1:
                gaps = [visit_days[i] - visit_days[i - 1] for i in range(1, len(visit_days))]
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                max_gap = max(gaps)
            else:
                avg_gap = std_gap = max_gap = np.nan
            metrics_text = f"Patient ID: {patient_id}\nHealing Duration: {healing_duration:.0f} days\nVisits: {len(visit_days)}"
            if not np.isnan(avg_gap):
                metrics_text += f"\nAvg gap: {avg_gap:.1f} days\nMax gap: {max_gap:.0f} days\nStd gap: {std_gap:.1f} days"
            plt.text(0.02, 0.02, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.title(f"Visit Timeline for Patient {patient_id}", fontsize=14, fontweight="bold")
            plt.xlabel("Days Since Accident", fontsize=12)
            plt.yticks([])
            plt.xlim(-30, healing_duration + 30)
            plt.ylim(0.8, 1.2)
            plt.grid(axis='x', alpha=0.3, linestyle="--")
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"timeline_patient_{patient_id}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Created timeline plot for patient {patient_id}")
        except Exception as e:
            logger.error(f"Error creating timeline for patient {patient_id}: {e}")
    return True


def generate_summary_report(first_visit_results, frequency_results, spacing_results, output_folder, logger):
    """Generate a summary report in Markdown format summarizing the time-based analysis."""
    report_path = os.path.join(output_folder, "time_based_analysis_report.md")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Time-Based Analysis Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write("## Overview\n\n")
            f.write(
                "This analysis explores the relationship between patient visit timing, frequency, and healing duration in polytrauma cases.\n\n")
            f.write("## First Visit Timing\n\n")
            if first_visit_results and 'correlation' in first_visit_results:
                corr_data = first_visit_results['correlation']
                f.write(
                    f"Correlation between days to first visit and healing duration: **r = {corr_data['r']:.3f}** (p = {corr_data['p_value']:.4f}).\n\n")
                if corr_data['significant']:
                    f.write(
                        "This indicates a statistically significant relationship between early intervention and recovery time.\n\n")
                else:
                    f.write("No statistically significant correlation was found.\n\n")
                if 'early_intervention' in first_visit_results:
                    f.write("### Early vs. Delayed Intervention\n\n")
                    for threshold, res in first_visit_results['early_intervention'].items():
                        f.write(
                            f"- **{threshold} days threshold:** Early group mean = {res['early_mean']:.1f} days, Delayed group mean = {res['delayed_mean']:.1f} days, p = {res['p_value']:.4f}, Cohen's d = {res['cohens_d']:.2f} ({res['effect_size_interpretation']}).\n")
                    f.write("\n")
            else:
                f.write("No valid first visit timing results available.\n\n")

            f.write("## Visit Frequency\n\n")
            if frequency_results:
                count_corr = frequency_results['visit_count_correlation']
                f.write(
                    f"Visit count vs. healing duration: **r = {count_corr['r']:.3f}** (p = {count_corr['p_value']:.4f}).\n\n")
                if frequency_results['visit_frequency_correlation']:
                    freq_corr = frequency_results['visit_frequency_correlation']
                    f.write(
                        f"Visit frequency (per month) vs. healing duration: **r = {freq_corr['r']:.3f}** (p = {freq_corr['p_value']:.4f}).\n\n")
                if frequency_results['category_comparison']:
                    comp = frequency_results['category_comparison']
                    f.write(
                        f"Comparison across visit frequency categories ({', '.join(comp['categories'])}) using {comp['test_type']} yielded p = {comp['p_value']:.4f}.\n\n")
            else:
                f.write("No valid visit frequency results available.\n\n")

            f.write("## Visit Spacing\n\n")
            if spacing_results:
                avg = spacing_results['average_spacing']
                f.write(
                    f"Average visit spacing vs. healing duration: **r = {avg['r']:.3f}** (p = {avg['p_value']:.4f}).\n\n")
                max_gap = spacing_results['maximum_gap']
                f.write(
                    f"Maximum gap vs. healing duration: **r = {max_gap['r']:.3f}** (p = {max_gap['p_value']:.4f}).\n\n")
                if spacing_results.get('spacing_variability'):
                    var_space = spacing_results['spacing_variability']
                    f.write(
                        f"Visit spacing variability vs. healing duration: **r = {var_space['r']:.3f}** (p = {var_space['p_value']:.4f}).\n\n")
            else:
                f.write("No valid visit spacing results available.\n\n")

            f.write("## Integration with Other Analyses\n\n")
            f.write(
                "These time-based metrics complement previous analyses (e.g., injury and demographic factors) by providing insights into the care process. Further work could integrate these findings into multivariate models or survival analysis.\n\n")
            f.write("## Limitations\n\n")
            f.write("1. Small sample size (only 30 patients) limits statistical power.\n")
            f.write("2. Visit reasons are not differentiated; visits may vary in clinical importance.\n")
            f.write("3. Missing data and variability in follow-up schedules may affect the results.\n")
        logger.info(f"Summary report generated at {report_path}")
    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)
        return None
    return report_path


if __name__ == "__main__":
    # Define file paths (adjust these paths as needed)
    data_file = r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\data\output\step1\Polytrauma_Analysis_Processed.xlsx"
    output_folder = r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\data\output\time_based_analysis"
    plots_folder = os.path.join(output_folder, "plots")
    log_folder = r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\logs"
    log_name = "time_based_analysis.log"

    # Setup logger
    logger = setup_logging(log_folder, log_name)
    logger.info("Starting time-based analysis module...")

    # Load dataset
    df = load_dataset(data_file, logger)

    # Calculate visit metrics at patient level
    patient_metrics_df = calculate_visit_metrics(df, output_folder, logger)

    # Analyze first visit timing impact
    first_visit_results = analyze_first_visit_impact(patient_metrics_df, plots_folder, logger)

    # Analyze visit frequency impact
    frequency_results = analyze_visit_frequency(patient_metrics_df, plots_folder, logger)

    # Analyze visit spacing impact
    spacing_results = analyze_visit_spacing(patient_metrics_df, plots_folder, logger)

    # Generate timeline plots for selected patients
    generate_patient_timeline_plots(df, patient_metrics_df, os.path.join(plots_folder, "timelines"), logger)

    # Generate summary report
    report_path = generate_summary_report(first_visit_results, frequency_results, spacing_results, output_folder,
                                          logger)

    logger.info("Time-based analysis completed successfully.")

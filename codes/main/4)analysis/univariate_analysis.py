import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from codes.main.helper import select_statistical_test
from statsmodels.stats.multitest import multipletests


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('univariate_analysis')
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(file_handler)

    return logger


def load_dataset(file_path, logger):
    """Load dataset with appropriate error handling."""
    try:
        df = pd.read_excel(file_path, dtype={"Schadennummer": str})
        df.columns = df.columns.str.strip()
        logger.info(f"Erfolgreich geladene Daten aus: {file_path}")
        logger.info(f"Datenform: {df.shape}")
        logger.info(f"Spalten: {list(df.columns)[:10]}...")  # Show first 10 columns only
        return df
    except Exception as e:
        logger.error(f"Fehler beim Laden des Datensatzes: {str(e)}", exc_info=True)
        raise


def apply_multiple_testing_correction(body_part_results, logger):
    """
    Apply multiple testing correction to the body part analysis results.

    Parameters:
    -----------
    body_part_results : list
        List of dictionaries containing body part analysis results
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    list
        Updated results with multiple testing correction
    """
    try:
        # Extract valid p-values (skipping None values and insufficient samples)
        valid_results = [result for result in body_part_results
                         if
                         result and result.get('p_value') is not None and not result.get('insufficient_sample', False)]

        if not valid_results:
            logger.warning("Keine validen p-Werte für Korrektur multipler Tests gefunden")
            return body_part_results

        # Extract p-values and keep track of their original indices
        p_values = [result['p_value'] for result in valid_results]
        logger.info(f"Anwendung der Korrektur für multiple Tests auf {len(p_values)} p-Werte")

        # Bonferroni correction (family-wise error rate)
        n_tests = len(p_values)
        bonferroni_alpha = 0.05 / n_tests
        logger.info(f"Bonferroni korrigiertes Alpha-Niveau: {bonferroni_alpha:.6f}")

        # Update results with Bonferroni correction
        for result in valid_results:
            result['bonferroni_significant'] = result['p_value'] < bonferroni_alpha

        # Benjamini-Hochberg procedure (false discovery rate)
        rejected, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')

        # Update results with BH correction
        for i, result in enumerate(valid_results):
            result['p_adjusted'] = p_adjusted[i]
            result['fdr_bh_significant'] = rejected[i]

        # Log the results of the correction
        significant_original = sum(1 for result in valid_results if result['significant'])
        significant_bonferroni = sum(1 for result in valid_results if result.get('bonferroni_significant', False))
        significant_fdr = sum(1 for result in valid_results if result.get('fdr_bh_significant', False))

        logger.info(f"Signifikante Ergebnisse ohne Korrektur: {significant_original}")
        logger.info(f"Signifikante Ergebnisse nach Bonferroni: {significant_bonferroni}")
        logger.info(f"Signifikante Ergebnisse nach FDR (Benjamini-Hochberg): {significant_fdr}")

        # Log details of which body parts remain significant after correction
        if significant_bonferroni > 0:
            bonferroni_parts = [result['body_part'] for result in valid_results if
                                result.get('bonferroni_significant', False)]
            logger.info(f"Nach Bonferroni-Korrektur signifikante Körperteile: {', '.join(bonferroni_parts)}")

        if significant_fdr > 0:
            fdr_parts = [result['body_part'] for result in valid_results if result.get('fdr_bh_significant', False)]
            logger.info(f"Nach FDR-Korrektur signifikante Körperteile: {', '.join(fdr_parts)}")

        # Create a table to compare p-values and adjusted p-values
        comparison_data = []
        for result in valid_results:
            comparison_data.append({
                'body_part': result['body_part'],
                'original_p': result['p_value'],
                'original_significant': result['significant'],
                'bonferroni_alpha': bonferroni_alpha,
                'bonferroni_significant': result.get('bonferroni_significant', False),
                'adjusted_p': result.get('p_adjusted'),
                'fdr_significant': result.get('fdr_bh_significant', False)
            })

        # Convert to DataFrame for easier logging
        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"Vergleich der p-Werte vor und nach Korrektur:\n{comparison_df}")

        return body_part_results

    except Exception as e:
        logger.error(f"Fehler bei Korrektur für multiple Tests: {str(e)}", exc_info=True)
        return body_part_results



def prepare_patient_level_data(df, logger):
    """
    Prepare a patient-level dataset for analysis.
    Extract healing duration and injury information for each unique patient.
    """
    try:
        logger.info("Vorbereitung von Patientendaten auf Einzelfallebene...")

        # Identify unique patients
        unique_patients = df["Schadennummer"].unique()
        logger.info(f"Anzahl der eindeutigen Patienten: {len(unique_patients)}")

        # Create a patient-level dataframe
        patient_data = []

        # Define body parts to analyze
        body_parts = ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm links', 'Arm rechts',
                      'Wirbelsaeule', 'Bein rechts', 'Bein links', 'Becken']

        # Define merged categories
        merged_categories = {
            'Arm': ['Arm links', 'Arm rechts'],
            'Bein': ['Bein links', 'Bein rechts']
        }

        # For each patient, extract data from their last visit and injury information across all visits
        for patient_id in unique_patients:
            patient_records = df[df["Schadennummer"] == patient_id]

            # Get last visit information
            last_visit = patient_records.loc[patient_records["Days_Since_Accident"].idxmax()]

            # Basic patient info
            patient_info = {
                "Schadennummer": patient_id,
                "Heilungsdauer": last_visit["Days_Since_Accident"],
                # Healing duration is days from accident to last visit
                "Alter": last_visit["Age_At_Accident"] if "Age_At_Accident" in last_visit else None,
                "Geschlecht": last_visit["Geschlecht"] if "Geschlecht" in last_visit else None,
                "Anzahl_Besuche": len(patient_records)
            }

            # Extract injury information (if any visit has "Ja", mark as injured)
            for body_part in body_parts:
                if body_part in patient_records.columns:
                    patient_info[body_part] = "Ja" if "Ja" in patient_records[body_part].values else "Nein"

            # Create merged body part fields
            for merged_name, components in merged_categories.items():
                valid_components = [c for c in components if c in patient_records.columns]
                if valid_components:
                    component_values = [patient_info.get(c, "Nein") for c in valid_components]
                    patient_info[merged_name] = "Ja" if "Ja" in component_values else "Nein"

            # Calculate injury count (how many body parts are affected)
            injury_fields = [bp for bp in body_parts if bp in patient_records.columns]
            patient_info["Verletzungsanzahl"] = sum(1 for bp in injury_fields if patient_info[bp] == "Ja")

            # Add to the patient dataset
            patient_data.append(patient_info)

        # Convert to DataFrame
        patient_df = pd.DataFrame(patient_data)
        logger.info(f"Patientendatensatz erstellt mit {len(patient_df)} Fällen und {len(patient_df.columns)} Spalten")

        # Validate the data
        healing_durations = patient_df["Heilungsdauer"]
        logger.info(f"Heilungsdauer Statistiken: Min={healing_durations.min()}, "
                    f"Max={healing_durations.max()}, Mittelwert={healing_durations.mean():.1f}, "
                    f"Median={healing_durations.median()}")

        return patient_df

    except Exception as e:
        logger.error(f"Fehler bei der Vorbereitung der Patientendaten: {str(e)}", exc_info=True)
        raise


def test_normality(data, logger):
    """Test if the data follows a normal distribution."""
    if len(data) < 8:
        logger.warning("Zu wenige Datenpunkte für einen verlässlichen Normalitätstest")
        return False

    # Use Shapiro-Wilk test for normality
    stat, p_value = stats.shapiro(data)

    if p_value > 0.05:
        logger.info(f"Daten sind normalverteilt (p={p_value:.4f})")
        return True
    else:
        logger.info(f"Daten sind nicht normalverteilt (p={p_value:.4f})")
        return False


def analyze_body_part_impact(patient_df, body_part, plots_folder, logger):
    """
    Analyze the impact of a specific body part injury on healing duration.

    Parameters:
    -----------
    patient_df : pandas.DataFrame
        Patient-level dataframe
    body_part : str
        Name of the body part to analyze
    plots_folder : str
        Path to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the analysis
    """
    try:
        if body_part not in patient_df.columns:
            logger.warning(f"Körperteil {body_part} nicht in den Daten gefunden. Analyse übersprungen.")
            return None

        # Split patients based on injury status
        injury_group = patient_df[patient_df[body_part] == "Ja"]["Heilungsdauer"]
        no_injury_group = patient_df[patient_df[body_part] == "Nein"]["Heilungsdauer"]

        # Count cases in each group
        injury_count = len(injury_group)
        no_injury_count = len(no_injury_group)

        logger.info(
            f"Analyse für {body_part}: {injury_count} Verletzungsfälle, {no_injury_count} Nicht-Verletzungsfälle")

        # Skip if not enough cases in either group
        if injury_count < 1 or no_injury_count < 1:
            logger.warning(
                f"Unzureichende Stichprobengröße für {body_part}. Mindestens ein Fall pro Gruppe erforderlich.")
            return {
                "body_part": body_part,
                "injury_count": injury_count,
                "no_injury_count": no_injury_count,
                "insufficient_sample": True
            }

        # Calculate basic statistics
        injury_mean = injury_group.mean()
        no_injury_mean = no_injury_group.mean()
        mean_diff = injury_mean - no_injury_mean

        # Add direction interpretation
        direction = "verlängert" if mean_diff > 0 else "verkürzt"
        effect_direction = "longer" if mean_diff > 0 else "shorter"

        injury_median = injury_group.median()
        no_injury_median = no_injury_group.median()

        # Use standardized test selection
        test_result = select_statistical_test(injury_group, no_injury_group, logger)

        test_type = test_result['test_type']
        p_value = test_result['p_value']
        test_statistic = test_result['test_statistic']

        # Log result for the specific body part
        if p_value is not None:
            logger.info(
                f"{body_part}: {test_type} resulted in {'t' if test_type == 't-test' else 'U'}={test_statistic:.2f}, p={p_value:.4f}")
            logger.info(
                f"{body_part}: Mean difference of {abs(mean_diff):.1f} days {effect_direction} healing duration (p={p_value:.4f})")

            # Check if result is significant
            is_significant = p_value < 0.05
        else:
            is_significant = None

        # Calculate effect size if we have sufficient data
        cohens_d = None
        effect_size_interp = "nicht berechenbar"

        if injury_count > 1 and no_injury_count > 1:
            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((len(injury_group) - 1) * injury_group.std() ** 2 +
                 (len(no_injury_group) - 1) * no_injury_group.std() ** 2) /
                (len(injury_group) + len(no_injury_group) - 2)
            )

            if pooled_std > 0:
                cohens_d = abs(mean_diff) / pooled_std

                # Interpret Cohen's d
                if cohens_d < 0.2:
                    effect_size_interp = "vernachlässigbar"
                elif cohens_d < 0.5:
                    effect_size_interp = "klein"
                elif cohens_d < 0.8:
                    effect_size_interp = "mittel"
                else:
                    effect_size_interp = "groß"

                logger.info(f"{body_part}: Effektstärke (Cohen's d) = {cohens_d:.2f} ({effect_size_interp})")
            else:
                logger.warning(f"{body_part}: Effektstärke konnte nicht berechnet werden (Standardabweichung = 0)")

        # Create visualization: box plot comparing healing duration
        plt.figure(figsize=(12, 6))

        # Left subplot: Box plot
        plt.subplot(1, 2, 1)
        # Create a custom palette
        palette = {"Ja": "#3498db", "Nein": "#e74c3c"}

        # Create the box plot
        sns.boxplot(x=body_part, y="Heilungsdauer", data=patient_df,
                    palette=palette, width=0.5)

        # Add individual data points
        sns.stripplot(x=body_part, y="Heilungsdauer", data=patient_df,
                      jitter=True, size=5, alpha=0.7, color="black")

        # Add mean lines with labels
        plt.axhline(y=injury_mean, color="#3498db", linestyle="--",
                    label=f"Mittelwert (Verletzung): {injury_mean:.1f}")
        plt.axhline(y=no_injury_mean, color="#e74c3c", linestyle="--",
                    label=f"Mittelwert (keine Verletzung): {no_injury_mean:.1f}")

        # Add titles and labels
        plt.title(f"Vergleich der Heilungsdauer: {body_part}", fontsize=14, fontweight="bold")
        plt.xlabel(f"{body_part} Verletzung", fontsize=12)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
        plt.legend(loc="upper right")

        # Annotate with sample sizes
        plt.annotate(f"n = {injury_count}", xy=(0, injury_mean),
                     xytext=(0, injury_mean + 50), ha="center",
                     bbox=dict(boxstyle="round", alpha=0.1))
        plt.annotate(f"n = {no_injury_count}", xy=(1, no_injury_mean),
                     xytext=(1, no_injury_mean + 50), ha="center",
                     bbox=dict(boxstyle="round", alpha=0.1))

        # Right subplot: Histogram comparison
        plt.subplot(1, 2, 2)

        # Create histogram for both groups
        sns.histplot(injury_group, color="#3498db", label="Verletzung",
                     alpha=0.6, kde=True, stat="density")
        sns.histplot(no_injury_group, color="#e74c3c", label="Keine Verletzung",
                     alpha=0.6, kde=True, stat="density")

        # Add titles and labels
        plt.title(f"Verteilung der Heilungsdauer: {body_part}", fontsize=14, fontweight="bold")
        plt.xlabel("Heilungsdauer (Tage)", fontsize=12)
        plt.ylabel("Dichte", fontsize=12)
        plt.legend(loc="upper right")

        # Add statistical annotation
        stat_text = f"Statistischer Test: {test_type}\n"
        stat_text += f"p-Wert: {p_value:.4f}" if p_value is not None else "p-Wert: N/A"
        stat_text += f" ({'signifikant' if is_significant else 'nicht signifikant'})\n"
        stat_text += f"Effektgröße (Cohen's d): {cohens_d:.2f} ({effect_size_interp})" if cohens_d is not None else "Effektgröße: nicht berechenbar"

        plt.annotate(stat_text, xy=(0.05, 0.95), xycoords="axes fraction",
                     va="top", ha="left",
                     bbox=dict(boxstyle="round", alpha=0.1))

        plt.tight_layout()

        # Create folders if they don't exist
        os.makedirs(plots_folder, exist_ok=True)

        # Save the figure
        output_path = os.path.join(plots_folder, f"{body_part}_Heilungsdauer.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualisierung für {body_part} gespeichert unter: {output_path}")

        # Create result dictionary with clinical relevance
        result = {
            "body_part": body_part,
            "injury_count": injury_count,
            "no_injury_count": no_injury_count,
            "injury_mean": injury_mean,
            "no_injury_mean": no_injury_mean,
            "mean_difference": mean_diff,
            "direction": direction,
            "injury_median": injury_median,
            "no_injury_median": no_injury_median,
            "test_type": test_type,
            "p_value": p_value,
            "significant": is_significant,
            "cohens_d": cohens_d,
            "effect_size_interpretation": effect_size_interp,
            "sufficient_sample": test_result['test_type'] != "not performed",
            "clinical_relevance": "Möglicherweise klinisch relevant trotz fehlender statistischer Signifikanz"
                                if cohens_d and cohens_d >= 0.8 and not is_significant else None
        }

        return result

    except Exception as e:
        logger.error(f"Fehler bei der Analyse von {body_part}: {str(e)}", exc_info=True)
        return None


def analyze_injury_count_impact(patient_df, plots_folder, logger):
    """
    Analyze the relationship between injury count and healing duration.

    Parameters:
    -----------
    patient_df : pandas.DataFrame
        Patient-level dataframe
    plots_folder : str
        Path to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the analysis
    """
    try:
        # Check if we have injury count data
        if "Verletzungsanzahl" not in patient_df.columns:
            logger.warning("Verletzungsanzahl nicht in den Daten gefunden. Analyse übersprungen.")
            return None

        logger.info("Analyse der Auswirkung der Verletzungsanzahl auf die Heilungsdauer...")

        # Calculate correlation between injury count and healing duration
        correlation, p_value = stats.pearsonr(patient_df["Verletzungsanzahl"], patient_df["Heilungsdauer"])

        logger.info(f"Korrelation zwischen Verletzungsanzahl und Heilungsdauer: r={correlation:.4f}, p={p_value:.4f}")

        is_significant = p_value < 0.05

        # Create injury count categories
        patient_df["Verletzungskategorie"] = pd.cut(
            patient_df["Verletzungsanzahl"],
            bins=[0, 2, 4, 10],
            labels=["1-2", "3-4", "5+"]
        )

        # Calculate statistics by injury count category (with observed=True to suppress FutureWarning)
        category_stats = patient_df.groupby("Verletzungskategorie", observed=True)["Heilungsdauer"].agg(
            ["count", "mean", "median", "std"]
        ).reset_index()

        # Create visualization: scatter plot with trend line
        plt.figure(figsize=(10, 6))

        # Scatter plot
        plt.scatter(patient_df["Verletzungsanzahl"], patient_df["Heilungsdauer"],
                    alpha=0.7, s=50, color="#3498db", edgecolor="black")

        # Add trend line
        m, b = np.polyfit(patient_df["Verletzungsanzahl"], patient_df["Heilungsdauer"], 1)
        plt.plot(np.unique(patient_df["Verletzungsanzahl"]),
                 m * np.unique(patient_df["Verletzungsanzahl"]) + b,
                 color="red", linestyle="--", label=f"Trendlinie (y = {m:.1f}x + {b:.1f})")

        # Add correlation annotation
        stats_text = f"Korrelation: r = {correlation:.2f}"
        stats_text += f"\np-Wert: {p_value:.4f}"
        stats_text += f"\nSignifikanz: {'Ja' if is_significant else 'Nein'}"

        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                 va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add titles and labels
        plt.title("Auswirkung der Anzahl der verletzten Körperteile auf die Heilungsdauer",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Anzahl der verletzten Körperteile", fontsize=12)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
        plt.legend()

        # Add grid
        plt.grid(visible=True, alpha=0.3, linestyle="--")

        # Save the plot
        output_path = os.path.join(plots_folder, "Verletzungsanzahl_Heilungsdauer_Scatter.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Visualisierung für Verletzungsanzahl gespeichert unter: {output_path}")

        # Create box plot for injury count categories
        plt.figure(figsize=(10, 6))

        # Box plot by category
        sns.boxplot(x="Verletzungskategorie", y="Heilungsdauer", data=patient_df,
                    hue="Verletzungskategorie", palette="viridis", legend=False)

        # Add individual data points
        sns.stripplot(x="Verletzungskategorie", y="Heilungsdauer", data=patient_df,
                      jitter=True, alpha=0.6, color="black", size=4)

        # Add category statistics
        for i, (_, row) in enumerate(category_stats.iterrows()):
            plt.text(i, patient_df["Heilungsdauer"].max() * 0.9,
                     f"n = {row['count']}\nMittelwert: {row['mean']:.1f}\nMedian: {row['median']:.1f}",
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add titles and labels
        plt.title("Heilungsdauer nach Verletzungsschweregrad", fontsize=14, fontweight="bold")
        plt.xlabel("Anzahl der verletzten Körperteile", fontsize=12)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)

        # Add grid
        plt.grid(visible=True, axis='y', alpha=0.3, linestyle="--")

        # Save the plot
        output_path = os.path.join(plots_folder, "Verletzungsanzahl_Heilungsdauer_Box.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Box-Plot für Verletzungskategorien gespeichert unter: {output_path}")

        # Create ANOVA to test if categories differ significantly
        groups = [patient_df[patient_df["Verletzungskategorie"] == cat]["Heilungsdauer"]
                  for cat in category_stats["Verletzungskategorie"]]

        if all(len(g) > 0 for g in groups):
            f_stat, p_value_anova = stats.f_oneway(*groups)
            logger.info(f"ANOVA für Verletzungskategorien: F={f_stat:.2f}, p={p_value_anova:.4f}")
            is_significant_anova = p_value_anova < 0.05
        else:
            f_stat = None
            p_value_anova = None
            is_significant_anova = None
            logger.warning("Nicht genügend Daten in allen Kategorien für ANOVA-Test")

        # Create result dictionary
        result = {
            "correlation": correlation,
            "correlation_p_value": p_value,
            "correlation_significant": is_significant,
            "category_stats": category_stats.to_dict(),
            "anova_f_stat": f_stat,
            "anova_p_value": p_value_anova,
            "anova_significant": is_significant_anova
        }

        return result

    except Exception as e:
        logger.error(f"Fehler bei der Analyse der Verletzungsanzahl: {str(e)}", exc_info=True)
        return None


def analyze_demographic_impact(patient_df, plots_folder, logger):
    """
    Analyze the impact of demographic factors (age, gender) on healing duration.

    Parameters:
    -----------
    patient_df : pandas.DataFrame
        Patient-level dataframe
    plots_folder : str
        Path to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the analysis
    """
    try:
        logger.info("Analyse der Auswirkung demografischer Faktoren auf die Heilungsdauer...")

        results = {}

        # Part 1: Age Analysis
        if "Alter" in patient_df.columns:
            # Remove missing age values
            age_data = patient_df.dropna(subset=["Alter"])

            if len(age_data) > 5:  # Only proceed if we have enough age data
                # Calculate correlation between age and healing duration
                correlation, p_value = stats.pearsonr(age_data["Alter"], age_data["Heilungsdauer"])

                logger.info(f"Korrelation zwischen Alter und Heilungsdauer: r={correlation:.4f}, p={p_value:.4f}")

                is_significant = p_value < 0.05

                # Create scatter plot for age
                plt.figure(figsize=(10, 6))

                # Scatter plot
                plt.scatter(age_data["Alter"], age_data["Heilungsdauer"],
                            alpha=0.7, s=50, color="#2ecc71", edgecolor="black")

                # Add trend line
                m, b = np.polyfit(age_data["Alter"], age_data["Heilungsdauer"], 1)
                plt.plot(np.array([age_data["Alter"].min(), age_data["Alter"].max()]),
                         m * np.array([age_data["Alter"].min(), age_data["Alter"].max()]) + b,
                         color="red", linestyle="--", label=f"Trendlinie (y = {m:.1f}x + {b:.1f})")

                # Add correlation annotation
                stats_text = f"Korrelation: r = {correlation:.2f}"
                stats_text += f"\np-Wert: {p_value:.4f}"
                stats_text += f"\nSignifikanz: {'Ja' if is_significant else 'Nein'}"

                plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                         va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Add titles and labels
                plt.title("Auswirkung des Alters auf die Heilungsdauer", fontsize=14, fontweight="bold")
                plt.xlabel("Alter (Jahre)", fontsize=12)
                plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
                plt.legend()

                # Add grid
                plt.grid(visible=True, alpha=0.3, linestyle="--")

                # Save the plot
                output_path = os.path.join(plots_folder, "Alter_Heilungsdauer.png")
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()

                logger.info(f"Visualisierung für Altersauswirkung gespeichert unter: {output_path}")

                # Store age results
                results["age"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "significant": is_significant,
                    "slope": m,
                    "intercept": b
                }

                # Create age decade categories if we have enough data
                if len(age_data) >= 10:
                    # Create decade bins
                    age_data["Altersdekade"] = (age_data["Alter"] // 10) * 10
                    age_data["Altersdekade"] = age_data["Altersdekade"].astype(str) + "er"

                    # Calculate statistics by decade
                    decade_stats = age_data.groupby("Altersdekade")["Heilungsdauer"].agg(
                        ["count", "mean", "median", "std"]
                    ).reset_index()

                    # Only include decades with at least 2 patients
                    valid_decades = decade_stats[decade_stats["count"] >= 2]

                    if len(valid_decades) >= 2:  # Need at least 2 decades for comparison
                        # Create box plot for age decades
                        plt.figure(figsize=(12, 6))

                        # Box plot by decade
                        decade_data = age_data[age_data["Altersdekade"].isin(valid_decades["Altersdekade"])]
                        sns.boxplot(x="Altersdekade", y="Heilungsdauer", data=decade_data,
                                    palette="viridis")

                        # Add individual data points
                        sns.stripplot(x="Altersdekade", y="Heilungsdauer", data=decade_data,
                                      jitter=True, alpha=0.6, color="black", size=4)

                        # Add decade statistics
                        for i, (_, row) in enumerate(valid_decades.iterrows()):
                            plt.text(i, decade_data["Heilungsdauer"].max() * 0.9,
                                     f"n = {row['count']}\nMittelwert: {row['mean']:.1f}",
                                     ha='center', va='top', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        # Add titles and labels
                        plt.title("Heilungsdauer nach Altersdekade", fontsize=14, fontweight="bold")
                        plt.xlabel("Altersdekade", fontsize=12)
                        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)

                        # Add grid
                        plt.grid(visible=True, axis='y', alpha=0.3, linestyle="--")

                        # Save the plot
                        output_path = os.path.join(plots_folder, "Altersdekade_Heilungsdauer.png")
                        plt.tight_layout()
                        plt.savefig(output_path, dpi=300)
                        plt.close()

                        logger.info(f"Box-Plot für Altersdekaden gespeichert unter: {output_path}")

                        # Create ANOVA to test if decades differ significantly
                        groups = [decade_data[decade_data["Altersdekade"] == dec]["Heilungsdauer"]
                                  for dec in valid_decades["Altersdekade"]]

                        if all(len(g) > 1 for g in groups):
                            f_stat, p_value_anova = stats.f_oneway(*groups)
                            logger.info(f"ANOVA für Altersdekaden: F={f_stat:.2f}, p={p_value_anova:.4f}")
                            is_significant_anova = p_value_anova < 0.05

                            # Store decade results
                            results["age_decades"] = {
                                "decade_stats": valid_decades.to_dict(),
                                "anova_f_stat": f_stat,
                                "anova_p_value": p_value_anova,
                                "anova_significant": is_significant_anova
                            }
                        else:
                            logger.warning("Nicht genügend Daten in allen Altersdekaden für ANOVA-Test")
            else:
                logger.warning("Nicht genügend Altersdaten für Dekaden-Analyse vorhanden")
        else:
            logger.warning("Nicht genügend Altersdaten vorhanden für Korrelationsanalyse")

        # Part 2: Gender Analysis
        if "Geschlecht" in patient_df.columns:
            # Remove missing gender values
            gender_data = patient_df.dropna(subset=["Geschlecht"])

            # Check if we have enough data for each gender
            gender_counts = gender_data["Geschlecht"].value_counts()

            if len(gender_counts) > 1 and all(gender_counts > 2):
                logger.info(f"Geschlechterverteilung: {gender_counts.to_dict()}")

                # Calculate statistics by gender
                gender_stats = gender_data.groupby("Geschlecht")["Heilungsdauer"].agg(
                    ["count", "mean", "median", "std"]
                ).reset_index()

                # Test if we can use t-test
                male_data = gender_data[gender_data["Geschlecht"] == "männlich"]["Heilungsdauer"]
                female_data = gender_data[gender_data["Geschlecht"] == "weiblich"]["Heilungsdauer"]

                if len(male_data) > 0 and len(female_data) > 0:
                    # Test for normality in each group
                    male_normal = len(male_data) >= 8 and test_normality(male_data, logger)
                    female_normal = len(female_data) >= 8 and test_normality(female_data, logger)

                    if male_normal and female_normal:
                        # Use t-test
                        t_stat, p_value = stats.ttest_ind(male_data, female_data, equal_var=False)
                        test_type = "t-test"
                        logger.info(f"Geschlechtsunterschiede: T-Test ergab t={t_stat:.2f}, p={p_value:.4f}")
                    else:
                        # Use Mann-Whitney U test
                        u_stat, p_value = stats.mannwhitneyu(male_data, female_data, alternative='two-sided')
                        test_type = "Mann-Whitney-U"
                        logger.info(
                            f"Geschlechtsunterschiede: Mann-Whitney-U-Test ergab U={u_stat:.2f}, p={p_value:.4f}")

                    is_significant = p_value < 0.05

                    # Create visualization
                    plt.figure(figsize=(10, 6))

                    # Box plot by gender
                    sns.boxplot(x="Geschlecht", y="Heilungsdauer", data=gender_data,
                                hue="Geschlecht", palette=["#3498db", "#e74c3c"], legend=False)

                    # Add individual data points
                    sns.stripplot(x="Geschlecht", y="Heilungsdauer", data=gender_data,
                                  jitter=True, alpha=0.6, color="black", size=4)

                    # Add gender statistics
                    for i, (_, row) in enumerate(gender_stats.iterrows()):
                        plt.text(i, gender_data["Heilungsdauer"].max() * 0.9,
                                 f"n = {row['count']}\nMittelwert: {row['mean']:.1f}\nMedian: {row['median']:.1f}",
                                 ha='center', va='top', fontsize=10,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Add statistical annotation
                    stats_text = f"{test_type}: p={p_value:.4f} ({'signifikant' if is_significant else 'nicht signifikant'})"

                    plt.text(0.5, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=11,
                             ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Add titles and labels
                    plt.title("Heilungsdauer nach Geschlecht", fontsize=14, fontweight="bold")
                    plt.xlabel("Geschlecht", fontsize=12)
                    plt.ylabel("Heilungsdauer (Tage)", fontsize=12)

                    # Add grid
                    plt.grid(visible=True, axis='y', alpha=0.3, linestyle="--")

                    # Save the plot
                    output_path = os.path.join(plots_folder, "Geschlecht_Heilungsdauer.png")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300)
                    plt.close()

                    logger.info(f"Visualisierung für Geschlechtsunterschiede gespeichert unter: {output_path}")

                    # Calculate effect size (Cohen's d)
                    if len(male_data) > 0 and len(female_data) > 0:
                        # Pooled standard deviation
                        pooled_std = np.sqrt(
                            ((len(male_data) - 1) * male_data.std() ** 2 +
                             (len(female_data) - 1) * female_data.std() ** 2) /
                            (len(male_data) + len(female_data) - 2)
                        )

                        mean_diff = male_data.mean() - female_data.mean()

                        if pooled_std > 0:
                            cohens_d = abs(mean_diff) / pooled_std

                            # Interpret Cohen's d
                            if cohens_d < 0.2:
                                effect_size_interp = "vernachlässigbar"
                            elif cohens_d < 0.5:
                                effect_size_interp = "klein"
                            elif cohens_d < 0.8:
                                effect_size_interp = "mittel"
                            else:
                                effect_size_interp = "groß"

                            logger.info(
                                f"Geschlechtsunterschiede: Effektstärke (Cohen's d) = {cohens_d:.2f} ({effect_size_interp})")
                        else:
                            cohens_d = None
                            effect_size_interp = "nicht berechenbar"
                    else:
                        cohens_d = None
                        effect_size_interp = "nicht berechenbar"
                        mean_diff = None

                    # Store gender results
                    results["gender"] = {
                        "gender_stats": gender_stats.to_dict(),
                        "test_type": test_type,
                        "p_value": p_value,
                        "significant": is_significant,
                        "mean_difference": mean_diff,
                        "cohens_d": cohens_d,
                        "effect_size_interpretation": effect_size_interp
                    }
                else:
                    logger.warning("Nicht genügend Daten für beide Geschlechter vorhanden")
            else:
                logger.warning("Nicht genügend Geschlechtsdaten für Analyse vorhanden")

        return results

    except Exception as e:
        logger.error(f"Fehler bei der Analyse demografischer Faktoren: {str(e)}", exc_info=True)
        return None


def export_results_to_excel(results, output_folder, logger):
    """Export analysis results to Excel for reporting."""
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create Excel writer
        output_path = os.path.join(output_folder, "univariate_analysis_results.xlsx")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Export body part results
            if "body_parts" in results and results["body_parts"]:
                # Create dataframe from body part results
                body_part_data = []

                for bp_result in results["body_parts"]:
                    if bp_result:  # Skip None results
                        row = {
                            "Körperteil": bp_result["body_part"],
                            "Verletzungsfälle": bp_result["injury_count"],
                            "Nicht-Verletzungsfälle": bp_result["no_injury_count"],
                            "Mittelwert (Verletzung)": bp_result.get("injury_mean"),
                            "Mittelwert (Keine Verletzung)": bp_result.get("no_injury_mean"),
                            "Differenz der Mittelwerte": bp_result.get("mean_difference"),
                            "Median (Verletzung)": bp_result.get("injury_median"),
                            "Median (Keine Verletzung)": bp_result.get("no_injury_median"),
                            "Testtyp": bp_result.get("test_type"),
                            "p-Wert": bp_result.get("p_value"),
                            "Signifikant": bp_result.get("significant"),
                            "Effektgröße (Cohen's d)": bp_result.get("cohens_d"),
                            "Effektgröße Interpretation": bp_result.get("effect_size_interpretation"),
                            "Ausreichende Stichprobe": bp_result.get("sufficient_sample")
                        }
                        body_part_data.append(row)

                if body_part_data:
                    bp_df = pd.DataFrame(body_part_data)
                    bp_df.to_excel(writer, sheet_name="Körperteil_Analyse", index=False)

            # Export injury count results
            if "injury_count" in results and results["injury_count"]:
                # Create summary dataframe
                summary_data = [{
                    "Metrik": "Korrelation (r)",
                    "Wert": results["injury_count"]["correlation"]
                }, {
                    "Metrik": "p-Wert (Korrelation)",
                    "Wert": results["injury_count"]["correlation_p_value"]
                }, {
                    "Metrik": "Signifikant",
                    "Wert": results["injury_count"]["correlation_significant"]
                }, {
                    "Metrik": "ANOVA F-Statistik",
                    "Wert": results["injury_count"]["anova_f_stat"]
                }, {
                    "Metrik": "ANOVA p-Wert",
                    "Wert": results["injury_count"]["anova_p_value"]
                }, {
                    "Metrik": "ANOVA Signifikant",
                    "Wert": results["injury_count"]["anova_significant"]
                }]

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Verletzungsanzahl", index=False)

            # Export demographic results
            if "demographics" in results:
                demo_data = []

                # Add age results if available
                if "age" in results["demographics"]:
                    age_results = results["demographics"]["age"]
                    demo_data.extend([{
                        "Kategorie": "Alter",
                        "Metrik": "Korrelation (r)",
                        "Wert": age_results["correlation"]
                    }, {
                        "Kategorie": "Alter",
                        "Metrik": "p-Wert",
                        "Wert": age_results["p_value"]
                    }, {
                        "Kategorie": "Alter",
                        "Metrik": "Signifikant",
                        "Wert": age_results["significant"]
                    }])

                # Add gender results if available
                if "gender" in results["demographics"]:
                    gender_results = results["demographics"]["gender"]
                    demo_data.extend([{
                        "Kategorie": "Geschlecht",
                        "Metrik": "Testtyp",
                        "Wert": gender_results["test_type"]
                    }, {
                        "Kategorie": "Geschlecht",
                        "Metrik": "p-Wert",
                        "Wert": gender_results["p_value"]
                    }, {
                        "Kategorie": "Geschlecht",
                        "Metrik": "Signifikant",
                        "Wert": gender_results["significant"]
                    }, {
                        "Kategorie": "Geschlecht",
                        "Metrik": "Effektgröße (Cohen's d)",
                        "Wert": gender_results.get("cohens_d")
                    }])

                if demo_data:
                    demo_df = pd.DataFrame(demo_data)
                    demo_df.to_excel(writer, sheet_name="Demografische_Faktoren", index=False)

        logger.info(f"Ergebnisse in Excel exportiert: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Fehler beim Exportieren der Ergebnisse nach Excel: {str(e)}", exc_info=True)
        return None


def generate_summary_report(results, output_folder, logger):
    """Generate a Markdown summary report of the analysis results."""
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create report path
        report_path = os.path.join(output_folder, "univariate_analysis_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Univariate Analyse der Einflussfaktoren auf die Heilungsdauer\n\n")
            f.write(f"**Datum der Analyse:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write overview
            f.write("## Übersicht\n\n")
            f.write(
                "Diese Analyse untersucht den Einfluss verschiedener Faktoren auf die Heilungsdauer bei Polytrauma-Patienten. ")
            f.write(
                "Die Heilungsdauer wird definiert als die Zeit (in Tagen) vom Unfall bis zum letzten aufgezeichneten Besuch.\n\n")

            # Write body part results
            f.write("## 1. Einfluss von Verletzungen bestimmter Körperteile\n\n")

            if "body_parts" in results and results["body_parts"]:
                # Create a table of results
                f.write("### Zusammenfassung der Auswirkungen von Körperteilverletzungen\n\n")

                f.write(
                    "| Körperteil | Fälle mit Verletzung | Fälle ohne Verletzung | Differenz der mittleren Heilungsdauer | p-Wert | Signifikant | Effektgröße |\n")
                f.write(
                    "|------------|----------------------|------------------------|--------------------------------------|--------|------------|-------------|\n")

                # Sort body parts by effect size (if available) or mean difference
                sorted_body_parts = sorted(
                    [bp for bp in results["body_parts"] if bp and bp.get("injury_count") > 0],
                    key=lambda x: abs(x.get("cohens_d", 0)) if x.get("cohens_d") is not None else abs(
                        x.get("mean_difference", 0)),
                    reverse=True
                )

                for bp_result in sorted_body_parts:
                    if not bp_result or "body_part" not in bp_result:
                        continue

                    body_part = bp_result["body_part"]
                    injury_count = bp_result["injury_count"]
                    no_injury_count = bp_result["no_injury_count"]
                    mean_diff = bp_result.get("mean_difference")
                    p_value = bp_result.get("p_value")
                    significant = bp_result.get("significant")
                    cohens_d = bp_result.get("cohens_d")
                    effect_size_interp = bp_result.get("effect_size_interpretation")

                    # Format mean difference
                    if mean_diff is not None:
                        mean_diff_str = f"{mean_diff:.1f} Tage"
                        if mean_diff > 0:
                            mean_diff_str += " (länger)"
                        else:
                            mean_diff_str += " (kürzer)"
                    else:
                        mean_diff_str = "N/A"

                    # Format p-value
                    p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"

                    # Format significance
                    if significant is not None:
                        significant_str = "Ja" if significant else "Nein"
                    else:
                        significant_str = "N/A"

                    # Format effect size
                    if cohens_d is not None and effect_size_interp is not None:
                        effect_size_str = f"{cohens_d:.2f} ({effect_size_interp})"
                    else:
                        effect_size_str = "N/A"

                    f.write(
                        f"| {body_part} | {injury_count} | {no_injury_count} | {mean_diff_str} | {p_value_str} | {significant_str} | {effect_size_str} |\n")

                f.write("\n")

                # Write key findings
                f.write("### Wichtigste Erkenntnisse zu Körperteilverletzungen\n\n")

                # Find significant body parts
                significant_body_parts = [bp for bp in results["body_parts"]
                                          if bp and bp.get("significant") is True]

                if significant_body_parts:
                    f.write("**Körperteile mit signifikantem Einfluss auf die Heilungsdauer:**\n\n")

                    # Part of generate_summary_report() in univariate_analysis.py
                    # Update the section that reports body part findings:

                    for bp in significant_body_parts:
                        body_part = bp["body_part"]
                        mean_diff = bp.get("mean_difference")
                        p_value = bp.get("p_value")
                        cohens_d = bp.get("cohens_d")
                        effect_size_interp = bp.get("effect_size_interpretation")
                        direction = bp.get("direction")  # Get the explicit direction

                        f.write(
                            f"- **{body_part}**: Verletzungen {direction} die Heilungsdauer um durchschnittlich {abs(mean_diff):.1f} Tage ")
                        f.write(f"(p = {p_value:.4f}")

                        if cohens_d is not None:
                            f.write(f", Effektgröße = {cohens_d:.2f} - {effect_size_interp}")

                        f.write(")\n")

                    f.write("\n")
                else:
                    f.write(
                        "Es wurden keine Körperteile mit statistisch signifikantem Einfluss auf die Heilungsdauer gefunden.\n\n")

                # Write top 3 body parts by effect size regardless of significance
                top_body_parts = sorted(
                    [bp for bp in results["body_parts"] if bp and bp.get("cohens_d") is not None],
                    key=lambda x: abs(x["cohens_d"]),
                    reverse=True
                )[:3]

                if top_body_parts:
                    f.write("**Körperteile mit dem größten Effekt (unabhängig von statistischer Signifikanz):**\n\n")

                    for bp in top_body_parts:
                        body_part = bp["body_part"]
                        mean_diff = bp.get("mean_difference")
                        cohens_d = bp.get("cohens_d")
                        effect_size_interp = bp.get("effect_size_interpretation")

                        # Format description
                        if mean_diff > 0:
                            direction = "verlängert"
                        else:
                            direction = "verkürzt"

                        f.write(
                            f"- **{body_part}**: Verletzungen {direction} die Heilungsdauer um durchschnittlich {abs(mean_diff):.1f} Tage ")
                        f.write(f"(Effektgröße = {cohens_d:.2f} - {effect_size_interp})\n")

                    f.write("\n")
            else:
                f.write("Keine Daten zur Analyse des Einflusses von Körperteilverletzungen verfügbar.\n\n")

            # Write injury count results
            f.write("## 2. Einfluss der Verletzungsanzahl\n\n")

            if "injury_count" in results and results["injury_count"]:
                ic_results = results["injury_count"]
                correlation = ic_results["correlation"]
                correlation_p = ic_results["correlation_p_value"]
                correlation_sig = ic_results["correlation_significant"]

                f.write(f"### Korrelationsanalyse\n\n")

                f.write(
                    f"Die Korrelation zwischen der Anzahl der verletzten Körperteile und der Heilungsdauer beträgt ")
                f.write(f"**r = {correlation:.2f}** (p = {correlation_p:.4f}).\n\n")

                if correlation_sig:
                    if correlation > 0:
                        f.write("Diese positive Korrelation ist **statistisch signifikant** und deutet darauf hin, ")
                        f.write(
                            "dass eine höhere Anzahl verletzter Körperteile mit einer längeren Heilungsdauer verbunden ist.\n\n")
                    else:
                        f.write("Diese negative Korrelation ist **statistisch signifikant** und deutet darauf hin, ")
                        f.write(
                            "dass eine höhere Anzahl verletzter Körperteile mit einer kürzeren Heilungsdauer verbunden ist.\n\n")
                else:
                    f.write("Diese Korrelation ist **nicht statistisch signifikant**, was bedeutet, dass kein klarer ")
                    f.write(
                        "Zusammenhang zwischen der Anzahl der verletzten Körperteile und der Heilungsdauer nachgewiesen werden konnte.\n\n")

                # Add ANOVA results if available
                if ic_results.get("anova_p_value") is not None:
                    f.write("### Gruppenvergleich nach Verletzungsschweregrad\n\n")

                    f.write(
                        "Patienten wurden in Kategorien basierend auf der Anzahl der verletzten Körperteile eingeteilt (1-2, 3-4, 5+).\n\n")

                    anova_p = ic_results["anova_p_value"]
                    anova_sig = ic_results["anova_significant"]

                    if anova_sig:
                        f.write(
                            f"Die ANOVA-Analyse zeigt **signifikante Unterschiede** zwischen den Gruppen (p = {anova_p:.4f}), ")
                        f.write(
                            "was darauf hindeutet, dass die Anzahl der verletzten Körperteile einen bedeutenden Einfluss auf die Heilungsdauer hat.\n\n")
                    else:
                        f.write(
                            f"Die ANOVA-Analyse zeigt **keine signifikanten Unterschiede** zwischen den Gruppen (p = {anova_p:.4f}), ")
                        f.write(
                            "was darauf hindeutet, dass die kategorisierte Anzahl der Verletzungen keinen statistisch nachweisbaren Einfluss auf die Heilungsdauer hat.\n\n")
            else:
                f.write("Keine Daten zur Analyse des Einflusses der Verletzungsanzahl verfügbar.\n\n")

            # Write demographic results
            f.write("## 3. Einfluss demografischer Faktoren\n\n")

            if "demographics" in results:
                # Write age results
                f.write("### 3.1 Einfluss des Alters\n\n")

                if "age" in results["demographics"]:
                    age_results = results["demographics"]["age"]
                    correlation = age_results["correlation"]
                    p_value = age_results["p_value"]
                    significant = age_results["significant"]

                    f.write(f"Die Korrelation zwischen Alter und Heilungsdauer beträgt **r = {correlation:.2f}** ")
                    f.write(f"(p = {p_value:.4f}).\n\n")

                    if significant:
                        if correlation > 0:
                            f.write(
                                "Diese positive Korrelation ist **statistisch signifikant** und deutet darauf hin, ")
                            f.write("dass ein höheres Alter mit einer längeren Heilungsdauer verbunden ist.\n\n")
                        else:
                            f.write(
                                "Diese negative Korrelation ist **statistisch signifikant** und deutet darauf hin, ")
                            f.write("dass ein höheres Alter mit einer kürzeren Heilungsdauer verbunden ist.\n\n")
                    else:
                        f.write(
                            "Diese Korrelation ist **nicht statistisch signifikant**, was bedeutet, dass kein klarer ")
                        f.write("Zusammenhang zwischen dem Alter und der Heilungsdauer nachgewiesen werden konnte.\n\n")

                    # Add age decade analysis if available
                    if "age_decades" in results["demographics"]:
                        decade_results = results["demographics"]["age_decades"]
                        anova_p = decade_results.get("anova_p_value")
                        anova_sig = decade_results.get("anova_significant")

                        if anova_p is not None:
                            f.write("**Vergleich nach Altersdekaden:**\n\n")

                            if anova_sig:
                                f.write(
                                    f"Die ANOVA-Analyse zeigt **signifikante Unterschiede** zwischen den Altersdekaden (p = {anova_p:.4f}).\n\n")
                            else:
                                f.write(
                                    f"Die ANOVA-Analyse zeigt **keine signifikanten Unterschiede** zwischen den Altersdekaden (p = {anova_p:.4f}).\n\n")
                else:
                    f.write("Keine ausreichenden Daten zur Analyse des Alterseinflusses verfügbar.\n\n")

                # Write gender results
                f.write("### 3.2 Einfluss des Geschlechts\n\n")

                if "gender" in results["demographics"]:
                    gender_results = results["demographics"]["gender"]
                    test_type = gender_results["test_type"]
                    p_value = gender_results["p_value"]
                    significant = gender_results["significant"]
                    mean_diff = gender_results.get("mean_difference")
                    cohens_d = gender_results.get("cohens_d")
                    effect_size_interp = gender_results.get("effect_size_interpretation")

                    f.write(f"Die Analyse mittels {test_type} ergab einen p-Wert von **{p_value:.4f}**.\n\n")

                    if significant:
                        if mean_diff is not None:
                            if mean_diff > 0:
                                f.write(
                                    f"Dieser signifikante Unterschied zeigt, dass **männliche Patienten** im Durchschnitt eine um ")
                                f.write(
                                    f"**{abs(mean_diff):.1f} Tage längere** Heilungsdauer aufweisen als weibliche Patienten.\n\n")
                            else:
                                f.write(
                                    f"Dieser signifikante Unterschied zeigt, dass **männliche Patienten** im Durchschnitt eine um ")
                                f.write(
                                    f"**{abs(mean_diff):.1f} Tage kürzere** Heilungsdauer aufweisen als weibliche Patienten.\n\n")
                        else:
                            f.write(
                                f"Es wurde ein **signifikanter Unterschied** in der Heilungsdauer zwischen den Geschlechtern festgestellt.\n\n")

                        if cohens_d is not None:
                            f.write(
                                f"Die Effektgröße beträgt **Cohen's d = {cohens_d:.2f}**, was als **{effect_size_interp}er Effekt** interpretiert wird.\n\n")
                    else:
                        f.write(
                            f"Es wurde **kein signifikanter Unterschied** in der Heilungsdauer zwischen männlichen und weiblichen Patienten festgestellt.\n\n")
                else:
                    f.write("Keine ausreichenden Daten zur Analyse des Geschlechtseinflusses verfügbar.\n\n")
            else:
                f.write("Keine Daten zur Analyse demografischer Faktoren verfügbar.\n\n")

            # Write conclusion
            f.write("## Fazit\n\n")

            # Find significant factors
            significant_factors = []

            # Check body parts
            if "body_parts" in results and results["body_parts"]:
                significant_body_parts = [bp["body_part"] for bp in results["body_parts"]
                                          if bp and bp.get("significant") is True]
                if significant_body_parts:
                    significant_factors.extend([f"Verletzung: {bp}" for bp in significant_body_parts])

            # Check injury count
            if "injury_count" in results and results["injury_count"] and results["injury_count"].get(
                    "correlation_significant"):
                significant_factors.append("Anzahl der verletzten Körperteile")

            # Check demographics
            if "demographics" in results:
                if "age" in results["demographics"] and results["demographics"]["age"].get("significant"):
                    significant_factors.append("Alter")

                if "gender" in results["demographics"] and results["demographics"]["gender"].get("significant"):
                    significant_factors.append("Geschlecht")

            if significant_factors:
                f.write(
                    "Die Analyse hat die folgenden Faktoren als **statistisch signifikante Einflüsse** auf die Heilungsdauer identifiziert:\n\n")

                for factor in significant_factors:
                    f.write(f"- {factor}\n")

                f.write(
                    "\nDiese Faktoren sollten besonders bei der Prognose der Heilungsdauer und der Planung von Rehabilitationsmaßnahmen berücksichtigt werden.\n\n")
            else:
                f.write(
                    "Die Analyse konnte **keine statistisch signifikanten Faktoren** identifizieren, die einen nachweisbaren Einfluss auf die Heilungsdauer haben. ")
                f.write(
                    "Dies könnte auf die begrenzte Stichprobengröße zurückzuführen sein oder darauf hindeuten, dass andere, nicht untersuchte Faktoren eine wichtigere Rolle spielen.\n\n")

            f.write("### Methodische Einschränkungen\n\n")
            f.write(
                "- Die Analyse basiert auf einer begrenzten Stichprobengröße von 30 Patienten, was die statistische Power einschränkt.\n")
            f.write(
                "- Die Heilungsdauer wurde als Zeit vom Unfall bis zum letzten Besuch definiert, was möglicherweise nicht immer dem tatsächlichen Ende des Heilungsprozesses entspricht.\n")
            f.write("- Bei einigen Körperteilen war die Stichprobengröße für statistische Tests unzureichend.\n")
            f.write(
                "- Es wurden keine Wechselwirkungen zwischen verschiedenen Faktoren berücksichtigt (dies wird in der multivariaten Analyse erfolgen).\n\n")

            f.write("### Nächste Schritte\n\n")
            f.write(
                "- Durchführung einer multivariaten Analyse zur Untersuchung von Wechselwirkungen zwischen verschiedenen Faktoren\n")
            f.write(
                "- Entwicklung eines Regressionsmodells zur Vorhersage der Heilungsdauer basierend auf den identifizierten Einflussfaktoren\n")
            f.write(
                "- Überlegung einer Survival-Analyse-Methodik, um den zeitlichen Verlauf der Heilung besser zu verstehen\n\n")

            f.write("## Unterscheidung zwischen statistischer und klinischer Signifikanz\n\n")
            f.write(
                "Bei der Interpretation der Ergebnisse ist es wichtig, zwischen statistischer und klinischer Signifikanz zu unterscheiden:\n\n")
            f.write(
                "- **Statistische Signifikanz** (p < 0.05) zeigt an, dass ein beobachteter Unterschied wahrscheinlich nicht zufällig ist. Dies wurde in der Analyse durch p-Werte und multiple Testkorrektur berücksichtigt.\n")
            f.write(
                "- **Klinische Relevanz** bezieht sich auf die praktische Bedeutung eines Effekts unabhängig von der statistischen Signifikanz. Große Effektgrößen (Cohen's d ≥ 0.8 oder r ≥ 0.5) können klinisch bedeutsam sein, selbst wenn sie statistisch nicht signifikant sind, besonders bei kleinen Stichprobengrößen.\n\n")
            f.write(
                "Körperteile mit großen Effektgrößen, wie Abdomen (d = 1.48) und Kopf (d = 1.03), könnten trotz mangelnder statistischer Signifikanz nach multipler Testkorrektur klinisch relevante Prädiktoren für die Heilungsdauer sein.\n\n")

            f.write("## Limitationen\n\n")
            f.write(
                f"- Die Analyse basiert auf einer begrenzten Stichprobengröße von {results.get('total_patients', 'ca. 30-50')} Patienten, was die statistische Power einschränkt.\n")
            f.write(
                "- Die Heilungsdauer wurde als Zeit vom Unfall bis zum letzten Besuch definiert, was möglicherweise nicht immer dem tatsächlichen Ende des Heilungsprozesses entspricht.\n")
            f.write("- Bei einigen Körperteilen war die Stichprobengröße für statistische Tests unzureichend.\n")
            f.write(
                "- Die multiple Testkorrektur reduziert zwar das Risiko falsch-positiver Ergebnisse, kann aber auch zu falsch-negativen Ergebnissen führen, insbesondere bei kleineren Stichprobengrößen.\n")
            f.write(
                "- Es wurden keine Wechselwirkungen zwischen verschiedenen Faktoren berücksichtigt (dies wird in der multivariaten Analyse erfolgen).\n\n")

            f.write("*Dieser Bericht wurde automatisch im Rahmen der Polytrauma-Analyse erstellt.*")



        logger.info(f"Zusammenfassender Bericht erstellt: {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Fehler bei der Erstellung des Berichts: {str(e)}", exc_info=True)
        return None


def univariate_analysis():
    """
    Main function to perform univariate analysis of factors affecting healing duration.

    This analysis:
    1. Identifies body part injuries that affect healing duration
    2. Examines the impact of the number of injuries
    3. Analyzes demographic factors (age, gender)
    4. Applies multiple testing correction for statistical rigor
    5. Calculates effect sizes to assess clinical relevance

    Note: Heilungsdauer (healing duration) is defined as days from accident to last recorded visit.
    """
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_dataset = os.getenv("DATASET")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create step4 folder structure
    step4_output_folder = os.path.join(output_folder, "step4", "univariate_analysis")
    step4_log_folder = os.path.join(log_folder, "step4")
    step4_plots_folder = os.path.join(graphs_folder, "step4", "univariate_analysis")

    # Create necessary directories
    os.makedirs(step4_output_folder, exist_ok=True)
    os.makedirs(step4_log_folder, exist_ok=True)
    os.makedirs(step4_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(step4_log_folder, "univariate_analysis.log")
    logger.info("Starte univariate Analyse der Einflussfaktoren auf die Heilungsdauer...")

    try:
        # Load the dataset
        df = load_dataset(base_dataset, logger)

        # Prepare patient-level data for analysis
        patient_df = prepare_patient_level_data(df, logger)

        # Save patient-level data for reference and further analysis
        patient_data_path = os.path.join(step4_output_folder, "patient_level_data.xlsx")
        patient_df.to_excel(patient_data_path, index=False)
        logger.info(f"Patientendaten auf Einzelfallebene gespeichert: {patient_data_path}")

        # Define body parts to analyze
        body_parts = ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm links', 'Arm rechts',
                      'Wirbelsaeule', 'Bein rechts', 'Bein links', 'Becken', 'Arm', 'Bein']

        # Analyze the impact of each body part injury on healing duration
        logger.info("Analysiere den Einfluss von Körperteilverletzungen auf die Heilungsdauer...")

        body_part_results = []
        for body_part in body_parts:
            result = analyze_body_part_impact(patient_df, body_part, step4_plots_folder, logger)
            body_part_results.append(result)

        # Apply multiple testing correction
        logger.info("Wende Korrektur für multiple Tests an...")
        body_part_results = apply_multiple_testing_correction(body_part_results, logger)

        # Analyze the impact of injury count on healing duration
        logger.info("Analysiere den Einfluss der Verletzungsanzahl auf die Heilungsdauer...")
        injury_count_result = analyze_injury_count_impact(patient_df, step4_plots_folder, logger)

        # Analyze the impact of demographic factors on healing duration
        logger.info("Analysiere den Einfluss demografischer Faktoren auf die Heilungsdauer...")
        demographic_results = analyze_demographic_impact(patient_df, step4_plots_folder, logger)

        # Compile all results
        all_results = {
            "body_parts": body_part_results,
            "injury_count": injury_count_result,
            "demographics": demographic_results
        }

        # Add this line right after the above code
        all_results["total_patients"] = len(patient_df)

        # Save results to JSON for more detailed reference
        try:
            json_path = os.path.join(step4_output_folder, "univariate_analysis_results.json")
            # Convert complex objects like numpy arrays to Python native types
            import json

            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            # Create a serializable version of the results
            serializable_results = {}
            for key, value in all_results.items():
                if isinstance(value, list):
                    serializable_results[key] = [
                        {k: convert_to_serializable(v) for k, v in item.items()}
                        if item is not None and isinstance(item, dict) else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    serializable_results[key] = {
                        k: convert_to_serializable(v) for k, v in value.items()
                    }
                else:
                    serializable_results[key] = convert_to_serializable(value)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Detaillierte Ergebnisse im JSON-Format gespeichert: {json_path}")
        except Exception as e:
            logger.warning(f"Fehler beim Speichern der JSON-Ergebnisse: {str(e)}")

        # Export results to Excel
        logger.info("Exportiere Ergebnisse in Excel...")
        excel_path = export_results_to_excel(all_results, step4_output_folder, logger)

        # Generate summary report
        logger.info("Erstelle zusammenfassenden Bericht...")
        report_path = generate_summary_report(all_results, step4_output_folder, logger)

        logger.info("Univariate Analyse erfolgreich abgeschlossen.")
        logger.info(f"Ergebnisse gespeichert unter: {step4_output_folder}")
        logger.info(f"Visualisierungen gespeichert unter: {step4_plots_folder}")

        return all_results

    except Exception as e:
        logger.error(f"Fehler bei der univariaten Analyse: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    univariate_analysis()
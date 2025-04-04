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


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('critical_injury_analysis')
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


def load_patient_level_data(file_path, logger):
    """Load the patient-level dataset created during univariate analysis."""
    try:
        df = pd.read_excel(file_path, dtype={"Schadennummer": str})
        logger.info(f"Successfully loaded patient-level data from: {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading patient-level dataset: {str(e)}", exc_info=True)
        raise


def prepare_patient_level_data_if_needed(base_dataset, output_folder, logger):
    """
    Create patient-level dataset if it doesn't already exist.
    This function replicates the logic from univariate_analysis.py.
    """
    # Check univariate folder first
    univariate_path = os.path.join(output_folder, "..", "univariate_analysis", "patient_level_data.xlsx")
    critical_injury_path = os.path.join(output_folder, "patient_level_data.xlsx")

    # Check if either exists
    if os.path.exists(univariate_path):
        logger.info(f"Patient-level data found in univariate analysis folder: {univariate_path}")
        return univariate_path
    elif os.path.exists(critical_injury_path):
        logger.info(f"Patient-level data found in critical injury folder: {critical_injury_path}")
        return critical_injury_path

    # If not, create it from the base dataset (similar to univariate_analysis.py)
    logger.info("Patient-level data not found. Creating from base dataset...")

    # Implementation similar to the prepare_patient_level_data function in univariate_analysis.py
    try:
        # Load the original dataset
        df = pd.read_excel(base_dataset, dtype={"Schadennummer": str})
        df.columns = df.columns.str.strip()

        # Identify unique patients
        unique_patients = df["Schadennummer"].unique()
        logger.info(f"Number of unique patients: {len(unique_patients)}")

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
        logger.info(f"Patient-level dataset created with {len(patient_df)} cases and {len(patient_df.columns)} columns")

        # Save the dataset
        patient_df.to_excel(critical_injury_path, index=False)
        logger.info(f"Patient-level data saved to: {critical_injury_path}")

        return critical_injury_path

    except Exception as e:
        logger.error(f"Error creating patient-level data: {str(e)}", exc_info=True)
        raise


def analyze_critical_injury_impact(df, critical_regions, output_folder, plots_folder, logger):
    """
    Perform a detailed analysis of the impact of critical injuries on healing duration.

    Parameters:
    -----------
    df : pandas.DataFrame
        Patient-level dataframe
    critical_regions : list
        List of critical body regions to analyze
    output_folder : str
        Path to save analysis results
    plots_folder : str
        Path to save plots
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the critical injury analysis
    """
    logger.info(f"Starting critical injury impact analysis for: {', '.join(critical_regions)}")

    results = {}
    summary_data = []

    # Ensure all critical regions exist in the data
    missing_regions = [region for region in critical_regions if region not in df.columns]
    if missing_regions:
        logger.warning(f"Missing critical regions in data: {missing_regions}")
        critical_regions = [region for region in critical_regions if region in df.columns]

    if not critical_regions:
        logger.error("No critical regions available for analysis")
        return None

    # Convert "Ja"/"Nein" to 1/0 for analysis
    for region in critical_regions:
        if df[region].dtype == 'object':
            df[f"{region}_Binary"] = df[region].map({"Ja": 1, "Nein": 0})
            logger.info(f"Converted {region} from Ja/Nein to binary")

    # Analyze each critical region
    for region in critical_regions:
        binary_col = f"{region}_Binary"

        # Create injury and non-injury groups
        injury_group = df[df[region] == "Ja"]["Heilungsdauer"]
        no_injury_group = df[df[region] == "Nein"]["Heilungsdauer"]

        injury_count = len(injury_group)
        no_injury_count = len(no_injury_group)

        logger.info(f"Analysis for {region}: {injury_count} injury cases, {no_injury_count} non-injury cases")

        # Skip if not enough cases in either group
        if injury_count < 2 or no_injury_count < 2:
            logger.warning(f"Insufficient sample size for {region}. Need at least 2 cases per group.")
            continue

        # Calculate basic statistics
        injury_mean = injury_group.mean()
        no_injury_mean = no_injury_group.mean()
        mean_diff = injury_mean - no_injury_mean

        direction = "longer" if mean_diff > 0 else "shorter"

        injury_median = injury_group.median()
        no_injury_median = no_injury_group.median()

        # Use standardized test selection
        test_result = select_statistical_test(injury_group, no_injury_group, logger)

        test_type = test_result['test_type']
        p_value = test_result['p_value']
        test_statistic = test_result['test_statistic']

        # Check if significant
        is_significant = p_value < 0.05 if p_value is not None else None

        # Calculate effect size (Cohen's d)
        cohens_d = None
        effect_size_interp = "not calculable"

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
                    effect_size_interp = "negligible"
                elif cohens_d < 0.5:
                    effect_size_interp = "small"
                elif cohens_d < 0.8:
                    effect_size_interp = "medium"
                else:
                    effect_size_interp = "large"

                logger.info(f"{region}: Effect size (Cohen's d) = {cohens_d:.2f} ({effect_size_interp})")
            else:
                effect_size_interp = "not calculable"

        # Store results
        results[region] = {
            "injury_count": injury_count,
            "no_injury_count": no_injury_count,
            "injury_mean": injury_mean,
            "no_injury_mean": no_injury_mean,
            "mean_difference": mean_diff,
            "direction": direction,
            "injury_median": injury_median,
            "no_injury_median": no_injury_median,
            "test_type": test_type,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "significant": is_significant,
            "cohens_d": cohens_d,
            "effect_size_interpretation": effect_size_interp
        }

        # Add to summary data for results table
        summary_data.append({
            "Region": region,
            "Injury_Count": injury_count,
            "No_Injury_Count": no_injury_count,
            "Injury_Mean": injury_mean,
            "No_Injury_Mean": no_injury_mean,
            "Mean_Difference": mean_diff,
            "Direction": direction,
            "Test_Type": test_type,
            "P_Value": p_value,
            "Significant": is_significant,
            "Cohens_d": cohens_d,
            "Effect_Size_Interpretation": effect_size_interp
        })

        # Create detailed visualization for each critical region
        plt.figure(figsize=(12, 8))

        # Left subplot: Violin plot with box plot inside
        plt.subplot(1, 2, 1)
        sns.violinplot(x=df[region], y=df["Heilungsdauer"],
                       hue=df[region], palette=["#3498db", "#e74c3c"],
                       inner='box', legend=False)

        # Add individual data points
        sns.stripplot(x=df[region], y=df["Heilungsdauer"],
                      jitter=True, alpha=0.6, color='black', size=4)

        # Add mean markers
        plt.plot(["Ja"], [injury_mean], marker='o', markersize=10, color="red",
                 label=f"Mean ({injury_mean:.1f} days)")
        plt.plot(["Nein"], [no_injury_mean], marker='o', markersize=10, color="red")

        # Add titles and labels
        plt.title(f"Heilungsdauer nach {region}-Verletzung", fontsize=14, fontweight="bold")
        plt.xlabel(f"{region}-Verletzung vorhanden", fontsize=12)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)

        # Add case counts to x-axis labels
        plt.xticks([0, 1], [f"Ja (n={injury_count})", f"Nein (n={no_injury_count})"])

        # Right subplot: Histogram comparison
        plt.subplot(1, 2, 2)
        # Calculate histogram bins (same for both groups)
        all_durations = pd.concat([injury_group, no_injury_group])
        bin_min, bin_max = all_durations.min(), all_durations.max()
        bins = np.linspace(bin_min, bin_max, 15)

        # For histogram labels
        plt.hist(injury_group, bins=bins, alpha=0.5, label=f"{region}-Verletzung (Ja)",
                 color="#3498db", density=True)
        plt.hist(no_injury_group, bins=bins, alpha=0.5, label=f"{region}-Verletzung (Nein)",
                 color="#e74c3c", density=True)

        # For mean line labels
        plt.axvline(injury_mean, color="#3498db", linestyle='dashed', linewidth=2,
                    label=f"Mittelwert mit Verletzung: {injury_mean:.1f}")
        plt.axvline(no_injury_mean, color="#e74c3c", linestyle='dashed', linewidth=2,
                    label=f"Mittelwert ohne Verletzung: {no_injury_mean:.1f}")

        # Add titles and labels
        plt.title(f"Verteilung der Heilungsdauer nach {region}-Verletzung", fontsize=14, fontweight="bold")
        plt.xlabel("Heilungsdauer (Tage)", fontsize=12)
        plt.ylabel("Dichte", fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        # Add statistical results as text
        stat_text = (f"Statistical Test: {test_type}\n"
                     f"p-value: {p_value:.4f}" + (" (signifikant)" if is_significant else " (nicht signifikant)") + "\n"
                                                                                                                  f"Mittlere Differenz: {abs(mean_diff):.1f} days {direction}\n"
                                                                                                                  f"Effektgröße (Cohen's d): {cohens_d:.2f} ({effect_size_interp})")

        plt.text(0.02, 0.02, stat_text, transform=plt.subplot(1, 2, 2).transAxes, fontsize=11,
                 va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # Save the visualization
        os.makedirs(plots_folder, exist_ok=True)
        output_path = os.path.join(plots_folder, f"{region}_healing_duration_detailed.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved detailed visualization for {region} to {output_path}")

    # Create combined visualization comparing all critical regions
    if len(results) > 1:
        plt.figure(figsize=(14, 8))

        # Prepare data for plotting
        plot_data = []
        for region in results:
            # Add injury group data
            for duration in df[df[region] == "Ja"]["Heilungsdauer"]:
                plot_data.append({"Region": region, "Status": "Verletzung", "Healing_Duration": duration})
            # Add non-injury group data
            for duration in df[df[region] == "Nein"]["Heilungsdauer"]:
                plot_data.append({"Region": region, "Status": "Keine Verletzung", "Healing_Duration": duration})

        # Convert to DataFrame for easier plotting with seaborn
        plot_df = pd.DataFrame(plot_data)

        # Create grouped box plot
        sns.boxplot(x="Region", y="Healing_Duration", hue="Status", data=plot_df,
                    palette={"Verletzung": "#3498db", "Keine Verletzung": "#e74c3c"})

        # Add individual data points
        sns.stripplot(x="Region", y="Healing_Duration", hue="Status", data=plot_df,
                      dodge=True, alpha=0.6, jitter=True, size=4, color="black")

        # Add titles and labels
        plt.title("Vergleich der Heilungsdauer bei kritischen Verletzungstypen", fontsize=16, fontweight="bold")
        plt.xlabel("Kritische Region", fontsize=14)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=14)
        plt.legend(title="Verletzungsstatus")

        # Add statistical annotation
        for i, region in enumerate(results):
            res = results[region]
            if res["p_value"] is not None:
                sig_marker = "*" if res["significant"] else "ns"
                plt.text(i, plot_df["Healing_Duration"].max() * 0.95,
                         f"p={res['p_value']:.3f} {sig_marker}\nd={res['cohens_d']:.2f}",
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Save the combined visualization
        output_path = os.path.join(plots_folder, "critical_injuries_comparison.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Saved combined visualization of critical injuries to {output_path}")

    # Save results to Excel
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(output_folder, "critical_injury_analysis_results.xlsx")
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"Saved critical injury analysis results to {excel_path}")

    return results


def generate_summary_report(results, output_folder, logger):
    """Generate a summary report of the critical injury analysis."""
    report_path = os.path.join(output_folder, "kritische_verletzungen_analyse_bericht.md")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Analyse der Auswirkungen kritischer Verletzungen\n\n")
            f.write(f"**Analysedatum:** {datetime.now().strftime('%Y-%m-%d')}\n\n")

            f.write("## Übersicht\n\n")
            f.write(
                "Diese Analyse untersucht die Auswirkungen kritischer Verletzungstypen (Kopf- und Wirbelsäulenverletzungen) ")
            f.write("auf die Heilungsdauer bei Polytrauma-Patienten. Für jede kritische Region berechnen wir ")
            f.write("die mittlere Differenz der Heilungsdauer im Vergleich zu Fällen ohne Verletzung, zusammen mit ")
            f.write("Effektgrößen und statistischer Signifikanz.\n\n")

            if not results:
                f.write(
                    "**Hinweis:** Aufgrund unzureichender Daten für kritische Regionen konnten keine gültigen Ergebnisse erzielt werden.\n\n")
                return report_path

            # Key findings for each critical region
            f.write("## Hauptergebnisse\n\n")

            for region, res in results.items():
                region_name_map = {
                    "Kopf": "Kopfverletzung",
                    "Wirbelsaeule": "Wirbelsäulenverletzung"
                }
                region_name = region_name_map.get(region, region)

                f.write(f"### {region_name}\n\n")
                f.write(
                    f"- **Stichprobengröße:** {res['injury_count']} Verletzungsfälle vs. {res['no_injury_count']} Fälle ohne Verletzung\n")

                mean_diff = res['mean_difference']
                direction = "längere" if mean_diff > 0 else "kürzere"

                f.write(
                    f"- **Mittlere Differenz:** {abs(mean_diff):.1f} Tage {direction} Heilungsdauer mit {region_name}\n")

                if res['no_injury_mean'] > 0:
                    percent_diff = (mean_diff / res['no_injury_mean']) * 100
                    f.write(f"- **Prozentuale Differenz:** {abs(percent_diff):.1f}% {direction} Heilungsdauer\n")

                f.write(f"- **Statistischer Test:** {res['test_type']}, p = {res['p_value']:.4f}")
                if res['significant']:
                    f.write(" (statistisch signifikant)\n")
                else:
                    f.write(" (nicht statistisch signifikant)\n")

                if res['cohens_d'] is not None:
                    # Translate effect size interpretation to German
                    effect_size_map = {
                        "negligible": "vernachlässigbar",
                        "small": "klein",
                        "medium": "mittel",
                        "large": "groß"
                    }
                    effect_size_de = effect_size_map.get(res['effect_size_interpretation'],
                                                         res['effect_size_interpretation'])

                    f.write(f"- **Effektgröße:** Cohen's d = {res['cohens_d']:.2f} ({effect_size_de})\n")

                    # Interpretation based on effect size and direction
                    f.write("- **Interpretation:** ")
                    if res['significant']:
                        if mean_diff > 0:
                            f.write(f"Patienten mit {region_name} haben signifikant längere Heilungsdauern. ")
                        else:
                            f.write(f"Patienten mit {region_name} haben signifikant kürzere Heilungsdauern. ")

                        if res['cohens_d'] >= 0.8:
                            f.write(
                                "Die große Effektgröße deutet darauf hin, dass dieser Unterschied von erheblicher klinischer Bedeutung ist.\n\n")
                        elif res['cohens_d'] >= 0.5:
                            f.write(
                                "Die mittlere Effektgröße legt nahe, dass dieser Unterschied von moderater klinischer Bedeutung ist.\n\n")
                        else:
                            f.write(
                                "Trotz statistischer Signifikanz deutet die kleine Effektgröße darauf hin, dass dieser Unterschied eine begrenzte klinische Bedeutung haben könnte.\n\n")
                    else:
                        if res['cohens_d'] >= 0.5:
                            f.write(
                                f"Obwohl nicht statistisch signifikant, deutet die {effect_size_de} Effektgröße darauf hin, dass es einen klinisch bedeutsamen Unterschied geben könnte, der mit einer größeren Stichprobe weiter untersucht werden sollte.\n\n")
                        else:
                            f.write(
                                f"Es wurde kein statistisch signifikanter Effekt gefunden, und die kleine Effektgröße deutet auf minimale klinische Unterschiede in der Heilungsdauer basierend auf {region_name} hin.\n\n")
                else:
                    f.write("\n")

            # Comparison of critical regions
            if len(results) > 1:
                f.write("## Vergleich der kritischen Regionen\n\n")

                # Table comparing effect sizes
                f.write(
                    "| Kritische Region | Mittlere Differenz (Tage) | Prozentuale Differenz | Effektgröße (d) | Signifikanz |\n")
                f.write(
                    "|------------------|----------------------------|----------------------|----------------|-------------|\n")

                for region, res in results.items():
                    region_name = region_name_map.get(region, region)
                    mean_diff = res['mean_difference']
                    direction = "+" if mean_diff > 0 else "-"

                    percent_diff = "N/A"
                    if res['no_injury_mean'] > 0:
                        percent_diff = f"{direction}{abs(mean_diff / res['no_injury_mean'] * 100):.1f}%"

                    effect_size = "N/A"
                    if res['cohens_d'] is not None:
                        effect_size_map = {
                            "negligible": "vernachlässigbar",
                            "small": "klein",
                            "medium": "mittel",
                            "large": "groß"
                        }
                        effect_size_de = effect_size_map.get(res['effect_size_interpretation'],
                                                             res['effect_size_interpretation'])
                        effect_size = f"{res['cohens_d']:.2f} ({effect_size_de})"

                    significance = "Ja (p < 0.05)" if res['significant'] else "Nein"

                    f.write(
                        f"| {region_name} | {direction}{abs(mean_diff):.1f} | {percent_diff} | {effect_size} | {significance} |\n")

                f.write("\n")

                # Relative impact
                f.write("### Relative Auswirkung auf die Heilungsdauer\n\n")

                # Sort regions by absolute effect size
                sorted_regions = sorted(
                    [r for r in results if results[r]['cohens_d'] is not None],
                    key=lambda x: abs(results[x]['cohens_d']),
                    reverse=True
                )

                if sorted_regions:
                    most_impactful = sorted_regions[0]
                    most_impact_name = region_name_map.get(most_impactful, most_impactful)

                    f.write(
                        f"Die kritische Region mit der stärksten Auswirkung auf die Heilungsdauer ist **{most_impact_name}** ")
                    if results[most_impactful]['significant']:
                        f.write("mit einem statistisch signifikanten Effekt ")
                    else:
                        f.write("obwohl der Effekt nicht statistisch signifikant ist ")

                    f.write(f"(Cohen's d = {results[most_impactful]['cohens_d']:.2f}).\n\n")

                f.write("## Klinische Implikationen\n\n")
                f.write(
                    "Basierend auf der Analyse kritischer Verletzungen ergeben sich folgende klinische Implikationen:\n\n")

                for region, res in results.items():
                    region_name = region_name_map.get(region, region)

                    if res['significant'] and res['mean_difference'] > 0 and res['cohens_d'] >= 0.5:
                        f.write(f"1. **{region_name}:** Patienten mit diesem Verletzungstyp sollten auf potenziell ")
                        f.write(
                            f"längere Heilungsdauern überwacht werden ({abs(res['mean_difference']):.1f} Tage länger im Durchschnitt). ")
                        f.write("Erwägen Sie intensivere Rehabilitationsmaßnahmen für diese Patienten.\n\n")
                    elif res['significant'] and res['mean_difference'] < 0 and res['cohens_d'] >= 0.5:
                        f.write(
                            f"1. **{region_name}:** Trotz der Schwere der Verletzung zeigten Patienten mit diesem Verletzungstyp ")
                        f.write(
                            f"kürzere Heilungsdauern ({abs(res['mean_difference']):.1f} Tage kürzer im Durchschnitt). ")
                        f.write(
                            "Dieses kontraintuitive Ergebnis könnte Unterschiede in den Behandlungsprotokollen oder ")
                        f.write("Nachsorgemustern für diese Patienten widerspiegeln.\n\n")

            # Limitations
            f.write("## Einschränkungen\n\n")
            f.write(
                "Bei der Interpretation dieser Ergebnisse sollten folgende Einschränkungen berücksichtigt werden:\n\n")
            f.write(
                "1. **Stichprobengröße:** Die Analyse basiert auf einer begrenzten Stichprobengröße, was die statistische Aussagekraft reduzieren kann.\n")
            f.write(
                "2. **Definition der Heilungsdauer:** Die Heilungsdauer ist definiert als Zeit vom Unfall bis zum letzten erfassten Besuch, was möglicherweise nicht exakt einer vollständigen physiologischen Genesung entspricht.\n")
            f.write(
                "3. **Verletzungsschwere:** Die Analyse berücksichtigt keine Unterschiede im Schweregrad innerhalb jedes Verletzungstyps.\n")
            f.write(
                "4. **Störfaktoren:** Andere Faktoren wie Alter, Begleiterkrankungen und Behandlungsansätze können den Zusammenhang zwischen kritischen Verletzungen und Heilungsdauer beeinflussen.\n\n")

            # Recommendations
            f.write("## Empfehlungen\n\n")
            f.write("Basierend auf den Ergebnissen werden folgende Empfehlungen vorgeschlagen:\n\n")

            any_significant = any(res['significant'] for res in results.values())

            if any_significant:
                f.write(
                    "1. **Maßgeschneiderte Behandlungsplanung:** Entwickeln Sie spezifische Behandlungsprotokolle für Patienten mit ")
                significant_regions = [region_name_map.get(r, r) for r in results if results[r]['significant']]
                f.write(f"{', '.join(significant_regions)}, ")
                f.write("die die erwarteten Unterschiede in der Heilungsdauer berücksichtigen.\n\n")

                f.write("2. **Ressourcenzuweisung:** Weisen Sie Rehabilitationsressourcen unter Berücksichtigung der ")
                f.write("unterschiedlichen Auswirkungen kritischer Verletzungstypen auf die Heilungsdauer zu.\n\n")
            else:
                f.write(
                    "1. **Weitere Untersuchung:** Obwohl keine statistisch signifikanten Unterschiede gefunden wurden, ")
                f.write(
                    "deuten einige Effektgrößen auf potenziell klinisch bedeutsame Unterschiede hin. Weitere Untersuchungen mit ")
                f.write("größeren Stichprobengrößen werden empfohlen.\n\n")

            f.write(
                "3. **Integrierte Bewertung:** Beziehen Sie Informationen zu kritischen Verletzungen in prädiktive Modelle der Heilungsdauer ")
            f.write(
                "zusammen mit anderen signifikanten Faktoren ein, die in früheren Analysen identifiziert wurden.\n\n")

            f.write("*Dieser Bericht wurde automatisch im Rahmen des Polytrauma-Analyseprojekts erstellt.*")

        logger.info(f"Zusammenfassender Bericht erstellt unter: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Fehler bei der Erstellung des Berichts: {str(e)}", exc_info=True)
        return None


def critical_injury_analysis():
    """Main function to perform critical injury impact analysis."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_dataset = os.getenv("DATASET", "data/output/step1/Polytrauma_Analysis_Processed.xlsx")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create folder structure for critical injury analysis (step5)
    critical_output_folder = os.path.join(output_folder, "step5", "critical_injury_analysis")
    critical_log_folder = os.path.join(log_folder, "step5")
    critical_plots_folder = os.path.join(graphs_folder, "step5", "critical_injury_analysis")

    # Create necessary directories
    os.makedirs(critical_output_folder, exist_ok=True)
    os.makedirs(critical_log_folder, exist_ok=True)
    os.makedirs(critical_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(critical_log_folder, "critical_injury_analysis.log")
    logger.info("Starting critical injury impact analysis...")

    try:
        # Define critical regions to analyze
        critical_regions = ['Kopf', 'Wirbelsaeule']  # Head and spine as most critical
        logger.info(f"Critical regions for analysis: {', '.join(critical_regions)}")

        # Get or create patient-level data
        patient_data_path = prepare_patient_level_data_if_needed(base_dataset, critical_output_folder, logger)

        # Load patient-level data
        patient_df = load_patient_level_data(patient_data_path, logger)

        # Perform critical injury analysis
        results = analyze_critical_injury_impact(patient_df, critical_regions, critical_output_folder,
                                                 critical_plots_folder, logger)

        # Generate summary report
        if results:
            report_path = generate_summary_report(results, critical_output_folder, logger)
            logger.info(f"Analysis completed successfully. Report saved to {report_path}")
        else:
            logger.error("Analysis did not produce valid results")

        logger.info("Critical injury impact analysis completed")

    except Exception as e:
        logger.error(f"Error in critical injury analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    critical_injury_analysis()
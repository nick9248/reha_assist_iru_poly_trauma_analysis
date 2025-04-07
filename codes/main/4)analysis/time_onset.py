import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(base_log_folder, log_name):
    """Setup logging to the specified folder."""
    log_folder = os.path.join(base_log_folder, "step4")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('problem_onset_analysis')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    # Specify utf-8 encoding for the file handler to handle special characters
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
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
        logger.info(f"Columns: {list(df.columns)[:10]}...")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise


def define_problem_categories(logger):
    """Define problem categories for analysis."""
    # Define the problem categories (excluding direct body injuries)
    problem_categories = {
        "Somatisch": [
            "Somatisch-- Funktionsstoerung",
            "Somatisch-- Schmerz",
            "Somatisch--Komplikationen"
        ],
        "Personenbezogen": [
            "Personenbezogen--Psychische Probleme/Compliance",
            "Personenbezogen--Entschädigungsbegehren",
            "Personenbezogen--Migrationshintergrund",
            "Personenbezogen--Suchtverhalten",
            "Personenbezogen--Zusätzliche Erkrankungen"
        ],
        "Taetigkeit": [
            "Taetigkeit--Arbeitsunfähig",
            "Tätigkeit--Wiedereingliederung",
            "Tätigkeit--Arbeitsfaehig",
            "Tätigkeit--BU/EU",
            "Altersrentner",
            "Ehrenamt",
            "Zuverdienst"
        ],
        "Umwelt": [
            "Umwelt--Beziehungsprobleme",
            "Umwelt--Soziale Isolation",
            "Umwelt--Mobilitaetsprobleme",
            "Umwelt--Wohnsituatuation",
            "Umwelt--Finazielle Probleme"
        ],
        "Med RM": [
            "Med RM--Arzt-Vorstellung",
            "Med RM--Arzt-Wechsel",
            "Med RM--Organisation ambulante Therapie",
            "Med RM--Organisation medizinische Reha",
            "Med RM--Wietere Krankenhausaufenthalte",
            "Med RM--Psychotherapie",
            "Organisation Pflege"
        ],
        "Soziales RM": [
            "Soziales RM--Lohnersatzleistungen",
            "Soziales RM--Arbeitslosenunterstuetzung",
            "Soziales RM--Antrag auf Sozialleistungen",
            "Soziales RM--Einleitung Begutachtung"
        ],
        "Technisches RM": [
            "Technisches RM--Hilfmittelversorgung",
            "Technisches RM--Mobilitätshilfe",
            "Technisches RM--Bauliche Anpassung",
            "Technisches RM--Arbetsplatzanpassung"
        ],
        "Berufliches RM": [
            "Berufliches RM--Leistungen zur Teilhabe am Arbeitsleben",
            "Berufliches RM--Integration/berufliche Neurientierung allgemeiner Arbeitsmarkt",
            "Berufliches RM--Wiedereingliederung geförderter Arbeitsmarkt",
            "Berufliches RM--BEM"
        ]
    }

    logger.info(f"Defined {len(problem_categories)} problem categories for analysis")
    return problem_categories


def analyze_problem_onset(df, problem_categories, output_folder, plots_folder, logger):
    """
    Analyze the onset timing of different problem categories and their relationship with healing duration.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing visit records
    problem_categories : dict
        Dictionary mapping category names to lists of column names
    output_folder : str
        Path to save the output data
    plots_folder : str
        Path to save the visualizations
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Results of the analysis
    """
    logger.info("Starting problem onset timing analysis...")

    # Ensure necessary columns exist
    if "Days_Since_Accident" not in df.columns or "Schadennummer" not in df.columns:
        logger.error("Required columns 'Days_Since_Accident' or 'Schadennummer' not found")
        return None

    # Create output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Get all unique patients
    patients = df["Schadennummer"].unique()
    logger.info(f"Analyzing problem onset for {len(patients)} patients")

    # Create a dataframe to store the first occurrence of each problem category
    onset_data = []

    # For each patient, find when each problem category first occurs
    for patient_id in patients:
        patient_df = df[df["Schadennummer"] == patient_id].sort_values("Days_Since_Accident")

        # Get the healing duration (max Days_Since_Accident)
        healing_duration = patient_df["Days_Since_Accident"].max()

        # For each problem category
        for category, problems in problem_categories.items():
            # Filter to existing problem columns
            existing_problems = [p for p in problems if p in df.columns]

            if not existing_problems:
                logger.warning(f"No columns found for category {category}. Skipping.")
                continue

            # Initialize onset days to None (in case no problems occur)
            onset_days = None
            problem_found = False

            # Check each visit to find the first occurrence of any problem in this category
            for _, visit in patient_df.iterrows():
                # Check if any problem in this category is marked as "Ja"
                for problem in existing_problems:
                    if visit[problem] == "Ja":
                        onset_days = visit["Days_Since_Accident"]
                        problem_found = True
                        break

                # If a problem was found in this visit, no need to check further visits
                if problem_found:
                    break

            # Add the data to our results
            onset_data.append({
                "Schadennummer": patient_id,
                "Problem_Category": category,
                "Onset_Days": onset_days,
                "Problem_Occurred": "Ja" if problem_found else "Nein",
                "Healing_Duration": healing_duration
            })

    # Convert to DataFrame
    onset_df = pd.DataFrame(onset_data)

    # Save the raw onset data
    onset_df.to_excel(os.path.join(output_folder, "problem_onset_timing.xlsx"), index=False)
    logger.info(f"Saved problem onset data to {os.path.join(output_folder, 'problem_onset_timing.xlsx')}")

    # Calculate summary statistics for each problem category
    summary_data = []

    for category in problem_categories.keys():
        # Filter to this category and where problems occurred
        category_df = onset_df[(onset_df["Problem_Category"] == category) &
                               (onset_df["Problem_Occurred"] == "Ja")]

        if len(category_df) == 0:
            logger.info(f"No occurrences found for {category}")
            continue

        # Calculate statistics
        median_onset = category_df["Onset_Days"].median()
        mean_onset = category_df["Onset_Days"].mean()
        min_onset = category_df["Onset_Days"].min()
        max_onset = category_df["Onset_Days"].max()
        occurrence_count = len(category_df)
        occurrence_rate = len(category_df) / len(patients) * 100

        summary_data.append({
            "Problem_Category": category,
            "Median_Onset_Days": median_onset,
            "Mean_Onset_Days": mean_onset,
            "Min_Onset_Days": min_onset,
            "Max_Onset_Days": max_onset,
            "Occurrence_Count": occurrence_count,
            "Occurrence_Rate_Percent": occurrence_rate
        })

        logger.info(f"{category}: Median onset = {median_onset:.1f} days, "
                    f"Occurrence rate = {occurrence_rate:.1f}% ({occurrence_count}/{len(patients)})")

    # Create summary DataFrame and sort by median onset
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Median_Onset_Days")

    # Save the summary to Excel
    summary_df.to_excel(os.path.join(output_folder, "problem_onset_summary.xlsx"), index=False)
    logger.info(f"Saved problem onset summary to {os.path.join(output_folder, 'problem_onset_summary.xlsx')}")

    # Visualization 1: Bar chart of median onset days by problem category
    plt.figure(figsize=(12, 8))
    bars = plt.barh(summary_df["Problem_Category"], summary_df["Median_Onset_Days"],
                    color=sns.color_palette("viridis", len(summary_df)))

    # Add labels with occurrence rates
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 10,
                 bar.get_y() + bar.get_height() / 2,
                 f"{summary_df.iloc[i]['Occurrence_Rate_Percent']:.1f}% of patients",
                 va='center')

    plt.title("Medianer Zeitpunkt bis zum Auftreten nach Problemkategorie", fontsize=16, fontweight="bold")
    plt.xlabel("Tage nach dem Unfall", fontsize=14)
    plt.ylabel("Problemkategorie", fontsize=14)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "median_onset_by_category.png"), dpi=300)
    plt.close()
    logger.info(f"Saved median onset visualization to {os.path.join(plots_folder, 'median_onset_by_category.png')}")

    # Analysis: Correlation between onset timing and healing duration
    correlation_results = []

    for category in problem_categories.keys():
        # Filter to this category and where problems occurred
        category_df = onset_df[(onset_df["Problem_Category"] == category) &
                               (onset_df["Problem_Occurred"] == "Ja")]

        if len(category_df) < 5:
            logger.warning(f"Insufficient data for correlation analysis of {category}")
            continue

        # Calculate correlation
        corr, p_value = stats.pearsonr(category_df["Onset_Days"], category_df["Healing_Duration"])

        correlation_results.append({
            "Problem_Category": category,
            "Correlation": corr,
            "P_Value": p_value,
            "Significant": p_value < 0.05,
            "Sample_Size": len(category_df)
        })

        logger.info(f"{category}: Correlation between onset and healing duration: r={corr:.3f}, p={p_value:.4f}")

        # Create scatter plot for this category
        plt.figure(figsize=(10, 6))
        plt.scatter(category_df["Onset_Days"], category_df["Healing_Duration"],
                    alpha=0.7, edgecolor="k")

        # Add regression line
        x = category_df["Onset_Days"]
        if len(x) > 1:  # Need at least 2 points for regression
            m, b = np.polyfit(x, category_df["Healing_Duration"], 1)
            plt.plot(x, m * x + b, 'r--', label=f"r={corr:.2f}, p={p_value:.4f}")
            plt.legend()

        plt.title(f"{category}: Zeitpunkt des Auftretens vs. Heilungsdauer", fontsize=14, fontweight="bold")
        plt.xlabel("Tage bis zum Auftreten des Problems", fontsize=12)
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{category.replace(' ', '_')}_onset_vs_healing.png"), dpi=300)
        plt.close()

    # Create correlation summary DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Save correlation results
    if not correlation_df.empty:
        correlation_df.to_excel(os.path.join(output_folder, "onset_healing_correlation.xlsx"), index=False)
        logger.info(f"Saved correlation results to {os.path.join(output_folder, 'onset_healing_correlation.xlsx')}")

        # Visualization 2: Bar chart of correlations
        plt.figure(figsize=(12, 8))
        bars = plt.barh(correlation_df["Problem_Category"], correlation_df["Correlation"],
                        color=[plt.cm.RdBu(0.1) if c < 0 else plt.cm.RdBu(0.9) for c in correlation_df["Correlation"]])

        # Add significance markers
        for i, (_, row) in enumerate(correlation_df.iterrows()):
            if row["Significant"]:
                plt.text(row["Correlation"] + 0.01 if row["Correlation"] >= 0 else row["Correlation"] - 0.15,
                         i,
                         "*",
                         fontsize=20,
                         va='center')

        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title("Korrelation zwischen Zeitpunkt des Auftretens und Heilungsdauer", fontsize=16, fontweight="bold")
        plt.xlabel("Korrelationskoeffizient (r)", fontsize=14)
        plt.ylabel("Problemkategorie", fontsize=14)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, "onset_healing_correlation.png"), dpi=300)
        plt.close()
        logger.info(f"Saved correlation visualization to {os.path.join(plots_folder, 'onset_healing_correlation.png')}")

    # Early vs. Late Onset Analysis
    early_late_results = []

    for category in problem_categories.keys():
        # Filter to this category and where problems occurred
        category_df = onset_df[(onset_df["Problem_Category"] == category) &
                               (onset_df["Problem_Occurred"] == "Ja")]

        if len(category_df) < 10:  # Need reasonable sample size to split into early/late
            continue

        # Calculate median onset for this category
        median_onset = category_df["Onset_Days"].median()

        # Split into early and late onset groups
        early_onset = category_df[category_df["Onset_Days"] <= median_onset]["Healing_Duration"]
        late_onset = category_df[category_df["Onset_Days"] > median_onset]["Healing_Duration"]

        # Skip if either group is too small
        if len(early_onset) < 5 or len(late_onset) < 5:
            continue

        # Compare healing durations
        t_stat, p_value = stats.ttest_ind(early_onset, late_onset, equal_var=False)
        early_mean = early_onset.mean()
        late_mean = late_onset.mean()
        difference = early_mean - late_mean

        early_late_results.append({
            "Problem_Category": category,
            "Median_Split_Point": median_onset,
            "Early_Onset_Mean_Healing": early_mean,
            "Late_Onset_Mean_Healing": late_mean,
            "Difference": difference,
            "T_Statistic": t_stat,
            "P_Value": p_value,
            "Significant": p_value < 0.05,
            "Early_Sample_Size": len(early_onset),
            "Late_Sample_Size": len(late_onset)
        })

        logger.info(f"{category}: Early onset (≤{median_onset} days) mean healing = {early_mean:.1f}, "
                    f"Late onset (>{median_onset} days) mean healing = {late_mean:.1f}, "
                    f"Difference = {difference:.1f} days, p={p_value:.4f}")

        # Create box plot comparing early vs late onset
        plt.figure(figsize=(10, 6))

        # Create a dataframe for the box plot
        box_df = pd.DataFrame({
            "Auftreten": ["Frühes Auftreten (≤" + str(int(median_onset)) + " Tage)"] * len(early_onset) +
                         ["Spätes Auftreten (>" + str(int(median_onset)) + " Tage)"] * len(late_onset),
            "Heilungsdauer": list(early_onset) + list(late_onset)
        })

        # Create the box plot
        sns.boxplot(x="Auftreten", y="Heilungsdauer", data=box_df)

        # Add individual data points
        sns.stripplot(x="Auftreten", y="Heilungsdauer", data=box_df, color="black", alpha=0.5, jitter=True)

        # Add significance annotation
        if p_value < 0.05:
            y_pos = max(early_onset.max(), late_onset.max()) * 1.1
            plt.plot([0, 0, 1, 1], [y_pos, y_pos + 20, y_pos + 20, y_pos], color='k')
            plt.text(0.5, y_pos + 30, f"p = {p_value:.4f} *", ha='center')

        plt.title(f"{category}: Heilungsdauer nach Auftrittszeitpunkt", fontsize=14, fontweight="bold")
        plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
        plt.xlabel("")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"{category.replace(' ', '_')}_early_vs_late.png"), dpi=300)
        plt.close()

    # Create early vs late comparison DataFrame
    early_late_df = pd.DataFrame(early_late_results)

    # Save early vs late results
    if not early_late_df.empty:
        early_late_df.to_excel(os.path.join(output_folder, "early_vs_late_onset.xlsx"), index=False)
        logger.info(f"Saved early vs late onset results to {os.path.join(output_folder, 'early_vs_late_onset.xlsx')}")

    # Compile all results
    results = {
        "summary": summary_df.to_dict('records') if not summary_df.empty else [],
        "correlations": correlation_df.to_dict('records') if not correlation_df.empty else [],
        "early_vs_late": early_late_df.to_dict('records') if not early_late_df.empty else []
    }

    # Generate summary report
    generate_problem_onset_report(results, output_folder, logger)

    return results


def generate_problem_onset_report(results, output_folder, logger):
    """Generate a markdown report summarizing the problem onset analysis."""
    report_path = os.path.join(output_folder, "problem_onset_analysis_report.md")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Problem Onset Timing Analysis Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Overview\n\n")
            f.write("This analysis examines when different types of problems first emerge during the recovery process ")
            f.write("and how the timing of problem onset relates to overall healing duration.\n\n")

            # Write summary of onset timing
            if results["summary"]:
                f.write("## Typical Onset Timing by Problem Category\n\n")
                f.write(
                    "The following table shows when different problem categories typically first appear after the accident:\n\n")

                f.write("| Problem Category | Median Onset (Days) | Occurrence Rate | Mean Onset (Days) |\n")
                f.write("|-----------------|---------------------|-----------------|-------------------|\n")

                for row in sorted(results["summary"], key=lambda x: x["Median_Onset_Days"]):
                    f.write(f"| {row['Problem_Category']} | {row['Median_Onset_Days']:.1f} | ")
                    f.write(f"{row['Occurrence_Rate_Percent']:.1f}% ({row['Occurrence_Count']} patients) | ")
                    f.write(f"{row['Mean_Onset_Days']:.1f} |\n")

                f.write("\n")

            # Write correlation results
            if results["correlations"]:
                f.write("## Correlation Between Onset Timing and Healing Duration\n\n")
                f.write(
                    "This section examines whether earlier or later onset of problems is associated with longer healing duration:\n\n")

                f.write("| Problem Category | Correlation | p-value | Significant? | Sample Size |\n")
                f.write("|-----------------|-------------|---------|--------------|-------------|\n")

                for row in sorted(results["correlations"], key=lambda x: abs(x["Correlation"]), reverse=True):
                    f.write(f"| {row['Problem_Category']} | {row['Correlation']:.3f} | {row['P_Value']:.4f} | ")
                    f.write(f"{'Yes' if row['Significant'] else 'No'} | {row['Sample_Size']} |\n")

                f.write("\n")

                # Add interpretation
                significant_correlations = [r for r in results["correlations"] if r["Significant"]]
                if significant_correlations:
                    f.write("### Significant Correlations\n\n")

                    for row in significant_correlations:
                        direction = "positive" if row["Correlation"] > 0 else "negative"
                        interpretation = ("later onset is associated with longer healing" if row["Correlation"] > 0
                                          else "earlier onset is associated with longer healing")

                        f.write(
                            f"- **{row['Problem_Category']}**: {direction} correlation (r = {row['Correlation']:.3f}), ")
                        f.write(f"meaning {interpretation}.\n")

                    f.write("\n")

            # Write early vs. late onset results
            if results["early_vs_late"]:
                f.write("## Early vs. Late Onset Comparison\n\n")
                f.write("For each problem category, patients were divided into 'early onset' and 'late onset' groups ")
                f.write(
                    "based on the median onset time. This section compares healing duration between these groups:\n\n")

                f.write(
                    "| Problem Category | Early Onset Mean Healing | Late Onset Mean Healing | Difference | p-value | Significant? |\n")
                f.write(
                    "|-----------------|---------------------------|--------------------------|------------|---------|---------------|\n")

                for row in sorted(results["early_vs_late"], key=lambda x: abs(x["Difference"]), reverse=True):
                    f.write(f"| {row['Problem_Category']} | {row['Early_Onset_Mean_Healing']:.1f} | ")
                    f.write(f"{row['Late_Onset_Mean_Healing']:.1f} | {row['Difference']:.1f} | ")
                    f.write(f"{row['P_Value']:.4f} | {'Yes' if row['Significant'] else 'No'} |\n")

                f.write("\n")

                # Add interpretation
                significant_differences = [r for r in results["early_vs_late"] if r["Significant"]]
                if significant_differences:
                    f.write("### Significant Differences\n\n")

                    for row in significant_differences:
                        longer_group = "early onset" if row["Difference"] > 0 else "late onset"

                        f.write(f"- **{row['Problem_Category']}**: The {longer_group} group had significantly ")
                        f.write(
                            f"longer healing duration (difference of {abs(row['Difference']):.1f} days, p={row['P_Value']:.4f}).\n")

                    f.write("\n")

            # Write key findings and clinical implications
            f.write("## Key Findings\n\n")

            # Dynamically generate key findings based on results
            if results["summary"]:
                earliest_problems = sorted(results["summary"], key=lambda x: x["Median_Onset_Days"])[:3]
                latest_problems = sorted(results["summary"], key=lambda x: x["Median_Onset_Days"], reverse=True)[:3]

                f.write("### Timing of Problem Emergence\n\n")

                f.write("**Earliest emerging problems:**\n")
                for p in earliest_problems:
                    f.write(f"- {p['Problem_Category']}: {p['Median_Onset_Days']:.1f} days (median onset)\n")

                f.write("\n**Latest emerging problems:**\n")
                for p in latest_problems:
                    f.write(f"- {p['Problem_Category']}: {p['Median_Onset_Days']:.1f} days (median onset)\n")

                f.write("\n")

            if results["correlations"] or results["early_vs_late"]:
                f.write("### Impact on Healing Duration\n\n")

                if results["correlations"]:
                    strongest_correlations = sorted(results["correlations"],
                                                    key=lambda x: abs(x["Correlation"]),
                                                    reverse=True)[:3]

                    f.write("**Strongest associations with healing duration:**\n")
                    for corr in strongest_correlations:
                        direction = "later" if corr["Correlation"] > 0 else "earlier"
                        f.write(f"- {corr['Problem_Category']}: {direction} onset associated with ")
                        f.write(f"{'longer' if corr['Correlation'] > 0 else 'shorter'} healing ")
                        f.write(f"(r = {corr['Correlation']:.3f}, p = {corr['P_Value']:.4f})\n")

                    f.write("\n")

            f.write("## Clinical Implications\n\n")
            f.write(
                "Based on the analysis of problem onset timing, the following clinical implications can be drawn:\n\n")

            # Dynamically generate implications
            implications = []

            if results["summary"]:
                implications.append(
                    "1. **Anticipatory guidance**: Healthcare providers can prepare patients for the typical sequence "
                    "of problems that may emerge during recovery, based on the median onset times identified.")

            if any(r["Significant"] for r in results.get("correlations", [])):
                implications.append(
                    "2. **Critical monitoring periods**: Special attention should be paid to the timing of specific "
                    "problem types, as their early or late emergence may signal higher risk for extended recovery.")

            if any(r["Significant"] for r in results.get("early_vs_late", [])):
                implications.append(
                    "3. **Intervention timing**: For problem categories where early onset is associated with "
                    "longer healing duration, early intervention strategies should be developed and tested.")

            implications.append(
                "4. **Resource allocation**: Healthcare resources can be allocated more efficiently by anticipating "
                "when different types of problems typically emerge during the recovery process.")

            for implication in implications:
                f.write(f"{implication}\n\n")

            f.write("## Limitations\n\n")
            f.write(
                "- **Sample size**: The analysis is limited by the sample size, particularly for less common problem types.\n")
            f.write(
                "- **Visit scheduling**: The timing of problem identification depends on when visits were scheduled, "
                "which may not capture the exact onset of problems.\n")
            f.write(
                "- **Interrelated problems**: Problem categories may be interrelated, but this analysis treats them independently.\n")
            f.write(
                "- **Causality**: Correlation between onset timing and healing duration does not necessarily imply causation.\n\n")

            f.write("*This report was automatically generated as part of the Polytrauma Analysis project.*")

        logger.info(f"Generated problem onset analysis report at {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Error generating problem onset report: {e}", exc_info=True)
        return None


def problem_onset_analysis():
    """Main function to analyze the timing of problem onset and its relationship with healing duration."""
    # Load environment variables
    load_dotenv()

    # Set file paths using environment variables or adjust as needed
    data_file = os.getenv(
        "DATASET"
    )

    # Output folders
    base_output = os.getenv("OUTPUT_FOLDER")
    output_folder = os.path.join(base_output, "step4", "problem_onset_analysis")

    # Graphs folder
    base_graphs = os.getenv("GRAPHS_FOLDER")
    plots_folder = os.path.join(base_graphs, "step4", "problem_onset_analysis")

    # Log folder
    base_logs = os.getenv("LOG_FOLDER")

    # Setup logger
    logger = setup_logging(base_logs, "problem_onset_analysis.log")
    logger.info("Starting problem onset analysis...")

    try:
        # Load dataset
        df = load_dataset(data_file, logger)

        # Define problem categories
        problem_categories = define_problem_categories(logger)

        # Perform problem onset analysis
        results = analyze_problem_onset(df, problem_categories, output_folder, plots_folder, logger)

        # Log completion
        if results:
            logger.info("Problem onset analysis completed successfully.")
        else:
            logger.error("Problem onset analysis failed to produce results.")

    except Exception as e:
        logger.error(f"Error in problem onset analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    problem_onset_analysis()
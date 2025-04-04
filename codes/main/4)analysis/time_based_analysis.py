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
import json


def setup_logging(base_log_folder, log_name):
    """
    Setup logging so that logs are saved under LOG_FOLDER/step4.
    """
    # Create log folder under base_log_folder/step4
    log_folder = os.path.join(base_log_folder, "step4")
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
        logger.info(f"Columns: {list(df.columns)[:10]}...")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise


def analyze_time_based(df, output_folder, plots_folder, logger):
    """
    Perform time-based analysis on patient visit data, with German labels for plots.

    This function computes:
      - First visit (minimum Days_Since_Accident)
      - Last visit (as healing duration: maximum Days_Since_Accident)
      - Average interval between visits
      - Early intervention flags (first visit within 30, 60, and 90 days)
    It then generates several plots (in German) and exports a patient-level Excel file.

    Returns:
      dict: A dictionary with computed metrics and regression results.
    """
    logger.info("Starting time-based analysis...")

    # Group data at patient level using unique patient identifier ("Schadennummer")
    patient_groups = df.groupby("Schadennummer")

    # Calculate metrics
    first_visit = patient_groups["Days_Since_Accident"].min()
    last_visit = patient_groups["Days_Since_Accident"].max()  # Healing Duration
    healing_duration = last_visit  # Already in days

    avg_interval = patient_groups["Days_Since_Accident"].apply(
        lambda x: np.mean(np.diff(np.sort(x))) if len(x) > 1 else np.nan
    )

    # Create patient-level DataFrame
    time_df = pd.DataFrame({
        "Schadennummer": first_visit.index,
        "Erster_Besuch": first_visit.values,
        "Letzter_Besuch": last_visit.values,
        "Heilungsdauer": healing_duration.values,
        "Durchschn_Intervall": avg_interval.values
    })

    # Create early intervention flags based on first visit timing
    time_df["Fruehe_Intervention_30"] = time_df["Erster_Besuch"].apply(lambda x: "Ja" if x <= 30 else "Nein")
    time_df["Fruehe_Intervention_60"] = time_df["Erster_Besuch"].apply(lambda x: "Ja" if x <= 60 else "Nein")
    time_df["Fruehe_Intervention_90"] = time_df["Erster_Besuch"].apply(lambda x: "Ja" if x <= 90 else "Nein")

    # Calculate visit frequency (visits per month)
    time_df["Besuchsfrequenz_pro_Monat"] = (
                                                   time_df["Schadennummer"].map(patient_groups.size()) / time_df[
                                               "Heilungsdauer"]
                                           ) * 30

    logger.info(f"Calculated time metrics for {len(time_df)} patients")
    logger.info(f"Erster Besuch: min={time_df['Erster_Besuch'].min()}, max={time_df['Erster_Besuch'].max()}")
    logger.info(f"Heilungsdauer: min={time_df['Heilungsdauer'].min()}, max={time_df['Heilungsdauer'].max()}")

    # Ensure output directories exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Save patient-level metrics to Excel
    excel_output_path = os.path.join(output_folder, "patient_time_metrics.xlsx")
    time_df.to_excel(excel_output_path, index=False)
    logger.info(f"Saved patient time metrics to {excel_output_path}")

    # --- Analysis 1: Scatter Plot of First Visit vs. Healing Duration ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=time_df, x="Erster_Besuch", y="Heilungsdauer",
                    hue="Fruehe_Intervention_30", s=80, edgecolor="k")
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_df["Erster_Besuch"], time_df["Heilungsdauer"])
    x_vals = np.linspace(time_df["Erster_Besuch"].min(), time_df["Erster_Besuch"].max(), 100)
    plt.plot(x_vals, intercept + slope * x_vals, 'r--', label=f"r={r_value:.2f}, p={p_value:.4f}")

    plt.title("Erster Besuch vs. Heilungsdauer", fontsize=14, fontweight="bold")
    plt.xlabel("Tage bis erster Besuch", fontsize=12)
    plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    scatter_first_path = os.path.join(plots_folder, "erster_besuch_vs_heilungsdauer.png")
    plt.tight_layout()
    plt.savefig(scatter_first_path, dpi=300)
    plt.close()
    logger.info(f"Saved scatter plot of first visit vs. healing duration to {scatter_first_path}")

    # --- Analysis 2: Scatter Plot of Average Interval vs. Healing Duration ---
    valid_avg = time_df.dropna(subset=["Durchschn_Intervall"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=valid_avg, x="Durchschn_Intervall", y="Heilungsdauer",
                    s=80, edgecolor="k", color="#2ecc71")
    if not valid_avg.empty:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            valid_avg["Durchschn_Intervall"], valid_avg["Heilungsdauer"]
        )
        x_vals2 = np.linspace(valid_avg["Durchschn_Intervall"].min(), valid_avg["Durchschn_Intervall"].max(), 100)
        plt.plot(x_vals2, intercept2 + slope2 * x_vals2, 'r--', label=f"r={r_value2:.2f}, p={p_value2:.4f}")

    plt.title("Durchschnittliches Intervall vs. Heilungsdauer", fontsize=14, fontweight="bold")
    plt.xlabel("Durchschnittliche Tage zwischen Besuchen", fontsize=12)
    plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    scatter_interval_path = os.path.join(plots_folder, "durchschn_intervall_vs_heilungsdauer.png")
    plt.tight_layout()
    plt.savefig(scatter_interval_path, dpi=300)
    plt.close()
    logger.info(f"Saved scatter plot of average interval vs. healing duration to {scatter_interval_path}")

    # --- Analysis 3: Boxplot of Healing Duration by Early Intervention (≤30 Days) ---
    plt.figure(figsize=(8, 6))
    # Removed palette= to avoid FutureWarning
    sns.boxplot(data=time_df, x="Fruehe_Intervention_30", y="Heilungsdauer", color="lightblue")
    sns.stripplot(data=time_df, x="Fruehe_Intervention_30", y="Heilungsdauer",
                  color="black", alpha=0.7, jitter=True)

    plt.title("Heilungsdauer nach früher Intervention (≤30 Tage)", fontsize=14, fontweight="bold")
    plt.xlabel("Frühe Intervention", fontsize=12)
    plt.ylabel("Heilungsdauer (Tage)", fontsize=12)
    plt.grid(alpha=0.3)

    boxplot_path = os.path.join(plots_folder, "heilungsdauer_nach_frueher_intervention.png")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    logger.info(f"Saved boxplot for early intervention to {boxplot_path}")

    results = {
        "patient_time_metrics": time_df.to_dict(orient="list"),
        "erster_besuch_vs_heilungsdauer": {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        },
        "durchschn_intervall_vs_heilungsdauer": {
            "slope": slope2 if not valid_avg.empty else None,
            "intercept": intercept2 if not valid_avg.empty else None,
            "r_value": r_value2 if not valid_avg.empty else None,
            "p_value": p_value2 if not valid_avg.empty else None,
            "std_err": std_err2 if not valid_avg.empty else None
        }
    }

    return results


def generate_summary_report(results, output_folder, logger):
    """
    Generate a Markdown summary report for the time-based analysis (in German).
    """
    report_path = os.path.join(output_folder, "time_based_analysis_report.md")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Zeitbasierte Analyse (Time-Based Analysis) Bericht\n\n")
            f.write(f"**Analyse-Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Übersicht\n\n")
            f.write("Diese Analyse untersucht den Zusammenhang zwischen Besuchszeitpunkten und der Heilungsdauer.\n\n")

            first_reg = results.get("erster_besuch_vs_heilungsdauer", {})
            f.write("## Erster Besuch Analyse\n\n")
            f.write(f"- **Steigung (Slope):** {first_reg.get('slope', 'N/A'):.2f}\n")
            f.write(f"- **Achsenabschnitt (Intercept):** {first_reg.get('intercept', 'N/A'):.2f}\n")
            f.write(f"- **Korrelation (r):** {first_reg.get('r_value', 'N/A'):.2f}\n")
            f.write(f"- **p-Wert:** {first_reg.get('p_value', 'N/A'):.4f}\n\n")

            avg_reg = results.get("durchschn_intervall_vs_heilungsdauer", {})
            f.write("## Durchschnittliches Intervall Analyse\n\n")
            if avg_reg.get("slope") is not None:
                f.write(f"- **Steigung (Slope):** {avg_reg.get('slope'):.2f}\n")
                f.write(f"- **Achsenabschnitt (Intercept):** {avg_reg.get('intercept'):.2f}\n")
                f.write(f"- **Korrelation (r):** {avg_reg.get('r_value'):.2f}\n")
                f.write(f"- **p-Wert:** {avg_reg.get('p_value'):.4f}\n\n")
            else:
                f.write("Nicht genügend Daten für die Analyse des durchschnittlichen Intervalls.\n\n")

            f.write("## Interpretation\n\n")
            f.write(
                "Die Ergebnisse zeigen, wie der Zeitpunkt des ersten Besuchs und das durchschnittliche Intervall zwischen Besuchen\n")
            f.write(
                "mit der gesamten Heilungsdauer zusammenhängen. Diese Informationen können helfen, frühe Interventionen\n")
            f.write("und Nachsorgetermine effektiver zu planen.\n\n")

            f.write("## Einschränkungen\n\n")
            f.write("- Kleine Stichprobengröße kann die statistische Aussagekraft einschränken.\n")
            f.write(
                "- Variabilität in der Besuchsplanung kann die Berechnung des durchschnittlichen Intervalls beeinflussen.\n\n")

            f.write("*Dieser Bericht wurde automatisch als Teil des Polytrauma Analysis Projekts erstellt.*\n")
        logger.info(f"Zusammenfassender Bericht erstellt unter: {report_path}")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Berichts: {e}", exc_info=True)
        return None
    return report_path


if __name__ == "__main__":
    # Set file paths using environment variables or adjust as needed
    load_dotenv()
    data_file = os.getenv(
        "DATASET",
        r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\data\output\step1\Polytrauma_Analysis_Processed.xlsx"
    )

    # Output for time_based_analysis: step4/temporal_analysis/time_based_analysis
    base_output = os.getenv("OUTPUT_FOLDER", r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\data\output")
    output_folder = os.path.join(base_output, "step4", "temporal_analysis", "time_based_analysis")

    # Graphs folder
    base_graphs = os.getenv("GRAPHS_FOLDER", r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\plots")
    plots_folder = os.path.join(base_graphs, "step4", "temporal_analysis", "time_based_analysis")

    # Log folder base (LOG_FOLDER/step4)
    base_logs = os.getenv("LOG_FOLDER", r"C:\Users\Mohammad\PycharmProjects\Polytrauma Analysis\logs")

    # Setup logger in LOG_FOLDER/step4
    logger = setup_logging(base_logs, "time_based_analysis.log")
    logger.info("Starte das zeitbasierte Analysemodul (time-based analysis)...")

    try:
        # Load dataset
        df = load_dataset(data_file, logger)

        # Perform time-based analysis
        results = analyze_time_based(df, output_folder, plots_folder, logger)

        # Generate summary report
        report_path = generate_summary_report(results, output_folder, logger)
        logger.info(f"Analyse erfolgreich abgeschlossen. Bericht gespeichert unter {report_path}")
    except Exception as e:
        logger.error(f"Fehler in der zeitbasierten Analyse: {e}", exc_info=True)
        raise

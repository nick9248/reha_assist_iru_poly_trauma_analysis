import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from dotenv import load_dotenv
import logging
import seaborn as sns
from scipy import stats
import json
from pathlib import Path


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('injury_category_distribution')
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
        logger.info(f"Successfully loaded data from: {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise


def analyze_category_distribution(df, categories, output_folder, logger):
    """
    Analyze the distribution of patients across different injury categories.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the patient data
    categories : dict
        Dictionary mapping category names to subcategory column names
    output_folder : str
        Path to save the outputs
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Distribution statistics for each category
    """
    # Prepare for category-level aggregation
    total_cases = df['Schadennummer'].nunique()
    logger.info(f"Total distinct cases: {total_cases}")

    # Create a dictionary to store results
    category_stats = {
        "total_cases": total_cases,
        "categories": {}
    }

    # Analyze each category
    for category, subcategories in categories.items():
        logger.info(f"Analyzing category: {category}")

        # Check if all subcategories exist in the dataframe
        missing_subcategories = [col for col in subcategories if col not in df.columns]
        if missing_subcategories:
            logger.warning(f"Missing subcategories for {category}: {missing_subcategories}")
            subcategories = [col for col in subcategories if col in df.columns]

        if not subcategories:
            logger.error(f"No valid subcategories found for {category}. Skipping.")
            continue

        # Mark case as 'Ja' if any subcategory has 'Ja' across any visit
        df['Category_Flag'] = df[subcategories].eq('Ja').any(axis=1)

        # Group by case and aggregate
        case_flags = df.groupby('Schadennummer')['Category_Flag'].any()

        # Count number of cases with 'Ja' for the category
        positive_cases = case_flags.sum()
        percentage = (positive_cases / total_cases) * 100

        # Store statistics
        category_stats["categories"][category] = {
            "positive_cases": int(positive_cases),
            "percentage": float(percentage),
            "subcategories": subcategories,
            "subcategory_stats": {}
        }

        logger.info(f"Category: {category}, Positive Cases: {positive_cases}, Percentage: {percentage:.2f}%")

        # Calculate statistics for each subcategory
        for subcategory in subcategories:
            # Mark case as 'Ja' if subcategory has 'Ja' across any visit
            df['Subcategory_Flag'] = df[subcategory].eq('Ja')

            # Group by case and aggregate
            subcategory_flags = df.groupby('Schadennummer')['Subcategory_Flag'].any()

            # Count number of cases with 'Ja' for the subcategory
            positive_subcases = subcategory_flags.sum()
            subpercentage = (positive_subcases / total_cases) * 100

            # Calculate percentage relative to positive cases in the category
            if positive_cases > 0:
                relative_percentage = (positive_subcases / positive_cases) * 100
            else:
                relative_percentage = 0.0

            # Get shortened subcategory name
            if "--" in subcategory:
                subcategory_name = subcategory.split("--")[-1].strip()
            else:
                subcategory_name = subcategory

            # Store statistics
            category_stats["categories"][category]["subcategory_stats"][subcategory] = {
                "name": subcategory_name,
                "positive_cases": int(positive_subcases),
                "percentage_of_total": float(subpercentage),
                "percentage_of_category": float(relative_percentage)
            }

            logger.info(f"  Subcategory: {subcategory_name}, Positive Cases: {positive_subcases}, "
                        f"Percentage of Total: {subpercentage:.2f}%, "
                        f"Percentage of Category: {relative_percentage:.2f}%")

    return category_stats


def plot_category_distribution(category_stats, output_folder, logger):
    """
    Create enhanced visualizations for category distributions.

    Parameters:
    -----------
    category_stats : dict
        Statistics for each category from analyze_category_distribution
    output_folder : str
        Path to save the visualizations
    logger : logging.Logger
        Logger instance
    """
    # Extract data for the main histogram
    categories = []
    percentages = []
    case_counts = []

    for category, stats in category_stats["categories"].items():
        categories.append(category)
        percentages.append(stats["percentage"])
        case_counts.append(stats["positive_cases"])

    # Sort by percentage descending
    sorted_indices = np.argsort(percentages)[::-1]
    categories = [categories[i] for i in sorted_indices]
    percentages = [percentages[i] for i in sorted_indices]
    case_counts = [case_counts[i] for i in sorted_indices]

    # 1. Create enhanced main histogram
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Main bars
    bars = plt.bar(categories, percentages, color=sns.color_palette("viridis", len(categories)),
                   edgecolor="black", alpha=0.8)

    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1f}%\n({case_counts[i]})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black"
        )

    # Add titles and labels
    plt.title("Prozentsatz der Fälle pro Kategorie", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Kategorien", fontsize=14, fontweight="bold")
    plt.ylabel("Prozentsatz der Fälle (%)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.ylim(0, 110)  # Leave some space at the top for annotations

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add overall stats text box
    stats_text = f"Gesamtfälle: {category_stats['total_cases']}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "category_percentage_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Enhanced category percentage histogram saved to {output_path}")

    # 2. Create a treemap visualization
    try:
        import squarify

        # Normalize sizes for the treemap
        sizes = np.array(percentages)

        # Create a figure with a specific size
        plt.figure(figsize=(18, 12))

        # Define colors using a viridis colormap
        cmap = plt.cm.viridis
        colors = [cmap(i / len(categories)) for i in range(len(categories))]

        # Plot the treemap
        squarify.plot(sizes=sizes,
                      label=[f"{cat}\n{perc:.1f}%\n({count})" for cat, perc, count in
                             zip(categories, percentages, case_counts)],
                      color=colors,
                      alpha=0.8,
                      text_kwargs={'fontsize': 12, 'fontweight': 'bold'})

        plt.axis('off')
        plt.title("Verteilung der Verletzungskategorien", fontsize=20, fontweight="bold", pad=20)

        # Save the treemap
        output_path = os.path.join(output_folder, "category_distribution_treemap.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Category distribution treemap saved to {output_path}")
    except ImportError:
        logger.warning("squarify module not found. Skipping treemap visualization.")

    # 3. Create a radar chart (spider plot) for category percentages
    try:
        # Create a figure with a specific size
        plt.figure(figsize=(12, 10))

        # Number of categories
        N = len(categories)

        # Angle for each category
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # Close the plot by appending the first angle
        angles += angles[:1]

        # Add percentages for the plot, also closing the circle
        radar_percentages = percentages + [percentages[0]]

        # Add categories for the plot, also closing the circle
        radar_categories = categories + [categories[0]]

        # Create the plot
        ax = plt.subplot(111, polar=True)

        # Plot the percentages
        ax.plot(angles, radar_percentages, 'o-', linewidth=2, color='b')
        ax.fill(angles, radar_percentages, alpha=0.25, color='b')

        # Set the angle ticks
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_categories[:-1], fontsize=12)

        # Set y-ticks (percentage values)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)

        # Add a title
        plt.title("Radardiagramm der Kategorienverteilung", fontsize=18, fontweight="bold", y=1.1)

        # Save the radar chart
        output_path = os.path.join(output_folder, "category_distribution_radar.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Category distribution radar chart saved to {output_path}")
    except Exception as e:
        logger.warning(f"Error creating radar chart: {str(e)}")

    # 4. Create individual histogram for each category with its subcategories
    for category, stats in category_stats["categories"].items():

        # Skip if no subcategories or only one subcategory
        if not stats["subcategory_stats"] or len(stats["subcategory_stats"]) < 2:
            logger.warning(f"Skipping subcategory plot for {category}: insufficient subcategories")
            continue

        subcategories = []
        subcategory_percentages = []
        subcategory_counts = []

        for subcategory, substats in stats["subcategory_stats"].items():
            subcategories.append(substats["name"])
            subcategory_percentages.append(substats["percentage_of_total"])
            subcategory_counts.append(substats["positive_cases"])

        # Sort by percentage descending
        sorted_indices = np.argsort(subcategory_percentages)[::-1]
        subcategories = [subcategories[i] for i in sorted_indices]
        subcategory_percentages = [subcategory_percentages[i] for i in sorted_indices]
        subcategory_counts = [subcategory_counts[i] for i in sorted_indices]

        # Create figure
        plt.figure(figsize=(14, 8))

        # Main bars
        bars = plt.bar(subcategories, subcategory_percentages,
                       color=sns.color_palette("viridis", len(subcategories)),
                       edgecolor="black", alpha=0.8)

        # Add value annotations
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%\n({subcategory_counts[i]})",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="black"
            )

        # Add titles and labels
        plt.title(f"Unterkategorienverteilung für {category}", fontsize=18, fontweight="bold", pad=20)
        plt.xlabel("Unterkategorien", fontsize=14, fontweight="bold")
        plt.ylabel("Prozentsatz der Gesamtfälle (%)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.ylim(0, max(subcategory_percentages) * 1.2)  # Leave space for annotations

        # Add grid
        plt.grid(True, alpha=0.3, linestyle="--")

        # Add stats text box
        stats_text = f"Kategorie: {category}\nPositive Fälle insgesamt: {stats['positive_cases']} ({stats['percentage']:.1f}%)"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_folder, f"{category.replace(' ', '_')}_subcategory_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Subcategory distribution for {category} saved to {output_path}")


def validate_categories(category_stats, logger):
    """
    Validate the logical consistency of category statistics.

    Parameters:
    -----------
    category_stats : dict
        Statistics for each category from analyze_category_distribution
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    dict
        Validation results
    """
    total_cases = category_stats["total_cases"]
    validation_results = {
        "passed": True,
        "warnings": [],
        "errors": []
    }

    logger.info("Starting validation process.")

    # Validate each category
    for category, stats in category_stats["categories"].items():
        positive_cases = stats["positive_cases"]

        # Check if positive cases exceed total cases
        if positive_cases > total_cases:
            error_msg = f"Validierungsfehler: Kategorie '{category}' hat mehr positive Fälle ({positive_cases}) als Gesamtfälle ({total_cases})."
            validation_results["errors"].append(error_msg)
            validation_results["passed"] = False
            logger.error(error_msg)

        # Check if zero positive cases
        elif positive_cases == 0:
            warning_msg = f"Validierungswarnung: Kategorie '{category}' hat keine positiven Fälle."
            validation_results["warnings"].append(warning_msg)
            logger.warning(warning_msg)

        # Check if subcategory totals match
        subcategory_total = 0
        for subcategory, substats in stats["subcategory_stats"].items():
            subcategory_total += substats["positive_cases"]

        # If we have multiple subcategories, check that the total isn't less than any subcategory
        if len(stats["subcategory_stats"]) > 1 and subcategory_total > 0:
            max_subcategory = max(substats["positive_cases"] for substats in stats["subcategory_stats"].values())
            if positive_cases < max_subcategory:
                warning_msg = f"Validierungswarnung: Kategorie '{category}' hat weniger positive Fälle ({positive_cases}) als ihre größte Unterkategorie ({max_subcategory})."
                validation_results["warnings"].append(warning_msg)
                logger.warning(warning_msg)

        logger.info(f"Validierung bestanden: Kategorie '{category}' hat {positive_cases} positive Fälle.")

    logger.info("Validation process completed.")
    return validation_results


def generate_summary_report(category_stats, validation_results, output_folder, logger):
    """
    Generate a comprehensive summary report in Markdown format.

    Parameters:
    -----------
    category_stats : dict
        Statistics for each category from analyze_category_distribution
    validation_results : dict
        Results from validate_categories
    output_folder : str
        Path to save the report
    logger : logging.Logger
        Logger instance
    """
    total_cases = category_stats["total_cases"]

    # Create the report path
    os.makedirs(output_folder, exist_ok=True)
    report_path = os.path.join(output_folder, "category_distribution_report.md")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Analysebericht zur Verteilung der Verletzungskategorien\n\n")
            f.write(f"**Datum der Analyse:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write overview
            f.write("## Übersicht\n\n")
            f.write(
                f"Diese Analyse untersucht die Verteilung von {total_cases} Polytrauma-Fällen über verschiedene Verletzungskategorien und Unterkategorien.\n\n")

            # Add note about merged arm and leg categories
            f.write(
                "**Hinweis zur Kategorieaggregation:** 'Arm links' und 'Arm rechts' wurden zur Kategorie 'Arm' zusammengefasst. ")
            f.write("Ebenso wurden 'Bein links' und 'Bein rechts' zur Kategorie 'Bein' zusammengefasst. ")
            f.write(
                "Ein Fall wird als positiv für diese Kategorien gezählt, wenn mindestens eine der Seiten betroffen ist.\n\n")

            # Write validation results
            f.write("## Validierungsergebnisse\n\n")

            if validation_results["passed"] and not validation_results["warnings"]:
                f.write("✅ Alle Validierungsprüfungen wurden erfolgreich bestanden.\n\n")
            else:
                if validation_results["errors"]:
                    f.write("### Fehler\n\n")
                    for error in validation_results["errors"]:
                        f.write(f"❌ {error}\n")
                    f.write("\n")

                if validation_results["warnings"]:
                    f.write("### Warnungen\n\n")
                    for warning in validation_results["warnings"]:
                        f.write(f"⚠️ {warning}\n")
                    f.write("\n")

            # Write category summary
            f.write("## Kategoriezusammenfassung\n\n")
            f.write("| Kategorie | Positive Fälle | Prozentual |\n")
            f.write("|----------|----------------|------------|\n")

            # Sort categories by percentage descending
            sorted_categories = sorted(
                category_stats["categories"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=True
            )

            for category, stats in sorted_categories:
                f.write(f"| {category} | {stats['positive_cases']} | {stats['percentage']:.1f}% |\n")

            f.write("\n")

            # Write detailed subcategory analysis
            f.write("## Detaillierte Unterkategorienanalyse\n\n")

            for category, stats in sorted_categories:
                f.write(f"### {category}\n\n")
                f.write(f"- **Positive Fälle insgesamt:** {stats['positive_cases']} ({stats['percentage']:.1f}%)\n")
                f.write(f"- **Anzahl der Unterkategorien:** {len(stats['subcategory_stats'])}\n\n")

                if stats["subcategory_stats"]:
                    f.write("#### Unterkategorienaufschlüsselung\n\n")
                    f.write("| Unterkategorie | Positive Fälle | % der Gesamtfälle | % innerhalb der Kategorie |\n")
                    f.write("|-------------|----------------|------------------|-------------------|\n")

                    # Sort subcategories by percentage of total descending
                    sorted_subcategories = sorted(
                        stats["subcategory_stats"].items(),
                        key=lambda x: x[1]["percentage_of_total"],
                        reverse=True
                    )

                    for subcategory, substats in sorted_subcategories:
                        subcategory_name = substats["name"]
                        positive_cases = substats["positive_cases"]
                        percentage_of_total = substats["percentage_of_total"]
                        percentage_of_category = substats["percentage_of_category"]

                        f.write(
                            f"| {subcategory_name} | {positive_cases} | {percentage_of_total:.1f}% | {percentage_of_category:.1f}% |\n")

                f.write("\n")

            # Write visualizations section
            f.write("## Visualisierungen\n\n")
            f.write("Die folgenden Visualisierungen wurden im Rahmen dieser Analyse erstellt:\n\n")

            # List all PNG files in the output folder
            png_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]

            for png_file in png_files:
                # Create relative path for Markdown
                rel_path = png_file
                # Create a descriptive title
                title = png_file.replace("_", " ").replace(".png", "").title()
                f.write(f"### {title}\n\n")
                f.write(f"![{title}]({rel_path})\n\n")

            # Write methodological notes
            f.write("## Methodische Hinweise\n\n")
            f.write(
                "- Ein Fall wird als positiv für eine Kategorie gezählt, wenn eine ihrer Unterkategorien über beliebige Besuche hinweg einen 'Ja'-Wert aufweist.\n")
            f.write("- Prozentsätze werden auf Basis der Gesamtzahl der verschiedenen Fälle berechnet.\n")
            f.write(
                "- Prozentsätze der Unterkategorien werden sowohl als Anteil der Gesamtfälle als auch als Anteil der positiven Fälle innerhalb der Kategorie dargestellt.\n")
            f.write("- Fehlende Werte in kategorischen Spalten wurden während der Verarbeitung als 'Nein' behandelt.\n")
            f.write(
                "- Die Kategorien 'Arm' und 'Bein' sind aggregierte Kategorien, die jeweils die linke und rechte Seite zusammenfassen.\n")

            f.write("\n## Schlussfolgerung\n\n")

            # Identify top categories
            top_categories = sorted(
                category_stats["categories"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=True
            )[:3]

            # Identify lowest categories
            bottom_categories = sorted(
                category_stats["categories"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=False
            )[:3]

            f.write("### Wichtigste Erkenntnisse\n\n")

            f.write("**Häufigste Kategorien:**\n")
            for category, stats in top_categories:
                f.write(f"- {category}: {stats['percentage']:.1f}% ({stats['positive_cases']} Fälle)\n")

            f.write("\n**Seltenste Kategorien:**\n")
            for category, stats in bottom_categories:
                f.write(f"- {category}: {stats['percentage']:.1f}% ({stats['positive_cases']} Fälle)\n")

            f.write("\n*Dieser Bericht wurde automatisch im Rahmen der erweiterten Analyse erstellt.*")

        logger.info(f"Summary report saved to: {report_path}")

        # Also export the data as JSON for further analysis
        json_path = os.path.join(output_folder, "category_statistics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(category_stats, f, indent=2)

        logger.info(f"Category statistics exported to JSON: {json_path}")

    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}", exc_info=True)


def injury_sub_category_distribution():
    """Main function to perform the optimized injury category distribution analysis."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment variables
    base_dataset = os.getenv("DATASET")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    graphs_folder = os.getenv("GRAPHS_FOLDER", "plots")

    # Create step3 folder structure
    step3_output_folder = os.path.join(output_folder, "step3")
    step3_log_folder = os.path.join(log_folder, "step3")
    step3_plots_folder = os.path.join(graphs_folder, "step3", "injury_subcategory_distribution")

    # Create necessary directories
    os.makedirs(step3_output_folder, exist_ok=True)
    os.makedirs(step3_log_folder, exist_ok=True)
    os.makedirs(step3_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(step3_log_folder, "injury_subcategory_distribution.log")
    logger.info("Starting enhanced injury category distribution analysis...")

    try:
        # Load the dataset
        df = load_dataset(base_dataset, logger)

        # Create merged fields for Arm and Bein
        logger.info("Creating merged categories for Arm and Bein...")

        # Merge Arm links and Arm rechts into a single Arm category
        df['Arm'] = df[['Arm links', 'Arm rechts']].apply(
            lambda row: 'Ja' if 'Ja' in row.values else 'Nein', axis=1
        )

        # Merge Bein links and Bein rechts into a single Bein category
        df['Bein'] = df[['Bein links', 'Bein rechts']].apply(
            lambda row: 'Ja' if 'Ja' in row.values else 'Nein', axis=1
        )

        logger.info("Merged fields created successfully")

        # Updated categories with merged Arm and Bein
        categories = {
            "Körperteil": ['Kopf', 'Hals', 'Thorax', 'Abdomen', 'Arm', 'Wirbelsaeule',
                           'Bein', 'Becken'],  # Using merged fields
            "Somatisch": ['Somatisch-- Funktionsstoerung', 'Somatisch-- Schmerz', 'Somatisch--Komplikationen'],
            "Personenbezogen": ['Personenbezogen--Psychische Probleme/Compliance',
                                'Personenbezogen--Entschädigungsbegehren',
                                'Personenbezogen--Migrationshintergrund', 'Personenbezogen--Suchtverhalten',
                                'Personenbezogen--Zusätzliche Erkrankungen'],
            "Taetigkeit": ['Taetigkeit--Arbeitsunfähig', 'Tätigkeit--Wiedereingliederung', 'Tätigkeit--Arbeitsfaehig',
                           'Tätigkeit--BU/EU', 'Altersrentner', 'Ehrenamt', 'Zuverdienst'],
            "Umwelt": ['Umwelt--Beziehungsprobleme', 'Umwelt--Soziale Isolation', 'Umwelt--Mobilitaetsprobleme',
                       'Umwelt--Wohnsituatuation', 'Umwelt--Finazielle Probleme'],
            "Med RM": ['Med RM--Arzt-Vorstellung', 'Med RM--Arzt-Wechsel', 'Med RM--Organisation ambulante Therapie',
                       'Med RM--Organisation medizinische Reha', 'Med RM--Wietere Krankenhausaufenthalte',
                       'Med RM--Psychotherapie', 'Organisation Pflege'],
            "Soziales RM": ['Soziales RM--Lohnersatzleistungen', 'Soziales RM--Arbeitslosenunterstuetzung',
                            'Soziales RM--Antrag auf Sozialleistungen', 'Soziales RM--Einleitung Begutachtung'],
            "Technisches RM": ['Technisches RM--Hilfmittelversorgung', 'Technisches RM--Mobilitätshilfe',
                               'Technisches RM--Bauliche Anpassung', 'Technisches RM--Arbetsplatzanpassung'],
            "Berufliches RM": ['Berufliches RM--Leistungen zur Teilhabe am Arbeitsleben',
                               'Berufliches RM--Integration/berufliche Neurientierung allgemeiner Arbeitsmarkt',
                               'Berufliches RM--Wiedereingliederung geförderter Arbeitsmarkt', 'Berufliches RM--BEM']
        }

        # Add explanation to the log about the merged categories
        logger.info("Using modified category structure with merged Arm and Bein categories")
        logger.info("Arm category: Combines 'Arm links' and 'Arm rechts'")
        logger.info("Bein category: Combines 'Bein links' and 'Bein rechts'")

        # Analyze category distribution
        logger.info("Analyzing category distribution...")
        category_stats = analyze_category_distribution(df, categories, step3_plots_folder, logger)

        # Validate category statistics
        logger.info("Validating category statistics...")
        validation_results = validate_categories(category_stats, logger)

        # Plot category distribution
        logger.info("Creating enhanced visualizations...")
        plot_category_distribution(category_stats, step3_plots_folder, logger)

        # Generate summary report
        logger.info("Generating summary report...")
        generate_summary_report(category_stats, validation_results, step3_plots_folder, logger)

        # Save category statistics to Excel
        excel_path = os.path.join(step3_output_folder, "injury_subcategory_statistics.xlsx")

        # Create a dataframe from the category statistics
        category_df = pd.DataFrame([
            {
                "Kategorie": category,
                "Positive_Fälle": stats["positive_cases"],
                "Prozentsatz": stats["percentage"],
                "Unterkategorien_Anzahl": len(stats["subcategory_stats"])
            }
            for category, stats in category_stats["categories"].items()
        ])

        # Create a dataframe from the subcategory statistics
        subcategory_rows = []
        for category, stats in category_stats["categories"].items():
            for subcategory, substats in stats["subcategory_stats"].items():
                subcategory_rows.append({
                    "Kategorie": category,
                    "Unterkategorie": subcategory,
                    "Name": substats["name"],
                    "Positive_Fälle": substats["positive_cases"],
                    "Prozentsatz_der_Gesamtfälle": substats["percentage_of_total"],
                    "Prozentsatz_der_Kategorie": substats["percentage_of_category"]
                })

        subcategory_df = pd.DataFrame(subcategory_rows)

        # Create an Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            category_df.to_excel(writer, sheet_name="Kategorien", index=False)
            subcategory_df.to_excel(writer, sheet_name="Unterkategorien", index=False)

            # Add a summary sheet
            summary_df = pd.DataFrame([
                {"Metrik": "Gesamtfälle", "Wert": category_stats["total_cases"]},
                {"Metrik": "Anzahl der Kategorien", "Wert": len(category_stats["categories"])},
                {"Metrik": "Kategorien mit 100% Abdeckung",
                 "Wert": sum(1 for stats in category_stats["categories"].values()
                             if stats["percentage"] == 100.0)},
                {"Metrik": "Kategorien mit <50% Abdeckung",
                 "Wert": sum(1 for stats in category_stats["categories"].values()
                             if stats["percentage"] < 50.0)},
                {"Metrik": "Hinweis zu Arm/Bein",
                 "Wert": "Arm links/rechts und Bein links/rechts wurden zusammengefasst"}
            ])

            summary_df.to_excel(writer, sheet_name="Zusammenfassung", index=False)

        logger.info(f"Category statistics exported to Excel: {excel_path}")

        # Copy the summary report to the main output folder
        import shutil
        report_source = os.path.join(step3_plots_folder, "subcategory_distribution_report.md")
        report_dest = os.path.join(step3_output_folder, "subcategory_distribution_report.md")

        if os.path.exists(report_source):
            shutil.copy2(report_source, report_dest)
            logger.info(f"Summary report copied to: {report_dest}")

        logger.info("Enhanced injury category distribution analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error in injury category distribution analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    injury_sub_category_distribution()
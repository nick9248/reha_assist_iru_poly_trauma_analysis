import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from dotenv import load_dotenv
import logging
from pathlib import Path


def setup_logging(log_folder, log_name):
    """Set up logging configuration with proper format."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)

    logger = logging.getLogger('injury_category_analysis')
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
        logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns only
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise


def create_merged_categories(df, logger):
    """Create merged categories for extremities."""
    # Define updated body part injury structure
    updated_body_part_columns = {
        "Kopf": ["Kopf"],
        "Hals": ["Hals"],
        "Thorax": ["Thorax"],
        "Abdomen": ["Abdomen"],
        "Bein": ["Bein links", "Bein rechts"],  # Merging both legs into "Bein"
        "Arm": ["Arm links", "Arm rechts"],  # Merging both arms into "Arm"
        "Wirbelsaeule": ["Wirbelsaeule"],
        "Becken": ["Becken"],
    }

    # Create new merged columns for "Bein" and "Arm"
    for new_col, old_cols in updated_body_part_columns.items():
        if len(old_cols) == 1:
            # Direct copy for non-merged categories
            df[new_col] = df[old_cols[0]]
        else:
            # For merged categories, check if any of the old columns has "Ja"
            df[new_col] = df[old_cols].apply(lambda row: "Ja" if "Ja" in row.values else "Nein", axis=1)

    logger.info("Created merged categories for extremities (Arm and Bein)")
    return df, list(updated_body_part_columns.keys())


def analyze_categories(df, categories, logger):
    """Analyze the prevalence of each main category."""
    total_cases = df["Schadennummer"].nunique()
    logger.info(f"Total distinct cases: {total_cases}")

    # Dictionary to store category results
    category_results = {
        "total_cases": total_cases,
        "categories": {}
    }

    # Analyze each category
    for category_name, columns in categories.items():
        logger.info(f"Analyzing category: {category_name}")

        # Check for missing columns
        valid_columns = [col for col in columns if col in df.columns]
        if len(valid_columns) != len(columns):
            missing = [col for col in columns if col not in df.columns]
            logger.warning(f"Missing columns for {category_name}: {missing}")
            if not valid_columns:
                logger.error(f"No valid columns for {category_name}. Skipping.")
                continue

        # Replace NaN with "Nein" in the columns
        df[valid_columns] = df[valid_columns].fillna("Nein")

        # Analyze at category level
        # A case is positive if ANY of the subcategories has "Ja" in ANY visit
        df_category = df.groupby("Schadennummer")[valid_columns].apply(
            lambda g: (g == "Ja").any().any()
        )

        # Count positive cases
        positive_cases = df_category.sum()
        percentage = (positive_cases / total_cases) * 100

        # Store results
        category_results["categories"][category_name] = {
            "positive_cases": int(positive_cases),
            "percentage": float(percentage),
            "column_count": len(valid_columns)
        }

        logger.info(f"  Result: {positive_cases} positive cases ({percentage:.2f}%)")

        # Analyze at column level (within category)
        df_columns = df.groupby("Schadennummer")[valid_columns].apply(
            lambda g: (g == "Ja").any()
        )
        column_counts = df_columns.sum()
        column_percentages = (column_counts / total_cases) * 100

        # Store column-level results
        category_results["categories"][category_name]["columns"] = {}
        for col in valid_columns:
            column_name = col.split("--")[-1].strip() if "--" in col else col
            category_results["categories"][category_name]["columns"][col] = {
                "name": column_name,
                "positive_cases": int(column_counts[col]),
                "percentage": float(column_percentages[col])
            }
            logger.info(f"    {column_name}: {column_counts[col]} cases ({column_percentages[col]:.2f}%)")

    return category_results


def visualize_category_comparison(results, output_folder, logger):
    """Create visualizations comparing the main categories."""
    # Extract data for plotting
    categories = []
    positive_cases = []
    percentages = []

    for cat_name, cat_data in sorted(
            results["categories"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
    ):
        categories.append(cat_name)
        positive_cases.append(cat_data["positive_cases"])
        percentages.append(cat_data["percentage"])

    total_cases = results["total_cases"]

    # 1. Create basic comparison bar chart
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Create the bars
    bars = plt.bar(categories, percentages, color=sns.color_palette("viridis", len(categories)))

    # Add percentage and count labels
    for bar, count in zip(bars, positive_cases):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f'{height:.1f}%\n({count}/{total_cases})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add titles and labels
    plt.title('Vergleich der Hauptverletzungskategorien', fontsize=18, fontweight='bold')  # Changed to German
    plt.xlabel('Kategorie', fontsize=14)  # Changed to German
    plt.ylabel('Prozentsatz der Fälle (%)', fontsize=14)  # Changed to German
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylim(0, max(percentages) * 1.15)  # Add some space for labels
    plt.grid(axis='y', alpha=0.3)

    # Save the figure
    output_path = os.path.join(output_folder, "category_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved category comparison chart to {output_path}")

    # 2. Create a horizontal bar chart for easier comparison
    plt.figure(figsize=(10, 8))

    # Sort categories by percentage
    sorted_indices = np.argsort(percentages)
    sorted_cats = [categories[i] for i in sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]
    sorted_counts = [positive_cases[i] for i in sorted_indices]

    # Create horizontal bars
    bars = plt.barh(sorted_cats, sorted_percentages,
                    color=sns.color_palette("viridis", len(sorted_cats)))

    # Add labels
    for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
        width = bar.get_width()
        plt.text(width + 1, i, f'{width:.1f}% ({count}/{total_cases})',
                 va='center', fontsize=10, fontweight='bold')

    # Add titles and labels
    plt.title('Verletzungskategorien (sortiert nach Prävalenz)', fontsize=18, fontweight='bold')  # Changed to German
    plt.xlabel('Prozentsatz der Fälle (%)', fontsize=14)  # Changed to German
    plt.grid(axis='x', alpha=0.3)

    # Save the figure
    output_path = os.path.join(output_folder, "category_comparison_horizontal.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved horizontal category comparison chart to {output_path}")

    # 3. Create a pie chart for overall distribution
    plt.figure(figsize=(12, 10))

    # We'll highlight the top categories and group the rest
    if len(categories) > 5:
        # Take top 5 categories
        top_indices = np.argsort(percentages)[-5:]
        top_cats = [categories[i] for i in top_indices]
        top_percentages = [percentages[i] for i in top_indices]

        # Group the rest
        other_indices = [i for i in range(len(categories)) if i not in top_indices]
        other_percentage = sum(percentages[i] for i in other_indices)

        # Create pie chart data
        pie_labels = top_cats + ['Andere Kategorien']  # Changed to German
        pie_sizes = top_percentages + [other_percentage]

        # Explode the first slice (largest percentage)
        explode = [0.1] + [0] * (len(pie_labels) - 1)
    else:
        pie_labels = categories
        pie_sizes = percentages
        explode = [0.1] + [0] * (len(categories) - 1)

    # Create pie chart
    plt.pie(pie_sizes, explode=explode, labels=pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=sns.color_palette("viridis", len(pie_labels)))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Add title
    plt.title('Verteilung der Verletzungskategorien', fontsize=18, fontweight='bold')  # Changed to German

    # Save the figure
    output_path = os.path.join(output_folder, "category_distribution_pie.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved category distribution pie chart to {output_path}")

    # 4. Create a radar chart to visualize the coverage across all domains
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Prepare the data
    theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    # Close the plot
    values = percentages + [percentages[0]]
    theta = np.append(theta, theta[0])

    # Plot the data
    ax.plot(theta, values, 'o-', linewidth=2)
    ax.fill(theta, values, alpha=0.25)

    # Set the angle labels
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Set radial limits
    ax.set_ylim(0, 100)

    # Set radial ticks and labels
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])

    # Add title
    plt.title('Abdeckung über Verletzungskategorien', fontsize=16, y=1.1)  # Changed to German

    # Save the figure
    output_path = os.path.join(output_folder, "category_radar_chart.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved category radar chart to {output_path}")


def create_heatmap(results, output_folder, logger):
    """Create a heatmap visualizing the relationship between top columns across categories."""
    # Extract data for the heatmap
    columns_data = []

    for cat_name, cat_data in results["categories"].items():
        for col_name, col_data in cat_data["columns"].items():
            columns_data.append({
                "category": cat_name,
                "column": col_data["name"],
                "full_column": col_name,
                "percentage": col_data["percentage"],
                "positive_cases": col_data["positive_cases"]
            })

    # Create a DataFrame for easier manipulation
    df_columns = pd.DataFrame(columns_data)

    # Fix for the deprecation warning - select top 3 columns from each category based on percentage
    # Instead of using groupby.apply, we'll use a different approach
    top_columns_list = []
    for category in df_columns['category'].unique():
        category_df = df_columns[df_columns['category'] == category]
        top3 = category_df.nlargest(3, 'percentage')
        top_columns_list.append(top3)

    # Combine all top columns
    top_columns = pd.concat(top_columns_list)

    # Pivot to create the heatmap data
    heatmap_data = top_columns.pivot_table(
        index="category",
        columns="column",
        values="percentage",
        aggfunc="first"
    ).fillna(0)

    # Create the heatmap
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=.5,
        cbar_kws={"label": "Prozentsatz der Fälle (%)"}  # Changed to German
    )

    # Adjust labels and title
    plt.title("Top-Unterkategorien nach Hauptkategorien", fontsize=18, fontweight="bold")  # Changed to German
    plt.xlabel("Unterkategorie", fontsize=14)  # Changed to German
    plt.ylabel("Hauptkategorie", fontsize=14)  # Changed to German

    # Rotate column labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=12)

    # Save the figure
    output_path = os.path.join(output_folder, "top_subcategories_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved top subcategories heatmap to {output_path}")


def export_results(results, step3_output_folder, logger, filename_prefix=""):
    """Export the analysis results to Excel and JSON files."""
    # Export to Excel
    excel_path = os.path.join(step3_output_folder, f"{filename_prefix}injury_category_analysis.xlsx")

    # Create summary DataFrame
    summary_data = []
    for cat_name, cat_data in sorted(
            results["categories"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
    ):
        summary_data.append({
            "Category": cat_name,
            "Positive Cases": cat_data["positive_cases"],
            "Percentage": cat_data["percentage"],
            "Subcategory Count": cat_data["column_count"]
        })

    summary_df = pd.DataFrame(summary_data)

    # Create subcategory DataFrame
    subcategory_data = []
    for cat_name, cat_data in results["categories"].items():
        for col_name, col_data in cat_data["columns"].items():
            subcategory_data.append({
                "Category": cat_name,
                "Subcategory": col_data["name"],
                "Original Column": col_name,
                "Positive Cases": col_data["positive_cases"],
                "Percentage": col_data["percentage"]
            })

    subcategory_df = pd.DataFrame(subcategory_data)

    # Export to Excel with multiple sheets
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Category Summary", index=False)
        subcategory_df.to_excel(writer, sheet_name="Subcategory Details", index=False)

        # Add a metadata sheet
        metadata = pd.DataFrame([
            {"Key": "Total Cases", "Value": results["total_cases"]},
            {"Key": "Analysis Date", "Value": pd.Timestamp.now().strftime("%Y-%m-%d")},
            {"Key": "Total Categories", "Value": len(results["categories"])},
            {"Key": "Top Category", "Value": summary_df.iloc[0]["Category"]},
            {"Key": "Top Category Percentage", "Value": f"{summary_df.iloc[0]['Percentage']:.2f}%"}
        ])
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

    logger.info(f"Exported analysis results to Excel: {excel_path}")

    # Export to JSON
    import json
    json_path = os.path.join(step3_output_folder, f"{filename_prefix}injury_category_analysis.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Exported analysis results to JSON: {json_path}")

    return excel_path, json_path


def create_non_body_summary_report(results, excel_path, step3_output_folder, logger):
    """Create a German markdown summary report for non-body categories."""
    report_path = os.path.join(step3_output_folder, "nicht_koerperliche_kategorien_bericht.md")

    with open(report_path, "w", encoding="utf-8") as f:
        # German headers and content
        f.write("# Analyse der nicht-körperlichen Verletzungskategorien\n\n")
        f.write(f"**Analysedatum:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        # Overview
        f.write("## Überblick\n\n")
        f.write(
            f"Diese Analyse untersucht die Verteilung von nicht-körperlichen Verletzungskategorien bei {results['total_cases']} ")
        f.write(
            "individuellen Patientenfällen. Sie gibt Einblicke in die Prävalenz verschiedener Kategorien und deren Beziehungen.\n\n")

        # Key findings
        f.write("## Wichtigste Erkenntnisse\n\n")

        # Sort categories by percentage
        sorted_categories = sorted(
            results["categories"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )

        # Top 3 most common categories
        f.write("### Häufigste Kategorien\n\n")
        f.write("| Rang | Kategorie | Fälle | Prozent |\n")
        f.write("|------|-----------|-------|--------|\n")

        for i, (cat_name, cat_data) in enumerate(sorted_categories[:min(3, len(sorted_categories))]):
            f.write(f"| {i + 1} | {cat_name} | {cat_data['positive_cases']} | {cat_data['percentage']:.1f}% |\n")

        f.write("\n")

        # Least common categories (bottom 3 or fewer)
        bottom_count = min(3, len(sorted_categories))
        if bottom_count > 0:
            f.write("### Seltenste Kategorien\n\n")
            f.write("| Rang | Kategorie | Fälle | Prozent |\n")
            f.write("|------|-----------|-------|--------|\n")

            bottom_categories = sorted_categories[-bottom_count:]
            bottom_categories.reverse()  # Show least common first

            for i, (cat_name, cat_data) in enumerate(bottom_categories):
                f.write(
                    f"| {len(sorted_categories) - i} | {cat_name} | {cat_data['positive_cases']} | {cat_data['percentage']:.1f}% |\n")

            f.write("\n")

        # Top subcategories
        f.write("### Top-Unterkategorien nach Kategorie\n\n")

        for cat_name, cat_data in sorted_categories:
            f.write(f"#### {cat_name}\n\n")

            # Sort columns by percentage
            sorted_columns = sorted(
                cat_data["columns"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=True
            )

            # Show top 3 or all if less than 3
            top_columns = sorted_columns[:min(3, len(sorted_columns))]

            f.write("| Unterkategorie | Fälle | Prozent |\n")
            f.write("|---------------|-------|--------|\n")

            for col_name, col_data in top_columns:
                f.write(f"| {col_data['name']} | {col_data['positive_cases']} | {col_data['percentage']:.1f}% |\n")

            f.write("\n")

        # [Methodology, Resources, and Conclusion sections similar to the original but in German]

        f.write("## Methodik\n\n")
        # [German methodology content]

        f.write("## Ressourcen\n\n")
        # [German resources content]

        f.write("## Fazit\n\n")
        # [German conclusion with most/least common categories]

        f.write("*Dieser Bericht wurde automatisch im Rahmen des Polytrauma-Analyseprojekts erstellt.*")

    logger.info(f"Created non-body summary report: {report_path}")
    return report_path

def create_body_part_summary_report(results, excel_path, step3_output_folder, logger, patients_without_injury):
    """Create a German markdown summary report for body part injuries."""
    report_path = os.path.join(step3_output_folder, "koerperteil_verletzungen_bericht.md")

    with open(report_path, "w", encoding="utf-8") as f:
        # German headers and content
        f.write("# Analyse der Körperteilverletzungen\n\n")
        f.write(f"**Analysedatum:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        # Overview
        f.write("## Überblick\n\n")
        f.write(f"Diese Analyse untersucht die Verteilung von Körperteilverletzungen bei {results['total_cases']} ")
        f.write(
            "individuellen Patientenfällen. Sie gibt Einblicke in die Prävalenz verschiedener Verletzungstypen.\n\n")

        # Patient without injury (if any)
        if patients_without_injury:
            f.write("## Patient ohne Körperteilverletzung\n\n")
            f.write(
                f"Ein Patient (Schadennummer: {', '.join(patients_without_injury)}) weist keine Körperteilverletzung auf. ")
            f.write(
                "Dieser Fall könnte weitere Untersuchungen erfordern, um die spezifischen Umstände zu verstehen.\n\n")

        # Sort body parts by percentage
        body_part_category = list(results["categories"].values())[0]  # There's only one category: "Körperteil"

        # Extract and sort body parts
        sorted_body_parts = sorted(
            body_part_category["columns"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )

        # Body part distribution
        f.write("## Verteilung der Körperteilverletzungen\n\n")
        f.write("| Körperteil | Fälle mit Verletzung | Prozent |\n")
        f.write("|------------|----------------------|--------|\n")

        for body_part, data in sorted_body_parts:
            f.write(f"| {data['name']} | {data['positive_cases']} | {data['percentage']:.1f}% |\n")

        f.write("\n")

        # Most common body part injuries
        f.write("## Häufigste Körperteilverletzungen\n\n")

        # Get the top 3 or fewer if less than 3 are available
        top_body_parts = sorted_body_parts[:min(3, len(sorted_body_parts))]

        for i, (body_part, data) in enumerate(top_body_parts):
            rank = i + 1
            f.write(f"{rank}. **{data['name']}**: {data['positive_cases']} Patienten ({data['percentage']:.1f}%)\n")

        f.write("\n")

        # Methodology
        f.write("## Methodik\n\n")
        f.write("Diese Analyse wurde mit folgender Methodik durchgeführt:\n\n")
        f.write("1. Jede eindeutige 'Schadennummer' repräsentiert einen individuellen Patientenfall\n")
        f.write(
            "2. Ein Patient wird als positiv für eine Verletzung gezählt, wenn bei irgendeinem Besuch 'Ja' vermerkt wurde\n")
        f.write(
            "3. Für Extremitäten wurden 'Arm links'/'Arm rechts' zu 'Arm' und 'Bein links'/'Bein rechts' zu 'Bein' zusammengefasst\n")
        f.write("4. Fehlende Werte wurden als 'Nein' behandelt\n")
        f.write("5. Prozentsätze werden auf Basis der Gesamtzahl der eindeutigen Fälle berechnet\n\n")

        # Resources
        f.write("## Ressourcen\n\n")
        f.write(f"- Vollständige Analyseergebnisse: [Excel-Datei]({os.path.basename(excel_path)})\n")
        f.write("- Visualisierungen:\n")
        f.write("  - Vergleichsdiagramm der Körperteile\n")
        f.write("  - Horizontales Vergleichsdiagramm\n")
        f.write("  - Kreisdiagramm der Verteilung\n")
        f.write("  - Radardiagramm der Körperteile\n")
        f.write("  - Heatmap der häufigsten Körperteilverletzungen\n\n")

        # Conclusion
        f.write("## Fazit\n\n")
        f.write("Die Analyse der Körperteilverletzungen zeigt bedeutende Muster im Polytrauma-Datensatz. ")

        # Most common body part
        if sorted_body_parts:
            top_body_part_name = sorted_body_parts[0][1]["name"]
            top_body_part_percentage = sorted_body_parts[0][1]["percentage"]
            f.write(
                f"Die häufigste Körperteilverletzung betrifft den {top_body_part_name} ({top_body_part_percentage:.1f}%). ")

            # If we have more body parts, mention the least common one
            if len(sorted_body_parts) > 1:
                least_body_part_name = sorted_body_parts[-1][1]["name"]
                least_body_part_percentage = sorted_body_parts[-1][1]["percentage"]
                f.write(
                    f"Die seltenste Körperteilverletzung betrifft den {least_body_part_name} ({least_body_part_percentage:.1f}%). ")

        f.write(
            "Diese Erkenntnisse deuten auf Bereiche hin, in denen Rehabilitationsressourcen basierend auf der Prävalenz priorisiert werden könnten.\n\n")

        f.write("*Dieser Bericht wurde automatisch im Rahmen des Polytrauma-Analyseprojekts erstellt.*")

    logger.info(f"Created body part summary report: {report_path}")
    return report_path

    """Create a German markdown summary report for body part injuries."""
    report_path = os.path.join(step3_output_folder, "koerperteil_verletzungen_bericht.md")

    with open(report_path, "w", encoding="utf-8") as f:
        # German headers and content
        f.write("# Analyse der Körperteilverletzungen\n\n")
        f.write(f"**Analysedatum:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        # Overview
        f.write("## Überblick\n\n")
        f.write(f"Diese Analyse untersucht die Verteilung von Körperteilverletzungen bei {results['total_cases']} ")
        f.write(
            "individuellen Patientenfällen. Sie gibt Einblicke in die Prävalenz verschiedener Verletzungstypen.\n\n")

        # Patient without injury (if any)
        if patients_without_injury:
            f.write("## Patient ohne Körperteilverletzung\n\n")
            f.write(
                f"Ein Patient (Schadennummer: {', '.join(patients_without_injury)}) weist keine Körperteilverletzung auf. ")
            f.write(
                "Dieser Fall könnte weitere Untersuchungen erfordern, um die spezifischen Umstände zu verstehen.\n\n")

        # Sort categories by percentage
        sorted_categories = sorted(
            results["categories"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )

        # Top 3 most common categories
        f.write("### Most Common Categories\n\n")
        f.write("| Rank | Category | Cases | Percentage |\n")
        f.write("|------|----------|-------|------------|\n")

        for i, (cat_name, cat_data) in enumerate(sorted_categories[:3]):
            f.write(f"| {i + 1} | {cat_name} | {cat_data['positive_cases']} | {cat_data['percentage']:.1f}% |\n")

        f.write("\n")

        # Least common categories (bottom 3)
        f.write("### Least Common Categories\n\n")
        f.write("| Rank | Category | Cases | Percentage |\n")
        f.write("|------|----------|-------|------------|\n")

        bottom_categories = sorted_categories[-3:]
        bottom_categories.reverse()  # Show least common first

        for i, (cat_name, cat_data) in enumerate(bottom_categories):
            f.write(
                f"| {len(sorted_categories) - i} | {cat_name} | {cat_data['positive_cases']} | {cat_data['percentage']:.1f}% |\n")

        f.write("\n")

        # Top subcategories
        f.write("### Top Subcategories by Category\n\n")

        for cat_name, cat_data in sorted_categories:
            f.write(f"#### {cat_name}\n\n")

            # Sort columns by percentage
            sorted_columns = sorted(
                cat_data["columns"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=True
            )

            # Show top 3 or all if less than 3
            top_columns = sorted_columns[:min(3, len(sorted_columns))]

            f.write("| Subcategory | Cases | Percentage |\n")
            f.write("|-------------|-------|------------|\n")

            for col_name, col_data in top_columns:
                f.write(f"| {col_data['name']} | {col_data['positive_cases']} | {col_data['percentage']:.1f}% |\n")

            f.write("\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("This analysis was conducted with the following methodology:\n\n")
        f.write("1. Each unique 'Schadennummer' represents a distinct patient case\n")
        f.write("2. A patient is positive for a category if any subcategory has 'Ja' in any visit\n")
        f.write(
            "3. For extremities, 'Arm links'/'Arm rechts' were merged into 'Arm' and 'Bein links'/'Bein rechts' into 'Bein'\n")
        f.write("4. Missing values were treated as 'Nein'\n")
        f.write("5. Percentages are calculated based on the total number of unique cases\n\n")

        # Resources
        f.write("## Resources\n\n")
        f.write(f"- Complete analysis results: [Excel File](injury_category_analysis.xlsx)\n")
        f.write("- Visualizations:\n")
        f.write("  - Category comparison chart\n")
        f.write("  - Horizontal category comparison\n")
        f.write("  - Category distribution pie chart\n")
        f.write("  - Category radar chart\n")
        f.write("  - Top subcategories heatmap\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The injury category analysis reveals significant patterns in the polytrauma dataset. ")

        # Add insights about top categories
        top_cat_name, top_cat_data = sorted_categories[0]
        second_cat_name, second_cat_data = sorted_categories[1]

        f.write(f"The most prevalent category is {top_cat_name} ({top_cat_data['percentage']:.1f}%), ")
        f.write(f"followed by {second_cat_name} ({second_cat_data['percentage']:.1f}%). ")

        # Add insights about least common category
        least_cat_name, least_cat_data = sorted_categories[-1]
        f.write(f"The least common category is {least_cat_name} ({least_cat_data['percentage']:.1f}%). ")

        f.write(
            "These findings suggest areas where rehabilitation resources might be prioritized based on prevalence.\n\n")

        f.write("*This report was automatically generated as part of the polytrauma analysis project.*")

    logger.info(f"Created summary report: {report_path}")
    return report_path


def injury_category_distribution():
    """Main function to perform the category-level injury analysis."""
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
    step3_plots_folder = os.path.join(graphs_folder, "step3", "injury_category_distribution")

    # Create necessary directories
    os.makedirs(step3_output_folder, exist_ok=True)
    os.makedirs(step3_log_folder, exist_ok=True)
    os.makedirs(step3_plots_folder, exist_ok=True)

    # Setup logging
    logger = setup_logging(step3_log_folder, "injury_category_distribution.log")
    logger.info("Starting injury category analysis...")

    try:
        # Load the dataset
        df = load_dataset(base_dataset, logger)

        # Create merged categories for extremities
        df, body_part_columns = create_merged_categories(df, logger)

        # CHANGE 1: Create separate category dictionaries
        body_part_categories = {
            "Körperteil": body_part_columns
        }

        # Define categories and their respective columns
        non_body_categories  = {
            "Somatisch": ["Somatisch-- Funktionsstoerung", "Somatisch-- Schmerz", "Somatisch--Komplikationen"],
            "Personenbezogen": [
                "Personenbezogen--Psychische Probleme/Compliance",
                "Personenbezogen--Entschädigungsbegehren",
                "Personenbezogen--Migrationshintergrund",
                "Personenbezogen--Suchtverhalten",
                "Personenbezogen--Zusätzliche Erkrankungen"
            ],
            "Taetigkeit": [
                "Taetigkeit--Arbeitsunfähig", "Tätigkeit--Wiedereingliederung", "Tätigkeit--Arbeitsfaehig",
                "Tätigkeit--BU/EU",
                "Altersrentner", "Ehrenamt", "Zuverdienst"
            ],
            "Umwelt": [
                "Umwelt--Beziehungsprobleme", "Umwelt--Soziale Isolation", "Umwelt--Mobilitaetsprobleme",
                "Umwelt--Wohnsituatuation", "Umwelt--Finazielle Probleme"
            ],
            "Med RM": [
                "Med RM--Arzt-Vorstellung", "Med RM--Arzt-Wechsel", "Med RM--Organisation ambulante Therapie",
                "Med RM--Organisation medizinische Reha", "Med RM--Wietere Krankenhausaufenthalte",
                "Med RM--Psychotherapie", "Organisation Pflege"
            ],
            "Soziales RM": [
                "Soziales RM--Lohnersatzleistungen", "Soziales RM--Arbeitslosenunterstuetzung",
                "Soziales RM--Antrag auf Sozialleistungen", "Soziales RM--Einleitung Begutachtung"
            ],
            "Technisches RM": [
                "Technisches RM--Hilfmittelversorgung", "Technisches RM--Mobilitätshilfe",
                "Technisches RM--Bauliche Anpassung", "Technisches RM--Arbetsplatzanpassung"
            ],
            "Berufliches RM": [
                "Berufliches RM--Leistungen zur Teilhabe am Arbeitsleben",
                "Berufliches RM--Integration/berufliche Neurientierung allgemeiner Arbeitsmarkt",
                "Berufliches RM--Wiedereingliederung geförderter Arbeitsmarkt", "Berufliches RM--BEM"
            ]
        }
        # Analyze separate categories
        logger.info("Analyzing body part injury categories...")
        body_part_results = analyze_categories(df, body_part_categories, logger)

        logger.info("Analyzing non-body injury categories...")
        non_body_results = analyze_categories(df, non_body_categories, logger)

        # Find patient with no body part injury
        body_part_flags = df.groupby("Schadennummer")[body_part_columns].apply(
            lambda g: (g == "Ja").any().any()
        )
        patients_without_injury = body_part_flags[~body_part_flags].index.tolist()
        if patients_without_injury:
            logger.info(f"Patient(s) with no body part injury: {patients_without_injury}")

        # Create separate folders
        body_part_plots_folder = os.path.join(step3_plots_folder, "body_part_injuries")
        os.makedirs(body_part_plots_folder, exist_ok=True)

        non_body_plots_folder = os.path.join(step3_plots_folder, "non_body_categories")
        os.makedirs(non_body_plots_folder, exist_ok=True)

        # Visualize results
        logger.info("Creating visualizations...")
        visualize_category_comparison(body_part_results, body_part_plots_folder, logger)
        create_heatmap(body_part_results, body_part_plots_folder, logger)

        visualize_category_comparison(non_body_results, non_body_plots_folder, logger)
        create_heatmap(non_body_results, non_body_plots_folder, logger)

        # Export results
        logger.info("Exporting analysis results...")
        body_excel_path, body_json_path = export_results(
            body_part_results, step3_output_folder, logger, filename_prefix="body_part_")

        non_body_excel_path, non_body_json_path = export_results(
            non_body_results, step3_output_folder, logger, filename_prefix="non_body_")

        # Create reports
        logger.info("Creating summary reports...")
        create_body_part_summary_report(
            body_part_results, body_excel_path, step3_output_folder, logger, patients_without_injury)

        create_non_body_summary_report(
            non_body_results, non_body_excel_path, step3_output_folder, logger)

        # [Success logging as before]

    except Exception as e:
        logger.error(f"Error in injury category analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    injury_category_distribution()
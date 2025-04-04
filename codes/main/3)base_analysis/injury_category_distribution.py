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


def export_results(results, step3_output_folder, logger):
    """Export the analysis results to Excel and JSON files."""
    # Export to Excel
    excel_path = os.path.join(step3_output_folder, "injury_category_analysis.xlsx")

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
    json_path = os.path.join(step3_output_folder, "injury_category_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Exported analysis results to JSON: {json_path}")

    return excel_path, json_path


def create_summary_report(results, excel_path, step3_output_folder, logger):
    """Create a markdown summary report of the analysis."""
    report_path = os.path.join(step3_output_folder, "injury_category_analysis_report.md")

    with open(report_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("# Injury Category Analysis Report\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        # Write overview
        f.write("## Overview\n\n")
        f.write(
            f"This analysis examines the distribution of injury categories across {results['total_cases']} unique patient cases. ")
        f.write(
            "It provides insights into the prevalence of different injury types and the relationships between them.\n\n")

        # Write key findings
        f.write("## Key Findings\n\n")

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

        # Define categories and their respective columns
        categories = {
            "Körperteil": body_part_columns,
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

        # Analyze categories
        logger.info("Analyzing main injury categories...")
        results = analyze_categories(df, categories, logger)

        # Visualize category comparison
        logger.info("Creating category comparison visualizations...")
        visualize_category_comparison(results, step3_plots_folder, logger)

        # Create heatmap of top subcategories
        logger.info("Creating heatmap of top subcategories...")
        create_heatmap(results, step3_plots_folder, logger)

        # Export results
        logger.info("Exporting analysis results...")
        excel_path, json_path = export_results(results, step3_output_folder, logger)

        # Create summary report
        logger.info("Creating summary report...")
        report_path = create_summary_report(results, excel_path, step3_output_folder, logger)

        logger.info("Injury category analysis completed successfully.")
        logger.info(f"Results saved to: {step3_output_folder}")
        logger.info(f"Visualizations saved to: {step3_plots_folder}")

    except Exception as e:
        logger.error(f"Error in injury category analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    injury_category_distribution()
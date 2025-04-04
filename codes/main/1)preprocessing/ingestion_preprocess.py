import os
import pandas as pd
from datetime import datetime
import numpy as np
import logging


def setup_logging(log_folder, log_name="processing.log"):
    """Set up logging with proper format"""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_name)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()


def preprocess_data(input_file_path, output_folder, log_folder, plots_folder):
    """
    Preprocess the polytrauma dataset with improved error handling and data validation
    """
    # Create step1 folders
    step1_output_folder = os.path.join(output_folder, "step1")
    step1_log_folder = os.path.join(log_folder, "step1")
    step1_plots_folder = os.path.join(plots_folder, "step1")

    os.makedirs(step1_output_folder, exist_ok=True)
    os.makedirs(step1_log_folder, exist_ok=True)
    os.makedirs(step1_plots_folder, exist_ok=True)

    logger = setup_logging(step1_log_folder, "processing.log")
    logger.info("Starting data preprocessing...")

    try:
        # Read the Excel file
        df = pd.read_excel(input_file_path, dtype={"Schadennummer": str})
        logger.info(f"Successfully loaded data from: {input_file_path}")
        logger.info(f"Initial data shape: {df.shape}")
        logger.info(f"Original column headers: {list(df.columns)}")

        # Clean column names (strip whitespace and standardize)
        df.columns = df.columns.str.strip()
        logger.info(f"Cleaned column headers: {list(df.columns)}")

        # Remove only the index column if it exists and is empty
        if 'index' in df.columns and df['index'].isnull().all():
            df = df.drop(columns=['index'])
            logger.info("Removed empty 'index' column")

        # Convert date columns to proper datetime
        date_columns = ['Unfalldatum', 'Besuchsdatum', 'Gebursdatum']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                invalid_dates = df[col].isnull().sum()
                if invalid_dates > 0:
                    logger.warning(f"{invalid_dates} invalid dates found in {col}")

        # Calculate age at accident time if birth date is available
        if 'Gebursdatum' in df.columns and 'Unfalldatum' in df.columns:
            # First make sure both date columns are properly converted
            df['Gebursdatum'] = pd.to_datetime(df['Gebursdatum'], errors='coerce', dayfirst=True)
            df['Unfalldatum'] = pd.to_datetime(df['Unfalldatum'], errors='coerce', dayfirst=True)

            # Calculate age only for rows where both dates are valid
            age_series = ((df['Unfalldatum'] - df['Gebursdatum']).dt.days / 365.25)

            # Convert to integer where possible, but handle NaN values
            df['Age_At_Accident'] = pd.Series([int(age) if pd.notna(age) else pd.NA for age in age_series])

            logger.info("Added Age_At_Accident column")
        else:
            logger.error("'Gebursdatum' or 'Unfalldatum' column is missing. Cannot calculate age.")

        # Calculate time intervals (0-3 months, 3-6 months, etc.)
        if 'Monat nach Unfall' in df.columns:
            df['Time_Interval'] = ((df['Monat nach Unfall'] // 3) + 1).astype('Int64')
            logger.info("Added Time_Interval column based on 'Monat nach Unfall'")
        else:
            logger.error("'Monat nach Unfall' column is missing. Cannot calculate time intervals.")

        # Calculate days between accident and visit for each record
        if 'Unfalldatum' in df.columns and 'Besuchsdatum' in df.columns:
            df['Days_Since_Accident'] = (df['Besuchsdatum'] - df['Unfalldatum']).dt.days
            logger.info("Added Days_Since_Accident column")

        # Save the processed DataFrame
        output_file_path = os.path.join(step1_output_folder, "Polytrauma_Analysis_Processed.xlsx")
        df.to_excel(output_file_path, index=False)

        logger.info(f"Processed data saved to: {output_file_path}")
        logger.info(f"Final data shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Load environment variables from .env file if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Using hardcoded paths.")

    # Get paths from environment or use defaults
    input_file_path = os.getenv("BASE_INPUT_EXCEL", "data/input/Polytrauma Analysis.xlsx")
    output_folder = os.getenv("OUTPUT_FOLDER", "data/output")
    log_folder = os.getenv("LOG_FOLDER", "logs")
    plots_folder = os.getenv("GRAPHS_FOLDER", "plots")

    preprocess_data(input_file_path, output_folder, log_folder, plots_folder)
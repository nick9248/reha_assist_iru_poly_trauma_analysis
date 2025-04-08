import subprocess
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv


def convert_md_to_pdf(md_file_path, output_pdf_path=None, project_root=None):
    """
    Convert a Markdown file to PDF using pandoc.

    Args:
        md_file_path (str): Path to the input Markdown file
        output_pdf_path (str, optional): Path for the output PDF file
        project_root (str, optional): Root directory of the project for image paths
    """
    # Check if input file exists
    if not os.path.exists(md_file_path):
        raise FileNotFoundError(f"Input file not found: {md_file_path}")

    # If output path not specified, create it from input path
    if output_pdf_path is None:
        output_pdf_path = str(Path(md_file_path).with_suffix('.pdf'))

    # If project_root not specified, use the directory containing the markdown file
    if project_root is None:
        project_root = os.path.dirname(md_file_path)

    # Create a temporary directory for the conversion
    temp_dir = os.path.join(os.path.dirname(md_file_path), "temp_pdf_conversion")
    os.makedirs(temp_dir, exist_ok=True)

    # Copy the markdown file to the temp directory
    temp_md_path = os.path.join(temp_dir, "temp.md")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix image paths in the content
    content = content.replace('/plots/', 'plots/')

    with open(temp_md_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Copy all plot directories to the temp directory
    plots_src = os.path.join(project_root, "plots")
    plots_dst = os.path.join(temp_dir, "plots")
    if os.path.exists(plots_src):
        shutil.copytree(plots_src, plots_dst, dirs_exist_ok=True)

    # Check if pandoc is installed
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Pandoc is not installed. Please install pandoc first.")

    # Create the pandoc command with a more widely available font
    cmd = [
        'pandoc',
        temp_md_path,
        '-o', output_pdf_path,
        '--pdf-engine=xelatex',
        '--template=default',
        '-V', 'geometry:margin=1in',
        '-V', 'mainfont:Arial',  # Changed to Arial which is more commonly available
        '-V', 'fontsize=12pt',
        '--toc',
        '--toc-depth=3',
        '--highlight-style=tango',
        '--resource-path=./'  # Add this to help pandoc find images
    ]

    try:
        # Run the pandoc command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=temp_dir)
        print(f"Successfully converted {md_file_path} to {output_pdf_path}")

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return output_pdf_path
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr}")
        # Clean up temporary directory even if there's an error
        shutil.rmtree(temp_dir)
        raise


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Set file paths using environment variables or adjust as needed
    project_root = os.getenv("PROJECT_ROOT")
    input_md = os.path.join(project_root, "documentation/polytrauma-report.md")
    output_pdf = os.path.join(project_root, "documentation/polytrauma_analysis_report.pdf")

    try:
        convert_md_to_pdf(input_md, output_pdf, project_root)
    except Exception as e:
        print(f"Error: {e}")
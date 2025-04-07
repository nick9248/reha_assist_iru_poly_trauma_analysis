import subprocess


def convert_md_to_pdf(input_file, output_file):
    """
    Convert a Markdown file to a PDF using Pandoc with specified options.

    Args:
        input_file (str): Path to the input Markdown (.md) file.
        output_file (str): Path where the output PDF should be saved.
    """
    # Command options:
    # --pdf-engine=xelatex: Uses XeLaTeX for better Unicode support.
    # -V geometry:margin=1in: Sets 1-inch margins.
    # -V papersize:a4: Sets the paper size to A4.
    command = [
        "pandoc",
        input_file,
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "papersize:a4",
        "-o", output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful! PDF saved as {output_file}")
    except subprocess.CalledProcessError as error:
        print("Error during conversion:", error)


if __name__ == '__main__':
    # Replace these with your actual file paths.
    input_md = r"C:/Users/Nick\PycharmProjects\polytrauma_analysis\documentation\polytrauma-report.md"
    output_pdf = r"C:/Users/Nick\PycharmProjects\polytrauma_analysis\documentation\polytrauma-report.pdf"
    convert_md_to_pdf(input_md, output_pdf)

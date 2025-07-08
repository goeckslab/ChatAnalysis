import json
import argparse
import pandas as pd
import logging
from weasyprint import HTML
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import os

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

# Function to generate PDF content from JSON chat history
def generate_html_from_json(json_file, output_pdf):
    # Load chat history from JSON
    with open(json_file, "r") as f:
        chat_history = json.load(f)
    
    css_styles = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            padding: 20px;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        pre {
            background-color: #282c34;
            color: white;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap; /* Wrap long lines */
        }
        img {
            max-width: 100%;
            display: block;
            margin: 20px auto;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
    """
    
    html_content = f"<html><head>{css_styles}</head><body>"
    for idx, message in enumerate(chat_history):
        role = message["role"].capitalize()
        content = message.get("content", "")
        html_content += f"<div class='container'><h2>{role}</h2><p>{content}</p>"

        if "code_excuted" in message and message["code_excuted"] is not None:
            highlighted_code = highlight(
                message['code_excuted'], PythonLexer(), HtmlFormatter(style="colorful", linenos=True)
            )
            html_content += f"<h3>Executed Code:</h3><pre>{highlighted_code}</pre>"

        if "image" in message:
            # Directly use the Base64 string to embed the image
            image_base64 = message["image"]
            html_content += f'<img src="data:image/png;base64,{image_base64}" alt="Generated Plot" />'

        if "content_df" in message:
            # Save the DataFrame as a CSV file
            df = pd.DataFrame(message["content_df"])
            csv_file = f"table_{idx}.csv"
            df.to_csv(csv_file, index=False)
            # Add download link for the CSV
            html_content += f'<p><a href="{csv_file}" download>The answer is a data table. Download Table as CSV</a></p>'
        
        html_content += "</div>"

    html_content += "</body></html>"

    # Convert HTML to PDF
    HTML(string=html_content, base_url=os.getcwd()).write_pdf(output_pdf)
    print(f"PDF successfully created at: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", help="Path to the JSON file containing the chat history", required=True)
    parser.add_argument("--output-pdf", help="Path to the output PDF file", required=True)
    args = parser.parse_args()
    
    generate_html_from_json(args.json_file, args.output_pdf)

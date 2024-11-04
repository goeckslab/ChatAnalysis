import json
import argparse
import pandas as pd
import logging  

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

# Function to generate a table from a DataFrame
def dict_to_html_table(data_dict, table_id):
    # Generate an enhanced HTML table with DataTables JavaScript library for better interactivity
    html_content = f"""
    <div class='container'>
        <h2>Data Table</h2>
        <div class='table-responsive'>
            <table id='{table_id}' class='display nowrap' style='width:100%'>
                <thead>
                    <tr>
    """
    
    # Add headers dynamically based on keys
    headers = data_dict.keys()
    for header in headers:
        html_content += f"<th>{header}</th>"
    
    html_content += """
                    </tr>
                </thead>
                <tbody>
    """

    # Transpose the dictionary to create rows
    num_rows = len(next(iter(data_dict.values())))
    for i in range(num_rows):
        html_content += "<tr>"
        for key in data_dict.keys():
            value = data_dict[key][i]
            html_content += f"<td>{value}</td>"
        html_content += "</tr>"
    
    html_content += f"""
                </tbody>
            </table>
        </div>
    </div>
    <script>
        $(document).ready(function() {{
            // Initialize the DataTable with the desired settings
            $('#{table_id}').DataTable({{
                "paging": true,
                "searching": true,
                "ordering": true,
                "info": true,
                "scrollX": true,
                "lengthMenu": [10, 25, 50, 100] // Options for rows per page selection
            }});
        }});
    </script>

    """
    
    return html_content

# Function to generate HTML from JSON chat history and DataFrame
def generate_html_from_json(json_file, output_html):
    # Load chat history from JSON
    with open(json_file, "r") as f:
        chat_history = json.load(f)
    
    css_styles = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            max-width: 100%;
            width: 95%;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .table-responsive {
            overflow-x: auto;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        .message-container {
            margin-bottom: 15px;
            border-radius: 10px;
            padding: 10px;
            max-width: 100%;
            word-wrap: break-word;
            background-color: #f0f0f5;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #d4eaff;
        }
        .assistant-message {
            background-color: #e5e5ea;
        }
        .message-container p {
            margin: 0;
        }
        .message-container pre {
            background-color: #282c34;
            color: white;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        img {
            max-width: 100%;
            border-radius: 8px;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .code-block {
            background-color: #282c34;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            overflow-x: auto;
        }
        .foldable {
            cursor: pointer;
            margin-bottom: 10px;
            padding: 12px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
            display: inline-block;
            text-align: center;
        }

        .foldable:hover {
            background-color: #0056b3;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .foldable:active {
            background-color: #004494;
            transform: translateY(2px); /* Adds a pressed effect when clicked */
        }
        .fold-content {
            display: none;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            max-width: 95%; /* Ensures it does not exceed the container width */
            overflow-x: auto; /* Adds horizontal scroll if necessary */
            word-wrap: break-word; /* Breaks long words */
            white-space: pre-wrap; /* Preserves formatting while wrapping long lines */
            
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            text-align: left;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        </style>
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.css">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.js"></script>
    <script>
        function toggleFoldable(id) {
            var content = document.getElementById(id);
            if (content.style.display === "none" || content.style.display === "") {
                content.style.display = "block";
            } else {
                content.style.display = "none";
            }
        }
    </script>
    """

    # Generate HTML content
    html_content = f"<html><head>{css_styles}</head><body><div class='container'><h2>Chat History</h2><div class='chat-container'>"
    
    for idx, message in enumerate(chat_history):
        role = message["role"].capitalize()
        content = message.get("content", "")
        
        # Both user and assistant messages are aligned to the left
        if message["role"] == "user":
            html_content += f"<div class='message-container user-message'><p><strong>{role}:</strong> {content}</p></div>"
        else:
            html_content += f"<div class='message-container assistant-message'><p><strong>{role}:</strong> {content}</p></div>"

            if "code_excuted" in message:
                html_content += f"<div class='foldable' onclick='toggleFoldable(\"fold-exec-{idx}\")'><strong>Show/Hide Executed Code</strong></div>"
                html_content += f"<div id='fold-exec-{idx}' class='fold-content'><pre>{message['code_excuted']}</pre></div>"

            # If an image is included
            if "image" in message:
                html_content += f'<img src="data:image/png;base64,{message["image"]}" alt="Image"/><br>'
            
            # If code was generated
            if "code_generated" in message:
                html_content += f"<div class='code-block'><pre>{message['code_generated']}</pre></div>"

            # If there's a DataFrame in the message (under 'content_df')
            if "content_df" in message:
                # Convert the DataFrame from JSON back to a pandas DataFrame
                df = pd.DataFrame(message["content_df"])
                # Convert the DataFrame to a dictionary and generate an enhanced HTML table
                data_dict = df.to_dict(orient='list')
                table_id = f"data-table-{idx}"
                try:
                    html_content += dict_to_html_table(data_dict, table_id)
                except Exception as e:
                    LOG.error(f"Error generating table: {e}")

    html_content += "</div></div></body></html>"

    # Write the HTML content to the output file
    with open(output_html, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", help="Path to the JSON file containing the chat history")
    parser.add_argument("--output-html", help="Path to the output HTML file")
    args = parser.parse_args()
    
    generate_html_from_json(args.json_file, args.output_html)

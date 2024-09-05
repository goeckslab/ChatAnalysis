import json
import argparse
import pandas as pd

# Function to generate a table from a DataFrame (or dict)
def dict_to_html_table(data_dict):
    # Generate an HTML table with modern CSS
    html_content = """
    <div class='container'>
        <h2>Data Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Key</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Iterate over the dictionary to populate rows
    for key, value in data_dict.items():
        html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
    
    html_content += """
            </tbody>
        </table>
    </div>
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
            max-width: 800px;
            width: 100%;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
    """

    # Generate HTML content
    html_content = f"<html><head>{css_styles}</head><body><div class='container'><h2>Chat History</h2><div class='chat-container'>"
    
    for message in chat_history:
        role = message["role"].capitalize()
        content = message.get("content", "")
        
        # Both user and assistant messages are aligned to the left
        if message["role"] == "user":
            html_content += f"<div class='message-container user-message'><p><strong>{role}:</strong> {content}</p></div>"
        else:
            html_content += f"<div class='message-container assistant-message'><p><strong>{role}:</strong> {content}</p></div>"

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
            # Convert the DataFrame to a dictionary and generate an HTML table
            data_dict = df.to_dict(orient='list')
            html_content += dict_to_html_table(data_dict)

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

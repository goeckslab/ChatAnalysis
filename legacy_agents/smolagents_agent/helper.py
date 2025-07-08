import re

def detect_image_path(context):
    # Simplified regex pattern to match any path ending with an image extension
    pattern = r"(/[^ ]+\.(?:png|jpg|jpeg))"
    
    match = re.search(pattern, context)
    
    if match:
        # Extract the file path from the matched string
        image_path = match.group(0)
        print(f"Image path detected: {image_path}")
        return image_path
    else:
        return None


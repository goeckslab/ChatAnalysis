# pages/1_Bookmarks.py
import streamlit as st
import os
from PIL import Image
import json


st.set_page_config(
    page_title="Bookmarks - Galaxy Chat Analysis",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'favicon.ico'),
    layout="wide",
)


st.markdown("""
<style>
    /* Target the specific section you identified (e.g., the second one) */
    div[data-testid="stAppViewContainer"] > section:nth-of-type(2) {
        max-width: 60% !important;   /* << ADJUST THIS VALUE to your desired width */
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2.5rem;           /* Optional: Adjust side padding */
        padding-right: 2.5rem;          /* Optional: Adjust side padding */
        /* You can add a light border here too if you want to see its final bounds, e.g.: */
        /* border: 1px solid lightgrey !important; */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* General style for all navigation links to ensure consistency */
    [data-testid="stSidebarNav"] ul li a {
        display: block; /* Makes the entire area clickable and styleable */
        padding: 0.5rem 0.75rem; /* Adjust padding: top/bottom left/right */
        margin-bottom: 0.2rem; /* Space between items */
        border-radius: 0.375rem; /* Rounded corners */
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out, border-left 0.2s ease-in-out;
        color: #4A5568; /* Default text color for non-active items (a grayish tone) */
        text-decoration: none; /* Remove underline from links */
        border-left: 4px solid transparent; /* Placeholder for active border */
    }

    /* Style for the CURRENTLY ACTIVE navigation link */
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] {
        background-color: #4299E1 !important; /* A brighter, more prominent blue */
        color: white !important;             /* White text for contrast */
        font-weight: 700 !important;         /* Even bolder */
        border-left: 4px solid #2B6CB0 !important; /* Prominent left border (darker blue) */
    }

    /* Hover effect for NON-ACTIVE links */
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]):hover {
        background-color: #E2E8F0; /* Light gray background on hover */
        color: #2B6CB0 !important; /* Darker blue text on hover */
        border-left: 4px solid #A0AEC0; /* Subtle border on hover for non-active */
    }

    /* Optional: Slightly dim non-active links if you want extreme focus on active */
    /*
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]) {
        opacity: 0.75;
    }
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]):hover {
        opacity: 1;
    }
    */
</style>
""", unsafe_allow_html=True)

st.title("🔖 Bookmark Manager")

def load_chat_history():
    if os.path.exists("bookmarks.json"):
        with open("bookmarks.json", "r") as f:
            file_contents = f.read().strip()
            if file_contents:
                history = json.loads(file_contents)
                st.session_state["bookmarks"] = history.get("bookmarks", [])
            else:
                st.session_state["bookmarks"] = []


load_chat_history()
bookmarks = st.session_state.get("bookmarks", [])

if not bookmarks:
    st.info("No bookmarks have been saved yet. You can save chat responses from the main Chat Analysis page.")
else:
    st.markdown(f"You have **{len(bookmarks)}** bookmark(s).")
    for i, b_data in enumerate(bookmarks):
        if not isinstance(b_data, dict): # Basic check for valid bookmark structure
            st.warning(f"Skipping invalid bookmark item at index {i}.")
            continue

        question = b_data.get("question", "Unknown question")
        answer = b_data.get("answer", "No answer saved")
        plot_paths = b_data.get("plots", [])
        file_paths = b_data.get("files", [])

        with st.expander(f"Bookmark {i + 1}: {question[:60]}"):
            st.markdown(f"**❓ Question:**\n{question}\n")
            st.markdown(f"**💡 Answer:**\n{answer}")

            if plot_paths and not (len(plot_paths) == 1 and plot_paths[0] == ""):
                st.markdown("**📊 Saved Plots:**")
                for plot_path_in_bookmark in plot_paths:
                    actual_plot_path = plot_path_in_bookmark
                    if actual_plot_path == "":
                        continue

                    if actual_plot_path and os.path.exists(actual_plot_path):
                        try:
                            image = Image.open(actual_plot_path)
                            st.image(image, caption=os.path.basename(actual_plot_path))
                        except Exception as e:
                            st.error(f"Could not load plot {os.path.basename(actual_plot_path)}: {e}")
                    else:
                        pass

            if file_paths and not (len(file_paths) == 1 and file_paths[0] == ""):
                st.markdown("**📄 Saved Files:**")
                for file_path_in_bookmark in file_paths:
                    actual_file_path = file_path_in_bookmark
                    if actual_file_path == "":
                        continue

                    if actual_file_path and os.path.exists(actual_file_path):
                        try:
                            with open(actual_file_path, "rb") as f_download:
                                st.download_button(
                                    label=f"Download {os.path.basename(actual_file_path)}",
                                    data=f_download,
                                    file_name=os.path.basename(actual_file_path),
                                    key=f"bm_dl_{i}_{os.path.basename(actual_file_path)}"
                                )
                        except Exception as e:
                            st.error(f"Could not prepare file {os.path.basename(actual_file_path)} for download: {e}")
                    else:
                        pass

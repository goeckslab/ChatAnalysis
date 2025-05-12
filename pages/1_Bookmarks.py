# pages/1_Bookmarks.py
import streamlit as st
import os
from PIL import Image # If displaying images from paths


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

# Ensure necessary session state variables are accessible
# These should have been set by Chat_Bot.py when it first ran.
output_dir = st.session_state.get("generate_file_path", "outputs_smolagents") # Default if not found

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = [] # Initialize if somehow not present

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
            st.markdown(f"**❓ Question:**\n```\n{question}\n```")
            st.markdown(f"**💡 Answer:**\n{answer}") # Assuming answer is markdown-compatible

            if plot_paths:
                st.markdown("**📊 Saved Plots:**")
                for plot_path_in_bookmark in plot_paths:
                    # Construct full path if paths are stored relative or just basenames
                    # Assuming paths in bookmark_data are already correct relative to execution
                    # or are absolute. If relative to output_dir, prepend it.
                    # For simplicity, let's assume plot_path_in_bookmark is usable as is
                    # or is a full path. If it's just a basename:
                    # actual_plot_path = os.path.join(output_dir, os.path.basename(plot_path_in_bookmark))
                    actual_plot_path = plot_path_in_bookmark # Use this if paths are stored fully qualified or correctly relative

                    if os.path.exists(actual_plot_path):
                        try:
                            image = Image.open(actual_plot_path)
                            st.image(image, caption=os.path.basename(actual_plot_path))
                        except Exception as e:
                            st.error(f"Could not load plot {os.path.basename(actual_plot_path)}: {e}")
                    else:
                        st.warning(f"Plot not found: {actual_plot_path}")

            if file_paths:
                st.markdown("**📄 Saved Files:**")
                for file_path_in_bookmark in file_paths:
                    # actual_file_path = os.path.join(output_dir, os.path.basename(file_path_in_bookmark))
                    actual_file_path = file_path_in_bookmark # Similar assumption as plots

                    if os.path.exists(actual_file_path):
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
                        st.warning(f"File not found: {actual_file_path}")

            # Add delete/rerun functionality if desired (would need to modify st.session_state.bookmarks and save)
            # e.g., if st.button("Delete Bookmark", key=f"delete_bm_{i}"):
            #   st.session_state.bookmarks.pop(i)
            #   # Need a way to trigger save_chat_history() from StreamlitApp if it's responsible,
            #   # or manage bookmark saving directly via session state + json persistence here.
            #   # For now, keep it simple.
            #   st.experimental_rerun()

# If you have common sidebar elements (like API config) that should appear on all pages,
# you might need to duplicate that logic here or move it to a shared utility function.
# For now, the Bookmarks page is simple and doesn't re-declare the LLM config sidebar.
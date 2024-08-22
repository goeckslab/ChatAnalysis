FROM python:3.9-slim

WORKDIR /ChatAnalysis

COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501

# # Define the entrypoint to run the Streamlit app
# ENTRYPOINT ["streamlit", "run", "chat_analysis.py"]


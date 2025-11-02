# ---------- Stage 1: Base Python Environment ----------
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first for dependency caching
COPY req.txt .

# Install dependencies
RUN pip install --no-cache-dir -r req.txt

# Copy the rest of your project files
COPY . .
# After COPY . .
COPY .env .env


# Copy environment variables (optional, you can also mount .env separately)
# ENV variables will be loaded automatically if you use python-dotenv or streamlit secrets
# Make sure not to expose sensitive keys when pushing to public repos

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app (or switch to app.py if that’s your main entry)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

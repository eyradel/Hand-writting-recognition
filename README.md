# Redox OCR Translator

This project is an Optical Character Recognition (OCR) and translation tool built with Streamlit. It allows users to upload images or PDF files, performs text detection using Google Cloud Vision API, and translates the detected text from German to English using OpenAI's language model.

## Features

- Upload images or PDF files for text extraction.
- Detect and extract text from uploaded files using Google Cloud Vision API.
- Translate the extracted text from German to English using OpenAI's language model.
- Display results with detection time and confidence level metrics.

## Prerequisites

- Python 3.7 or higher
- [Google Cloud Vision API](https://cloud.google.com/vision) API key
- [OpenAI API](https://openai.com/) key

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/eyradel/LLM-OCR-Translator.git
    cd LLM-OCR-Translator
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project directory and add your API keys:
    ```plaintext
    GOOGLE_API_KEY=your_google_cloud_vision_api_key
    OPENAI_KEY=your_openai_api_key
    ```

5. Ensure your `.gitignore` file includes the following lines to prevent sensitive information from being pushed to your repository:
    ```plaintext
    .env
    credentials.json
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open the provided URL in your web browser (usually http://localhost:8501).

3. Use the application to upload an image or PDF file and obtain the translated text results.

## Project Structure

- `app.py`: The main application file containing the Streamlit app code.
- `.env`: File to store your API keys (not included in the repository).
- `requirements.txt`: List of dependencies required for the project.
- `.gitignore`: Git ignore file to exclude sensitive files and directories from version control.

```bash

docker run -p 8501:8501 --env-file .env -v $(pwd)/credentials.json:/app/credentials.json streamlit-ocr-translator

```

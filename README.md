# Chat with LLM || PDF file

This project implements a ChatGPT-like text assistant that can handle both general chat queries and questions related to the contents of an uploaded PDF file. It uses Streamlit for the user interface and integrates the `simple_chat` and `RAG` classes for processing user inputs.

## Features

- **General Chat**: Allows users to interact with the assistant like a standard chat interface.
- **PDF Query**: Users can upload a PDF file and ask questions about its content.
- **Seamless Experience**: The system automatically handles queries based on user input and file uploads.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or above
- Streamlit
- dotenv
- fitz (PyMuPDF)
- langchain
- sentence-transformers
- faiss-cpu
- Any required AI model libraries (like Groq)

You can install these dependencies using `pip`:

```bash
pip install streamlit python-dotenv pymupdf langchain sentence-transformers faiss-cpu
```

## Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com//chatgpt-clone-with-pdf.git
cd chatgpt-clone-with-pdf
```

2. **Set up environment variables**:

Create a `.env` file in the root directory and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here

```

3. **Run the Streamlit application**:

```bash
streamlit run app.py
```

## Files

- **`app.py`**: The main Streamlit application file.
- **`class_for_RAG.py`**: Contains the `simple_chat` and `RAG` classes used for processing user inputs and PDF queries.

## Usage

- **Chat Interaction**: Simply enter your text in the input box and press enter. The assistant will respond to your queries.
- **PDF Queries**: Upload a PDF file using the "Upload a PDF file" button, then enter your question about the PDF content in the input box.

The assistant will handle the rest!

## Contributing

Feel free to open issues or submit pull requests for enhancements or bug fixes.

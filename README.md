# Local LLM Document QA System

This repository contains a local implementation of a Retrieval-Augmented Generation (RAG) system for answering questions based on user-uploaded PDF files. The system uses Langchain, Llama models, and Gradio to create a simple Q&A application that includes PDF browsing functionality.

## Features

- **PDF Processing:** Upload PDF documents, convert them into readable chunks, and extract key content.
- **Custom Embeddings:** Uses a custom embedding model based on a BERT-tiny transformer to create vector representations of PDF text.
- **Model Selection:** Choose between different Llama models for question-answering (e.g., Llama-3.2-1B and Meta-Llama-3.1-8B).
- **Gradio Interface:** A user-friendly interface for uploading PDFs, browsing pages, and querying the content.
- **Offline Capability:** Downloads models locally, reducing dependence on external services.
- **Interactive PDF Navigation:** Browse through pages of the PDF to understand the context better.

## Installation

To use this system, you need Python 3.8 or above. Follow these steps to get started:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Vicchan2324/Local_LLM_RAG.git
   cd https://github.com/Vicchan2324/Local_LLM_RAG.git
   ```

2. **Install Dependencies:**
   Use the package manager `pip` to install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. **Create Necessary Directories:**
   Ensure that model and data directories exist:
   ```sh
   mkdir -p models data/vectorstore
   ```

## Usage

1. **Launch the Application:**
   Run the script to launch the Gradio interface:
   ```sh
   python local_llm_with_pdf_reader.py
   ```
   The Gradio app will be available at a local server address, which can be shared if desired.
   Please note that the first time you run the app, it may take longer than usual as the model is downloaded to your local machine. Subsequent runs will be faster.
2. **Upload a PDF:**
   - Use the file upload button to select a PDF.
   - The system will process the document, split it into chunks, and create embeddings.

3. **Initialize the Model:**
   - Select the desired Llama model from the dropdown menu and click "Initialize Model".
   - Models are downloaded automatically if they do not exist locally.

4. **Ask Questions:**
   - Type in a question related to the document content.
   - The system will display the answer along with the corresponding page of the document.

5. **Navigate the PDF:**
   - Use the "Next Page" and "Previous Page" buttons to navigate through the document visually.

## File Structure

- `main.py` : Main script to run the application.
- `models/` : Stores downloaded Llama models.
- `data/vectorstore/` : Stores vector representations of PDF texts for efficient retrieval.

## Dependencies

- **Langchain**: For chain management and vector stores.
- **Transformers**: For custom embeddings using BERT-tiny.
- **Gradio**: User interface for the web application.
- **PyPDFLoader**: To process and extract content from PDFs.
- **Chroma**: To create a vector database for efficient retrieval.
- **LLamaCpp**: A Python binding for efficient inference using Llama models.

## Notes

- **Model Download**: The models are downloaded automatically from Hugging Face if not available locally.
- **Device Compatibility**: The script automatically uses Apple M1/M2 GPU ("mps") if available; otherwise, it falls back to the CPU.
- **Data Persistence**: Vector stores are saved locally to speed up subsequent queries.

## Future Improvements

- **Support More File Types**: Extend document processing capabilities beyond PDF.
- **Improve Embeddings Model**: Switch to a larger transformer model for better answer quality.
- **Distributed Setup**: Enable deployment across multiple nodes to handle larger document sets.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For any inquiries or issues, please contact the author at [your-email@example.com].


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import gradio as gr
import os
import requests
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import time
from pdf2image import convert_from_path
import hashlib
import json 
import pickle

class SimpleEmbeddings:
    def __init__(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny").to(device)
        self.device = device
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class RAGSystem:
    def __init__(self):
        self.qa_chain = None
        self.embeddings = SimpleEmbeddings()
        self.model_path = None
        self.pdf_images = []
        self.current_page = 0

        
    def download_model(self, model_path: str):
        if not os.path.exists(model_path):
            print("Downloading model... This might take a while.")
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf"
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
    
    def init_llm(self, model_selection):
        if model_selection == "Llama-3.2-1B":
            model_path = "./models/Llama-3.2-1B-Instruct-f16.gguf"
        elif model_selection == "Meta-Llama-3.1-8B":
            model_path = "./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
        else:
            raise ValueError("Invalid model selection")
        
        self.download_model(model_path)
        self.model_path = model_path
        
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_tokens=2000,
            n_ctx=1024,
            n_batch=128,
            f16_kv=True,
            verbose=True,
            streaming=True
        )
    
    def process_pdf(self, file_path):
        try:
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            # Convert PDF pages to images
            self.pdf_images = convert_from_path(file_path, fmt="png")
            return texts
        except Exception as e:
            raise gr.Error(f"Error processing PDF: {str(e)}")
    
    def create_vectorstore(self, texts):
        try:
            vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./data/vectorstore"
            )
            return vectorstore
        except Exception as e:
            raise gr.Error(f"Error creating vector store: {str(e)}")
    
    def upload_pdf(self, file):
        try:
            if not file.name.endswith('.pdf'):
                raise gr.Error("Please upload a PDF file")
            
            
            print("Creating new vector store")
            texts = self.process_pdf(file.name)
            vectorstore = self.create_vectorstore(texts)
            pdf_images = self.pdf_images

            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            self.current_page = 0  # Reset to the first page
            return f"PDF processed successfully. Created {len(texts)} chunks.", self.pdf_images[0]
        except Exception as e:
            raise gr.Error(str(e))
    def calculate_pdf_hash(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_hash = hashlib.md5(file.read()).hexdigest()
        return pdf_hash

    def query_streaming(self, question):
        if self.qa_chain is None:
            raise gr.Error("Please upload a PDF first")

        try:
            # Create a prompt based on the PDF file
            pdf_prompt = f"You are an expert on the document '{self.pdf_images[0].filename}'. Please answer the following question based on the file provided: {question}"

            # Query the model with the prompt
            response = self.qa_chain(pdf_prompt)
            answer = response['result']
            sources = response.get('source_documents', [])

            # Check if the answer is acceptable
            acceptable_prompt = f"Is the following answer acceptable for the question '{question}': {answer}? Answer the question in a single word: Yes or No. "
            acceptable_response = self.qa_chain(acceptable_prompt)
            acceptable_answer = acceptable_response['result']
            print('='*25)
            print('Checker output: ', acceptable_answer)
            # If the answer is not acceptable, display a message instead
            if "yes" not in acceptable_answer.lower():
                return "Sorry, I couldn't find a suitable answer to your question.", self.pdf_images[0]

            # If the answer is acceptable, display the answer and the page where the answer is based on
            page_number = None
            for source in sources:
                if source.page_content in answer:
                    page_number = source.page_number
                    break

            if page_number is not None:
                self.current_page = page_number - 1  # Subtract 1 because page numbers are 1-indexed
                return answer, self.pdf_images[self.current_page]
            else:
                return answer, self.pdf_images[0]

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def get_page(self, direction):
        if direction == "next" and self.current_page < len(self.pdf_images) - 1:
            self.current_page += 1
        elif direction == "prev" and self.current_page > 0:
            self.current_page -= 1
        return self.pdf_images[self.current_page] if self.pdf_images else None

# Initialize the RAG system
rag_system = RAGSystem()

# Create the Gradio interface
with gr.Blocks(title="Document Q&A System with PDF Viewer") as demo:
    gr.Markdown("""
    # Document Q&A System
    Upload a PDF document, browse through it, and ask questions about its content.
    """)

    # Define layout with two columns
    with gr.Row():
        # Left column for PDF viewer
        with gr.Column(scale=1):
            pdf_viewer = gr.Image(label="PDF Viewer", type="numpy", width=400, height=600)  # Fixed size for PDF viewer
            next_button = gr.Button("Next Page")
            prev_button = gr.Button("Previous Page")
        
        # Right column for controls and question input
        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=["Llama-3.2-1B", "Meta-Llama-3.1-8B"],
                value="Llama-3.2-1B",
                interactive=True
            )
            init_button = gr.Button("Initialize Model")
            init_output = gr.Textbox(label="Model Initialization Status")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_button = gr.Button("Process PDF")
            upload_output = gr.Textbox(label="Upload Status")
            
            question_input = gr.Textbox(label="Your Question", placeholder="Ask a question about the document...", lines=2)
            query_button = gr.Button("Ask Question")
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=10)

    # Event handlers
    init_button.click(fn=lambda model_selection: rag_system.init_llm(model_selection), inputs=[model_selector], outputs=[init_output])
    upload_button.click(fn=rag_system.upload_pdf, inputs=[file_input], outputs=[upload_output, pdf_viewer])
    query_button.click(fn=rag_system.query_streaming, inputs=[question_input], outputs=[answer_output, pdf_viewer])    
    next_button.click(fn=lambda: rag_system.get_page("next"), outputs=[pdf_viewer])
    prev_button.click(fn=lambda: rag_system.get_page("prev"), outputs=[pdf_viewer])

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./data/vectorstore", exist_ok=True)
    demo.launch(share=True)

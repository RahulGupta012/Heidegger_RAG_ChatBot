# Philosophy RAG Chatbot

A Retrieval-Augmented Generation (RAG) application built to provide highly accurate and context-aware conversations based on a curated knowledge base of a philosopher known for his critical views on technology and modern technological advancement.

## Features

* Automated data collection through document loading and web scraping
* Advanced text preprocessing and chunking pipeline
* Semantic embedding generation for contextual understanding
* Vector database integration for efficient similarity search
* Retrieval-Augmented Generation (RAG) architecture
* Prompt-engineered response generation to improve factual accuracy and consistency
* Interactive chatbot-based graphical user interface
* Low-hallucination response design through grounded retrieval


## 🚀 Built With

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python">
  <img src="https://img.shields.io/badge/Cohere-LLM-orange">
  <img src="https://img.shields.io/badge/LangChain-Orchestration-green">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Database-red">
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-purple">
  <img src="https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-yellow">
  <img src="https://img.shields.io/badge/Pandas-Analytics-150458?logo=pandas">
  <img src="https://img.shields.io/badge/NumPy-Computing-013243?logo=numpy">
  <img src="https://img.shields.io/badge/Tkinter-Desktop%20UI-blue">
</p>



## Architecture

Data Collection → Preprocessing → Chunking → Embedding Generation → Vector Storage → Semantic Retrieval → Cohere LLM → Response Generation

## Getting Started

### 1. Clone the Repository

```bash
git clone <https://github.com/RahulGupta012/Heidegger_RAG_ChatBot/>
cd <Heidegger_RAG_ChatBot>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

This project uses Cohere as the language model provider.

Create a `.env` file in the project root directory and add your API key:

```env
API_KEY=your_cohere_api_key
```

You can obtain a free API key from the Cohere developer platform.

### 4. Launch the Application

Run the chatbot interface using:

```bash
python frontend.py
```

## Project Goal

The primary objective of this project is to demonstrate how Retrieval-Augmented Generation can be used to build domain-specific AI assistants that produce grounded, context-aware responses while minimizing hallucinations through retrieval-based reasoning and carefully engineered prompts.

## Notes

* A valid Cohere API key is required to run the application.
* All dependencies are listed in `requirements.txt`.
* The quality of responses depends on the provided knowledge base and retrieved context.



# 📚 AI-Powered Library Semantic Search

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla-F7DF1E.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC.svg)

A full-stack web application that modernizes library catalog navigation using local Machine Learning. Instead of relying on exact keyword matches, this application uses **Semantic Search** to understand the meaning, concept, or mood behind a user's query (e.g., *"how stars are formed"* or *"I'm stressed and want something inspiring"*).

## ✨ Features

* **Semantic Vector Search:** Powered by local, API-key-free sentence embeddings (`all-MiniLM-L6-v2`), keeping the application fast and cost-effective.
* **Vector Database Integration:** Uses ChromaDB for in-memory cosine similarity queries combined with metadata filtering (e.g., filtering by library floor or availability).
* **Real-time Inventory Management:** Simulates checking out and returning books, dynamically updating stock levels in both the UI and the backend.
* **Modern Frontend:** A lightweight, responsive, single-page application built with Vanilla JS and Tailwind CSS, featuring dynamic concept-match progress bars.

## 🛠️ Architecture & Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI / ML:** Sentence-Transformers (Hugging Face)
* **Vector Store:** ChromaDB
* **Frontend:** HTML5, JavaScript, Tailwind CSS

## 🚀 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone [https://github.com/Novaz-Edd/library_books_vector_search.git](https://github.com/Novaz-Edd/library_books_vector_search.git)
cd library_books_vector_search
2. Install backend dependencies
Ensure you have Python installed, then set up your environment:

Bash
pip install fastapi uvicorn chromadb sentence-transformers
3. Start the FastAPI server
Navigate to the backend directory and spin up the ASGI server. Note: The first run will take a few moments to download the local ML weights from Hugging Face.

Bash
cd backend
uvicorn backend:app --reload --port 8000
4. Launch the UI
Open index.html (located in the frontend folder or root directory) directly in your web browser. No frontend build step or package manager is required!

🧠 How It Works Under the Hood
Database Seeding: On startup, the backend automatically embeds a mock dataset of books into ChromaDB.

Querying: When a user types a search, FastAPI converts the text into a vector embedding.

Retrieval: ChromaDB compares the query vector against the book descriptions, calculating a similarity score, while simultaneously applying any requested metadata filters.

Rendering: The frontend dynamically generates book cards based on the JSON response, color-coding the semantic match percentage.

👨‍💻 Author
Novaz Edd
Software Engineering Student | Aspiring AI & Machine Learning Engineer

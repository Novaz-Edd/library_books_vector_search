"""
AI-Powered Library Management & Semantic Search — Backend
-----------------------------------------------------------
Run with:
    pip install fastapi uvicorn chromadb sentence-transformers
    uvicorn backend:app --reload --port 8000

Architecture:
- ChromaDB (in-memory, ephemeral client) as the vector store + metadata payload store.
- sentence-transformers 'all-MiniLM-L6-v2' for local, API-key-free embeddings.
- FastAPI exposes /api/search, /api/books, and /api/checkout|/api/return
  to simulate real-time stock changes.
"""

import uuid
import random
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. App & DB setup
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Library Semantic Search API")

# Allow the frontend (served from anywhere/file://) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local, free, no-API-key embedding function (runs on CPU).
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# In-memory ChromaDB instance.
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.get_or_create_collection(
    name="library_books",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------------
# 2. Mock dataset generator
# ---------------------------------------------------------------------------

# Each entry: title, author, description, floor, shelf sector, stock_total
RAW_BOOKS = [
    # ---------------- Floor 1: Humanities, Arts, Fiction, Motivation ----------------
    ("Atomic Habits", "James Clear",
     "A practical guide to building good habits and breaking bad ones, focused on small "
     "incremental changes, motivation, self-improvement, and personal growth strategies for a better life.",
     1, "Aisle A", 5),
    ("Man's Search for Meaning", "Viktor Frankl",
     "A profound memoir on finding purpose and hope through extreme adversity, exploring "
     "psychology, resilience, motivation, and the human spirit during hardship.",
     1, "Aisle A", 3),
    ("The Great Gatsby", "F. Scott Fitzgerald",
     "A classic novel of the Jazz Age exploring wealth, love, the American dream, ambition, "
     "and tragedy in 1920s New York high society.",
     1, "Aisle B", 4),
    ("A Brief History of Art", "Helen Gardner",
     "An illustrated survey of painting, sculpture, and architecture across civilizations, "
     "covering Renaissance masters, modern art movements, and aesthetic theory.",
     1, "Aisle C", 2),
    ("The Stoic Path to Wisdom", "Massimo Pigliucci",
     "An exploration of ancient Stoic philosophy applied to modern life, covering resilience, "
     "emotional control, motivation, and living a meaningful, inspired life.",
     1, "Aisle A", 0),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez",
     "A landmark of magical realism following generations of a family, blending fiction, "
     "myth, history, and the human condition in a small town.",
     1, "Aisle B", 2),

    # ---------------- Floor 2: Computer Science, AI, Math, Hardware ----------------
    ("Python Crash Course", "Eric Matthes",
     "A hands-on introduction to programming in Python, covering coding fundamentals, "
     "data structures, automation projects, and software development best practices for laptops and computers.",
     2, "Aisle A", 6),
    ("Introduction to Algorithms", "Cormen, Leiserson, Rivest, Stein",
     "A comprehensive textbook on algorithms and data structures, covering sorting, graph "
     "theory, dynamic programming, and computational complexity for software engineers.",
     2, "Aisle A", 3),
    ("Deep Learning", "Ian Goodfellow",
     "A foundational text on artificial intelligence and neural networks, covering machine "
     "learning, deep learning architectures, backpropagation, and AI model training.",
     2, "Aisle B", 2),
    ("Computer Organization and Design", "Patterson & Hennessy",
     "An in-depth look at computer hardware architecture, processors, memory systems, and "
     "how software interacts with physical hardware engineering components.",
     2, "Aisle C", 0),
    ("Linear Algebra Done Right", "Sheldon Axler",
     "A rigorous treatment of vector spaces, linear transformations, and matrices, essential "
     "mathematics for engineering, computer science, and machine learning.",
     2, "Aisle D", 4),
    ("Clean Code", "Robert C. Martin",
     "A guide to writing maintainable, readable software, covering coding best practices, "
     "refactoring, software craftsmanship, and professional programming on modern laptops.",
     2, "Aisle A", 1),
    ("Artificial Intelligence: A Modern Approach", "Russell & Norvig",
     "The definitive textbook on artificial intelligence, covering search algorithms, "
     "machine learning, robotics, natural language processing, and intelligent agents.",
     2, "Aisle B", 3),

    # ---------------- Floor 3: Physics, Astrophysics, Astronomy, Chemistry ----------------
    ("A Brief History of Time", "Stephen Hawking",
     "An accessible exploration of cosmology, black holes, the Big Bang, and the nature of "
     "space and time, covering how the universe and stars were formed.",
     3, "Aisle A", 4),
    ("Astrophysics for People in a Hurry", "Neil deGrasse Tyson",
     "A concise tour of the cosmos covering galaxies, stars, planets, dark matter, and how "
     "stars form, evolve, and die across the universe.",
     3, "Aisle A", 0),
    ("Principles of Chemistry", "Peter Atkins",
     "A foundational chemistry textbook covering atomic structure, chemical bonding, "
     "reactions, thermodynamics, and the periodic table for science students.",
     3, "Aisle B", 5),
    ("The Elegant Universe", "Brian Greene",
     "An exploration of string theory, quantum mechanics, and general relativity, "
     "explaining how physicists seek a unified theory of the cosmos and fundamental forces.",
     3, "Aisle C", 2),
    ("Cosmos", "Carl Sagan",
     "A celebrated journey through astronomy and the universe, covering planets, stars, "
     "galaxies, the history of science, and humanity's place in the cosmos.",
     3, "Aisle A", 1),
    ("Organic Chemistry", "Paula Bruice",
     "A detailed textbook on carbon-based compounds, reaction mechanisms, molecular "
     "structures, and laboratory techniques in organic chemistry.",
     3, "Aisle D", 3),

    # ---------------- Floor 4: Business, Finance, Economics, Management ----------------
    ("The Intelligent Investor", "Benjamin Graham",
     "A classic guide to value investing, covering stock market strategy, financial analysis, "
     "risk management, and long-term wealth building.",
     4, "Aisle A", 3),
    ("Principles of Economics", "N. Gregory Mankiw",
     "An introductory textbook covering supply and demand, markets, macroeconomics, "
     "microeconomics, and how economies allocate resources.",
     4, "Aisle B", 0),
    ("Good to Great", "Jim Collins",
     "A research-based study of how companies transition from average to exceptional "
     "performance, covering leadership, management strategy, and business growth.",
     4, "Aisle A", 2),
    ("Zero to One", "Peter Thiel",
     "A perspective on startups, innovation, and building businesses that create new "
     "markets, covering entrepreneurship, technology, and competitive strategy.",
     4, "Aisle C", 4),
    ("Financial Statement Analysis", "Martin Fridson",
     "A practical guide to reading and interpreting balance sheets, income statements, and "
     "cash flow statements for corporate finance and investment decisions.",
     4, "Aisle B", 1),
    ("Thinking, Fast and Slow", "Daniel Kahneman",
     "An exploration of the two systems that drive human decision-making, covering "
     "behavioral economics, cognitive bias, and judgment under uncertainty.",
     4, "Aisle D", 2),
]


def seed_database():
    """Populate the ChromaDB collection with the mock dataset (idempotent)."""
    if collection.count() > 0:
        return

    ids, documents, metadatas = [], [], []
    for title, author, description, floor, shelf, stock_total in RAW_BOOKS:
        book_id = str(uuid.uuid4())
        # Embed title + description for richer semantic context
        documents.append(f"{title}. {description}")
        metadatas.append({
            "id": book_id,
            "title": title,
            "author": author,
            "description": description,
            "floor": floor,
            "shelf": shelf,
            "stock_total": stock_total,
            # available <= total; some books start fully checked out for demo realism
            "stock_available": stock_total,
        })
        ids.append(book_id)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)


seed_database()


# ---------------------------------------------------------------------------
# 3. Pydantic models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    floor_filter: Optional[List[int]] = Field(
        default=None, description="Restrict results to these floors, e.g. [1, 2]"
    )
    only_available: bool = Field(
        default=False, description="If true, exclude books with stock_available == 0"
    )
    top_k: int = Field(default=8, ge=1, le=50)


class BookResult(BaseModel):
    id: str
    title: str
    author: str
    description: str
    floor: int
    shelf: str
    stock_total: int
    stock_available: int
    similarity: float  # 0.0 - 1.0, higher = better match


class StockChangeRequest(BaseModel):
    book_id: str


# ---------------------------------------------------------------------------
# 4. Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/books", response_model=List[BookResult])
def list_books():
    """Return every book in the catalog (no search ranking applied)."""
    data = collection.get(include=["metadatas"])
    results = []
    for meta in data["metadatas"]:
        results.append(BookResult(
            id=meta["id"], title=meta["title"], author=meta["author"],
            description=meta["description"], floor=meta["floor"], shelf=meta["shelf"],
            stock_total=meta["stock_total"], stock_available=meta["stock_available"],
            similarity=1.0,
        ))
    return results


@app.post("/api/search", response_model=List[BookResult])
def search_books(req: SearchRequest):
    """
    Hybrid semantic + metadata search.

    1. Build a Chroma 'where' filter from floor_filter / only_available.
    2. Run vector similarity search (cosine) against the query embedding.
    3. Convert Chroma's cosine *distance* into an intuitive similarity percentage.
    """
    where_clauses = []
    if req.floor_filter:
        where_clauses.append({"floor": {"$in": req.floor_filter}})
    if req.only_available:
        where_clauses.append({"stock_available": {"$gt": 0}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    # Ask Chroma for more candidates than top_k in case filtering is strict
    n_results = min(max(req.top_k * 3, req.top_k), collection.count() or 1)

    query_result = collection.query(
        query_texts=[req.query],
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )

    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]

    results = []
    for meta, dist in zip(metadatas, distances):
        # Cosine distance in Chroma is in [0, 2]; convert to similarity in [0, 1]
        similarity = max(0.0, 1.0 - (dist / 2.0))
        results.append(BookResult(
            id=meta["id"], title=meta["title"], author=meta["author"],
            description=meta["description"], floor=meta["floor"], shelf=meta["shelf"],
            stock_total=meta["stock_total"], stock_available=meta["stock_available"],
            similarity=round(similarity, 4),
        ))

    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[: req.top_k]


@app.post("/api/checkout")
def checkout_book(req: StockChangeRequest):
    """Simulate a student checking out a book (decrements stock_available)."""
    record = collection.get(ids=[req.book_id], include=["metadatas"])
    if not record["metadatas"]:
        raise HTTPException(status_code=404, detail="Book not found")

    meta = record["metadatas"][0]
    if meta["stock_available"] <= 0:
        raise HTTPException(status_code=400, detail="No copies available")

    meta["stock_available"] -= 1
    collection.update(ids=[req.book_id], metadatas=[meta])
    return {"id": req.book_id, "stock_available": meta["stock_available"]}


@app.post("/api/return")
def return_book(req: StockChangeRequest):
    """Simulate a student returning a book (increments stock_available, capped at total)."""
    record = collection.get(ids=[req.book_id], include=["metadatas"])
    if not record["metadatas"]:
        raise HTTPException(status_code=404, detail="Book not found")

    meta = record["metadatas"][0]
    if meta["stock_available"] >= meta["stock_total"]:
        raise HTTPException(status_code=400, detail="All copies already returned")

    meta["stock_available"] += 1
    collection.update(ids=[req.book_id], metadatas=[meta])
    return {"id": req.book_id, "stock_available": meta["stock_available"]}


@app.get("/api/stats")
def stats():
    """Quick aggregate stats, e.g. for a dashboard widget."""
    data = collection.get(include=["metadatas"])
    total_books = sum(m["stock_total"] for m in data["metadatas"])
    available_books = sum(m["stock_available"] for m in data["metadatas"])
    by_floor = {}
    for m in data["metadatas"]:
        by_floor.setdefault(m["floor"], {"total": 0, "available": 0, "titles": 0})
        by_floor[m["floor"]]["total"] += m["stock_total"]
        by_floor[m["floor"]]["available"] += m["stock_available"]
        by_floor[m["floor"]]["titles"] += 1

    return {
        "total_titles": collection.count(),
        "total_books": total_books,
        "available_books": available_books,
        "by_floor": by_floor,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
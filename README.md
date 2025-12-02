#📚 LitSense – Semantic Book Recommender System ⚡
LitSense is a lightning-fast, AI-powered book recommendation engine built using Gradio and Sentence Transformers. It goes beyond simple keyword matching by using Semantic Search to understand the "vibe" and meaning of your query. It leverages FAISS for vector indexing, Hugging Face pipelines for emotion/genre classification, and the Google Gemini API to generate personalized explanations for every recommendation.

<p align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Gradio-Frontend-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/> <img src="https://img.shields.io/badge/Sentence_Transformers-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/> <img src="https://img.shields.io/badge/Google%20Gemini-GenAI-4285F4?style=for-the-badge&logo=google&logoColor=white"/> <img src="https://img.shields.io/badge/FAISS-Vector_Search-00ADD8?style=for-the-badge&logo=meta&logoColor=white"/> <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

</p>

🌐 Live Demo
You can experience the interactive application live here: 📚 LitSense Live (Replace with your live link)

📚 Table of Contents
📚 Overview

🌐 Live Demo

🚀 Features

🛠 Tech Stack

📸 Screenshots

⚙️ How It Works

⚡ Performance

🔮 Future Enhancements

🚀 Getting Started

Prerequisites

Installation

Running the Application

📄 License

📬 Contact

🚀 Features
🧠 Semantic Search (Not Just Keywords)
Uses Sentence-BERT (all-MiniLM-L6-v2) to convert search queries into vector embeddings.

Finds books based on meaning, plot, and themes rather than just matching titles.

😊 Mood & Genre Filtering
Mood Analysis: Uses a DistilRoBERTa model to detect emotional tones (e.g., Joy, Calm, Suspense) in book descriptions.

Smart Genre Classification: Combines aggressive keyword matching with a Zero-Shot BART classifier for high accuracy.

✨ AI-Powered Explanations
Integrated with Google Gemini 2.5 Flash to provide a "Why this matches" explanation for every book recommended.

Explains the connection between your specific query and the book's content.

⚡ High-Performance Architecture
Parallel Processing: Uses ThreadPoolExecutor to fetch book data and generate explanations concurrently.

Smart Caching: Implements LRU Caching and persistent disk caching (pickle) for instant results on repeated queries.

Progressive Loading: Fetches data in batches to ensure the UI remains responsive.

🛠 Tech Stack
1. Core AI & Search:
Sentence Transformers: Generates dense vector embeddings for book descriptions to enable semantic search.

FAISS (Facebook AI Similarity Search): Performs high-speed similarity searches on the vector embeddings.

Hugging Face Transformers: - facebook/bart-large-mnli for Zero-Shot Genre Classification.

j-hartmann/emotion-english-distilroberta-base for Mood/Emotion detection.

2. Backend & APIs:
Python: The core logic handling data processing, API calls, and threading.

Google Books API: The primary source for fetching real-time book metadata.

Open Library API: Serves as a robust fallback data source.

Google Gemini API: Generates context-aware textual explanations for recommendations.

3. Frontend:
Gradio: Builds the clean, responsive web interface with dropdowns, sliders, and real-time Markdown output.

📸 Screenshots
🏠 Home Interface
The clean landing page with semantic search input, mood sliders, and genre filters.

😊 Mood Filtering
Example of filtering specifically for "Calm" books suitable for bedtime reading.

📂 Genre Filtering
Targeted search results filtering for "Business" and productivity books.

💡 Pro Tips & Instructions
Built-in guide helping users maximize the search potential.

🔄 Refreshed Results
The system tracks history to ensure you get fresh recommendations when searching the same topic again.

⚙️ How It Works
The application follows a high-speed pipeline optimized for sub-4-second responses:
1. Data Fetching: The system fetches book candidates in parallel from Google Books API based on the user's query.

2. Vector Embedding: The book descriptions are passed through all-MiniLM-L6-v2 to create 384-dimensional vectors.

3. Semantic Search: A FAISS index is built on the fly, and the user's query is compared against the books to find the closest semantic matches.

4. Intelligent Filtering: - Genre: Checks metadata first, then uses keyword patterns, and falls back to AI classification if necessary. - Mood: Analyzes the emotional tone of the description to match the user's requested mood.

5. AI Explanation: The top matches are sent to Google Gemini to generate a custom "Why this matches" explanation.

6. Display: Results are rendered in Gradio with covers, metadata, and the AI explanation.

⚡ Performance Optimization
LitSense is engineered for speed. It includes a dedicated test_performance.py suite to benchmark the pipeline.

Time Budgeting: The app allocates strict time budgets (e.g., 40% fetch, 30% search) to prevent timeouts.

Cache Persistence: utils.py manages a books_cache.pkl to store API responses, making repeated searches instant.

Speed Presets: config.py allows toggling between 'Fastest', 'Balanced', and 'Accurate' modes to tune the speed/accuracy tradeoff.

🔮 Future Enhancements
Integration with local LLMs (Llama 3) for offline explanations.

User bookshelves and "To Read" lists using local storage.

Advanced visualization of the vector space (book clusters).

Commercial API integration (Amazon/Goodreads) for direct purchase links.

🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8+

pip package manager

Google Gemini API Key

Google Books API Key (Optional, but recommended)

Installation
1. Clone the repository:

Bash

git clone https://github.com/YourUsername/LitSense.git
cd LitSense
pip install -r requirements.txt
2. Set up environment variables:

Create a .env file in the root directory and add your API keys:

Bash

GEMINI_API_KEY=your_gemini_key_here
GOOGLE_BOOKS_API_KEY=your_google_books_key_here
Running the Application
Bash

python app.py
Once running, open the web app in your browser at:

Bash

http://127.0.0.1:7860
📄 License
This project is licensed under the MIT License — free to use, modify, and distribute with proper attribution.

📬 Contact
📨 Email: [Your Email Here]

🔗 LinkedIn: [Your LinkedIn URL Here]

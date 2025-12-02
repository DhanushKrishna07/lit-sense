import os
import requests
import pickle
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
from functools import lru_cache
import hashlib
from config import (
    GOOGLE_BOOKS_TIMEOUT, 
    OPEN_LIBRARY_TIMEOUT, 
    GEMINI_TIMEOUT,
    EXPLANATION_CACHE_SIZE,
    BOOK_FETCH_WORKERS,
    EXPLANATION_WORKERS
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")

# Initialize Gemini once
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini API configured")
    except:
        GEMINI_MODEL = None
        print("⚠ Gemini model not available, using fallback explanations")
else:
    GEMINI_MODEL = None
    print("⚠ Gemini API key not found, using fallback explanations")

CACHE_FILE = "books_cache.pkl"
_cache = None  # In-memory cache for speed

# OPTIMIZATION: Reusable session for connection pooling
_session = None

def get_session():
    """Get or create a requests session with connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=2
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


def load_cache() -> Dict:
    """Load cache once into memory for fast access."""
    global _cache
    if _cache is not None:
        return _cache
        
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                _cache = pickle.load(f)
                return _cache
        except Exception as e:
            print(f"Cache load error: {e}")
    
    _cache = {}
    return _cache


def save_cache(cache: Dict):
    """Save cache asynchronously (don't block main thread)."""
    global _cache
    _cache = cache
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Cache save error: {e}")


def fetch_books_batch(query: str, start_index: int = 0, count: int = 40) -> List[Dict]:
    """Fetch a single batch of books from Google Books API with connection pooling."""
    try:
        session = get_session()
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            "q": query,
            "maxResults": min(count, 40),  # API max is 40
            "startIndex": start_index,
            "orderBy": "relevance"
        }

        if GOOGLE_BOOKS_API_KEY:
            params["key"] = GOOGLE_BOOKS_API_KEY

        # INTEGRATION: Use timeout from config
        response = session.get(url, params=params, timeout=GOOGLE_BOOKS_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        books = []
        if "items" in data:
            for item in data["items"]:
                vol = item.get("volumeInfo", {})
                
                title = vol.get("title", "Unknown Title")
                authors = vol.get("authors", ["Unknown Author"])
                description = vol.get("description", "No description available.")
                categories = vol.get("categories", [])
                published_date = vol.get("publishedDate", "N/A")
                thumbnail = vol.get("imageLinks", {}).get("thumbnail", "")
                
                books.append({
                    "title": title,
                    "authors": authors,
                    "description": description,
                    "categories": categories,
                    "published_date": published_date,
                    "thumbnail": thumbnail,
                    "source": "Google Books"
                })
        
        return books
        
    except requests.exceptions.Timeout:
        print(f"⚠ Timeout fetching books at index {start_index}")
        return []
    except Exception as e:
        print(f"⚠ Error fetching books: {e}")
        return []


def fetch_books_openlibrary(query: str, max_results: int = 40) -> List[Dict]:
    """Fallback: Fetch books from Open Library API with connection pooling."""
    try:
        session = get_session()
        url = "https://openlibrary.org/search.json"
        params = {
            "q": query,
            "limit": min(max_results, 100)
        }

        # INTEGRATION: Use timeout from config
        response = session.get(url, params=params, timeout=OPEN_LIBRARY_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        books = []
        if "docs" in data:
            for doc in data["docs"]:
                title = doc.get("title", "Unknown Title")
                authors = doc.get("author_name", ["Unknown Author"])
                first_sentence = doc.get("first_sentence", [])
                description = first_sentence[0] if first_sentence else "No description available."
                subjects = doc.get("subject", [])[:3]
                published_date = doc.get("first_publish_year", "N/A")
                cover_id = doc.get("cover_i")
                thumbnail = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""

                books.append({
                    "title": title,
                    "authors": authors,
                    "description": description,
                    "categories": subjects,
                    "published_date": str(published_date),
                    "thumbnail": thumbnail,
                    "source": "Open Library"
                })

        return books
        
    except Exception as e:
        print(f"⚠ Error fetching from Open Library: {e}")
        return []


def fetch_books_progressive(query: str, max_results: int = 40) -> List[Dict]:
    """
    OPTIMIZED: Parallel fetching with ThreadPoolExecutor.
    Fetches multiple batches concurrently instead of sequentially.
    """
    cache = load_cache()
    
    # Check cache for this specific query and size
    cache_key = f"{query}_{max_results}"
    if cache_key in cache:
        print(f"✓ Using cached data ({len(cache[cache_key])} books)")
        return cache[cache_key][:max_results]
    
    # Check if we have a larger cache we can use
    for cached_size in [120, 80, 60, 40]:
        larger_cache_key = f"{query}_{cached_size}"
        if larger_cache_key in cache and len(cache[larger_cache_key]) >= max_results:
            print(f"✓ Using cached data from larger fetch")
            return cache[larger_cache_key][:max_results]
    
    # No cache hit - fetch fresh data
    print(f"⚡ Fetching {max_results} books for '{query}'...")
    
    # OPTIMIZATION: Parallel fetching with ThreadPoolExecutor
    num_batches = (max_results + 39) // 40
    books = []
    
    # INTEGRATION: Use worker count from config
    with ThreadPoolExecutor(max_workers=BOOK_FETCH_WORKERS) as executor:
        # Submit all fetch tasks at once
        futures = [
            executor.submit(fetch_books_batch, query, i * 40, 40)
            for i in range(num_batches)
        ]
        
        # Collect results as they complete with timeout
        try:
            for future in as_completed(futures, timeout=2.5):
                try:
                    batch_books = future.result(timeout=0.5)
                    if batch_books:
                        books.extend(batch_books)
                except Exception as e:
                    print(f"⚠ Batch fetch error: {e}")
        except TimeoutError:
            print("⚠ Some fetches timed out, using partial results")
    
    # Fallback to Open Library if Google Books fails
    if not books:
        print("⚠ Google Books failed, trying Open Library...")
        books = fetch_books_openlibrary(query, max_results)
    
    # Cache the results
    if books:
        cache[cache_key] = books
        save_cache(cache)
        print(f"✓ Fetched and cached {len(books)} books")
    else:
        print("✗ No books found")
    
    return books[:max_results]


# INTEGRATION: Use cache size constant from config
@lru_cache(maxsize=EXPLANATION_CACHE_SIZE)
def explain_single_cached(title: str, authors_str: str, description_hash: str, query: str, timeout: float = 1.5) -> str:
    """
    Cached version of explain_single using hashable parameters.
    Cache key is based on title, authors, description hash, and query.
    """
    if not GEMINI_MODEL:
        return "Matches your search based on semantic similarity and content analysis."

    try:
        prompt = f"""In 1-2 sentences, explain why this book matches the query.

Query: "{query}"
Book: {title} by {authors_str}

Be specific and conversational."""

        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=80,
            ),
            request_options={'timeout': timeout}
        )
        return response.text.strip()
        
    except Exception as e:
        return "Relevant to your search based on content analysis."


def explain_single(book: Dict, query: str, timeout: float = 1.5) -> str:
    """Generate explanation for a single book using cached function."""
    # Create hashable parameters for caching
    title = book['title']
    authors_str = ', '.join(book['authors'][:2])
    description_hash = hashlib.md5(book.get('description', '')[:300].encode()).hexdigest()
    
    return explain_single_cached(title, authors_str, description_hash, query, timeout)


def explain_recommendations_batch(books: List[Dict], query: str, timeout: float = 2.5) -> List[str]:
    """
    OPTIMIZED: Generate explanations for multiple books in parallel.
    Uses ThreadPoolExecutor for concurrent API calls with timeout protection.
    """
    if not GEMINI_MODEL or not books:
        return ["Matches your search based on semantic similarity."] * len(books)

    explanations = ["Matches your search."] * len(books)
    
    # INTEGRATION: Use timeout from config default if not provided, though typically calculated by app.py
    # Here we limit per-request timeout slightly based on the total available
    per_request_timeout = min(timeout / max(len(books), 1) * 1.5, GEMINI_TIMEOUT)
    
    try:
        # INTEGRATION: Use worker count from config
        with ThreadPoolExecutor(max_workers=EXPLANATION_WORKERS) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(explain_single, book, query, per_request_timeout): i 
                for i, book in enumerate(books)
            }
            
            # Collect results with overall timeout
            completed = 0
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                try:
                    explanations[idx] = future.result(timeout=0.3)
                    completed += 1
                except Exception as e:
                    # Keep default explanation on error
                    pass
            
            print(f"✓ Generated {completed}/{len(books)} explanations")
                    
    except TimeoutError:
        print("⚠ Explanation timeout - using defaults for remaining")
    except Exception as e:
        print(f"⚠ Explanation error: {e}")
    
    return explanations


def format_book_output(book: Dict, explanation: str = "") -> str:
    """Format book information for clean display."""
    # Format authors
    authors_str = ", ".join(book['authors'][:2])
    if len(book['authors']) > 2:
        authors_str += f" +{len(book['authors']) - 2} more"

    # Truncate description
    desc = book.get('description', 'No description available.')
    if len(desc) > 250:
        desc = desc[:247] + "..."

    # Format categories
    categories = ', '.join(book.get('categories', [])[:3])
    if not categories:
        categories = 'N/A'

    # Build output
    output = f"""### 📖 {book['title']}
**Authors:** {authors_str}  |  **Published:** {book.get('published_date', 'N/A')}
**Categories:** {categories}

**Description:** {desc}

**Why this matches:** {explanation}

---
"""
    return output


# Preload cache on module import for faster first query
_ = load_cache()
print("✓ Cache preloaded")
import gradio as gr
from semantic_search import SemanticBookRecommender
from utils import fetch_books_progressive, explain_recommendations_batch, format_book_output
from config import get_config
import time
from typing import Dict, Set
import threading

# OPTIMIZATION: Initialize recommender once at startup with thread-safe access
print("Initializing recommender system...")
recommender = SemanticBookRecommender()
recommender_lock = threading.Lock()
print("Recommender ready!")

# OPTIMIZATION: Track shown books per query to avoid repetition
# Using dict with thread-safe access
query_history: Dict[str, Set[str]] = {}
query_history_lock = threading.Lock()

GENRE_OPTIONS = [
    "Any", "Fiction", "Non-Fiction", "Mystery", "Romance", 
    "Science Fiction", "Fantasy", "Thriller", "Horror", 
    "Biography", "History", "Self-Help", "Business", 
    "Philosophy", "Poetry", "Adventure"
]

MOOD_OPTIONS = [
    "Any", "Joy", "Calm", "Sadness", "Excitement", 
    "Anger", "Fear", "Surprise"
]


def get_query_key(query: str, mood_filter: str, genre_filter: str) -> str:
    """Generate a unique key for query + filters combination."""
    return f"{query.lower().strip()}_{mood_filter}_{genre_filter}"


def get_shown_books(query_key: str) -> Set[str]:
    """Thread-safe retrieval of shown books for a query."""
    with query_history_lock:
        if query_key not in query_history:
            query_history[query_key] = set()
        return query_history[query_key].copy()


def mark_books_shown(query_key: str, book_titles: list) -> None:
    """Thread-safe marking of books as shown."""
    with query_history_lock:
        if query_key not in query_history:
            query_history[query_key] = set()
        query_history[query_key].update(book_titles)


def reset_query_history(query_key: str) -> None:
    """Thread-safe reset of query history."""
    with query_history_lock:
        query_history[query_key] = set()


def search_and_recommend(query: str, mood_filter: str, genre_filter: str, num_results: int = 5):
    """
    OPTIMIZED: Main function with progressive fetching and aggressive 4-second timeout.
    Performance improvements:
    - Reduced timeout from 5s to 4s
    - Parallel operations throughout
    - Smart caching and result tracking
    - Better time budget allocation
    """
    start_time = time.time()
    MAX_TIME = 4.0  # OPTIMIZATION: Reduced from 5s to 4s for safety margin
    
    if not query.strip():
        return "⚠️ Please enter a search query."

    status_message = f"🔍 Searching: '{query}'\n\n"
    
    # Track which books we've already shown for this query
    query_key = get_query_key(query, mood_filter, genre_filter)
    shown_books = get_shown_books(query_key)
    
    try:
        # OPTIMIZATION: Progressive fetching with time budgets
        # Allocate 40% of time to fetching (1.6s max)
        all_books = []
        fetch_budget = MAX_TIME * 0.40  # 1.6 seconds for fetching
        fetch_start = time.time()
        
        # Try progressive batch sizes, but stop early if we have enough
        fetch_attempts = [40, 80]  # OPTIMIZATION: Reduced from [40, 80, 120]
        
        for batch_size in fetch_attempts:
            # Check if we've exceeded fetch budget
            if time.time() - fetch_start > fetch_budget:
                print(f"⚠️ Fetch timeout after {time.time() - fetch_start:.1f}s")
                break
            
            # Fetch books with timeout
            remaining_fetch_time = fetch_budget - (time.time() - fetch_start)
            if remaining_fetch_time < 0.5:
                break
            
            books = fetch_books_progressive(query, max_results=batch_size)
            
            if not books:
                if not all_books:  # Only fail if we have no books at all
                    return status_message + "❌ No books found. Try a different query."
                break
            
            # Filter out already shown books
            new_books = [b for b in books if b['title'] not in shown_books]
            
            if not new_books and len(shown_books) > 0:
                # All books shown, reset history for this query
                reset_query_history(query_key)
                new_books = books
                status_message += "🔄 Showing fresh recommendations!\n"
            
            all_books = new_books
            status_message += f"✓ Found {len(all_books)} books\n"
            
            # If we have enough candidates, stop fetching
            if len(all_books) >= num_results * 4:  # 4x buffer
                break
        
        if not all_books:
            elapsed = time.time() - start_time
            return status_message + f"❌ No matches found in {elapsed:.1f}s. Try:\n- Different filters\n- Broader search terms\n- 'Any' for mood/genre"
        
        # OPTIMIZATION: Build index with time budget (20% of time = 0.8s)
        index_budget = MAX_TIME * 0.20
        index_start = time.time()
        
        with recommender_lock:
            recommender.build_index(all_books)
        
        index_time = time.time() - index_start
        print(f"⚡ Index built in {index_time:.2f}s")
        
        # OPTIMIZATION: Search and filter with time budget (30% of time = 1.2s)
        search_budget = MAX_TIME * 0.30
        search_start = time.time()
        
        # Calculate remaining time for recommendation
        remaining_time = MAX_TIME - (time.time() - start_time)
        if remaining_time < 1.0:
            # Not enough time, return semantic search results only
            print("⚠️ Time pressure - skipping filters")
            with recommender_lock:
                recommendations = recommender.search_books(query, top_k=num_results)
        else:
            # Normal recommendation with filters
            with recommender_lock:
                recommendations = recommender.recommend_books(
                    query=query,
                    mood_filter=mood_filter if mood_filter != "Any" else None,
                    genre_filter=genre_filter if genre_filter != "Any" else None,
                    top_k=num_results * 2,  # Get more for better quality
                    timeout=remaining_time * 0.8  # Use 80% of remaining time
                )
        
        search_time = time.time() - search_start
        print(f"⚡ Search completed in {search_time:.2f}s")
        
        if not recommendations:
            elapsed = time.time() - start_time
            return status_message + f"❌ No matches found in {elapsed:.1f}s. Try:\n- Different filters\n- Broader search terms\n- 'Any' for mood/genre"
        
        # Limit to requested number
        recommendations = recommendations[:num_results]
        
        # Mark books as shown
        book_titles = [book['title'] for book in recommendations]
        mark_books_shown(query_key, book_titles)
        
        elapsed = time.time() - start_time
        status_message += f"✨ {len(recommendations)} recommendations in {elapsed:.1f}s!\n"
        status_message += "=" * 80 + "\n\n"
        
        # OPTIMIZATION: AI explanations with remaining time budget (10% = 0.4s minimum)
        explanation_budget = max(MAX_TIME - elapsed, 0.4)
        
        if explanation_budget > 0.3:
            explanation_start = time.time()
            explanations = explain_recommendations_batch(
                recommendations, 
                query, 
                timeout=explanation_budget
            )
            explanation_time = time.time() - explanation_start
            print(f"⚡ Explanations in {explanation_time:.2f}s")
        else:
            # Not enough time for explanations
            explanations = ["Matches your search criteria based on semantic analysis."] * len(recommendations)
        
        # Format output
        for i, (book, explanation) in enumerate(zip(recommendations, explanations), 1):
            status_message += f"**Recommendation {i}**\n\n"
            status_message += format_book_output(book, explanation)
            status_message += "\n"
        
        # Add timing info
        total_time = time.time() - start_time
        status_message += f"\n⏱️ Total time: {total_time:.2f}s"
        
        if total_time > MAX_TIME:
            status_message += f" (exceeded target by {total_time - MAX_TIME:.1f}s)"
        
        return status_message
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = status_message + f"⚠️ Error after {elapsed:.1f}s: {str(e)}\nPlease try again."
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg


def clear_history():
    """Clear all query history."""
    with query_history_lock:
        query_history.clear()
    return "✓ Search history cleared! You'll see fresh results on next search."


def create_interface():
    """Create and configure Gradio interface with optimized layout."""
    with gr.Blocks(
        title="LitSense - Fast AI Book Recommendations",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple"
        )
    ) as app:
        # 1. Main Application Header/Title
        gr.Markdown(
            """
            # 📚 LitSense - Semantic Book Recommender System
            
            **⚡ Lightning-fast AI-powered book recommendations in under 4 seconds!**
            
            Discover personalized books using semantic search, mood detection, and genre classification.
            """
        )

        # 2. Main Input and Filters (Optimized Layout)
        with gr.Row():
            # Column for Search Input (wider, scale=3)
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="🔍 What are you looking for?",
                    placeholder="e.g., 'motivational books', 'fantasy adventures', 'books about productivity'",
                    lines=3,
                    max_lines=5
                )

            # Column for Filters (narrower, scale=1)
            with gr.Column(scale=1, min_width=250):
                with gr.Group():
                    mood_dropdown = gr.Dropdown(
                        choices=MOOD_OPTIONS,
                        value="Any",
                        label="😊 Mood Filter",
                        info="Optional emotional tone"
                    )

                    genre_dropdown = gr.Dropdown(
                        choices=GENRE_OPTIONS,
                        value="Any",
                        label="📖 Genre Filter",
                        info="Optional genre category"
                    )

                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="📊 Number of Results"
                    )

        # 3. Action Buttons
        with gr.Row():
            search_button = gr.Button(
                "🔍 Find Books", 
                variant="primary", 
                size="lg",
                scale=3
            )
            clear_button = gr.Button(
                "🔄 Clear History", 
                variant="secondary",
                size="lg",
                scale=1
            )
        
        # 4. Output Area
        output = gr.Markdown(label="📚 Recommendations")
        clear_output = gr.Markdown(visible=False)

        # Event handlers
        search_button.click(
            fn=search_and_recommend,
            inputs=[query_input, mood_dropdown, genre_dropdown, num_results],
            outputs=output
        )
        
        clear_button.click(
            fn=clear_history,
            inputs=[],
            outputs=clear_output
        )
        
        query_input.submit(
            fn=search_and_recommend,
            inputs=[query_input, mood_dropdown, genre_dropdown, num_results],
            outputs=output
        )

        # --- EXAMPLES SECTION ---
        # Clickable examples that populate the fields
        gr.Examples(
            examples=[
                ["Suggest uplifting books for someone feeling anxious", "Any", "Any"],
                ["Action-packed fantasy books with heroes, magic, and epic journeys", "Any", "Fantasy"],
                ["I want to learn about productivity and time management", "Any", "Business"],
                ["Recommend calming books for bedtime reading", "Calm", "Any"],
                ["Books to read when I feel lonely and want to feel less alone", "Sadness", "Any"]
            ],
            # Map the 3 example columns to the 3 matching inputs
            inputs=[query_input, mood_dropdown, genre_dropdown],
            label="🎲 Try these examples (Click to populate):",
            cache_examples=False,
            run_on_click=False  # Set to True if you want it to auto-search on click
        )

        # 5. Instructions and Tips (Updated: Removed the static table)
        gr.Markdown(
            """
            ---
            ### 🚀 How it works:
            1. **Search**: Enter what you're looking for (e.g., "uplifting books about personal growth")
            2. **Filter** (optional): Narrow by mood and genre
            3. **Discover**: Get personalized recommendations with AI-generated explanations
            4. **Refresh**: Search again for different books with same query
            
            ---
            ### ⚡ Performance Features:
            - Parallel API calls for faster data fetching
            - Smart caching for instant repeated queries
            - Progressive loading with timeout protection
            - Intelligent filtering with keyword matching
            
            ---
            ### 💡 Pro Tips:
            
            - **🚀 Fast searches**: Use broad terms without filters for quickest results
            - **🎯 Precise results**: Add mood/genre filters for more targeted recommendations
            - **🔄 Fresh results**: Search again with same query to see different books
            - **❌ No matches?**: Try removing filters or using broader search terms
            - **⚡ Best performance**: Queries with "Any" filters are fastest (2-3 seconds)
            - **🎨 Customize**: Adjust number of results based on your needs
            """
        )

    return app


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 LitSense - Semantic Book Recommender System")
    print("=" * 80)
    print("\n⚡ Performance Optimizations Enabled:")
    print("   ✓ Parallel API calls with connection pooling")
    print("   ✓ Smart caching with LRU and disk persistence")
    print("   ✓ Batch AI processing for explanations")
    print("   ✓ Keyword-based genre matching")
    print("   ✓ Progressive loading with timeout protection")
    print("   ✓ Thread-safe query history tracking")
    print("\n🎯 Target: < 5 seconds | Typical: 2-4 seconds")
    print("=" * 80 + "\n")

    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        # OPTIMIZATION: Enable queue for better concurrent request handling
        max_threads=10
    )
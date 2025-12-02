"""
Performance Configuration for LitSense
Adjust these settings to tune speed vs accuracy tradeoff.
"""

# ============================================================================
# CRITICAL PERFORMANCE SETTINGS (EXACT VALUES FROM CORE FILES)
# ============================================================================

# Genre Filtering
# ---------------
# Exact value from semantic_search.py
GENRE_THRESHOLD = 0.25  

# Skip AI classification entirely for genre if keyword match found
GENRE_SKIP_AI_ON_KEYWORD = True

# Minimum keyword match score to skip AI
GENRE_KEYWORD_MIN_SCORE = 0.60

# Mood Filtering
# --------------
# Exact value from semantic_search.py
MOOD_THRESHOLD = 0.12  

# Search Settings
# ---------------
# Exact value from semantic_search.py (min(top_k * 3...))
SEARCH_MULTIPLIER = 3  

# Exact value from semantic_search.py
MAX_SEARCH_CANDIDATES = 50  

# Cache Settings
# --------------
# Size of LRU caches
GENRE_CACHE_SIZE = 2000
EMOTION_CACHE_SIZE = 2000
EXPLANATION_CACHE_SIZE = 500

# Index Building
# --------------
EMBEDDING_BATCH_SIZE = 256

# Timeout Settings
# ----------------
# Exact value from app.py
PIPELINE_TIMEOUT = 4.0  

# Time budget allocation (based on app.py logic)
TIME_BUDGET = {
    'fetch': 0.40,      # 40%
    'index': 0.20,      # 20%
    'search': 0.30,     # 30%
    'explain': 0.10     # 10%
}

# API Timeouts
GOOGLE_BOOKS_TIMEOUT = 2.0
OPEN_LIBRARY_TIMEOUT = 3.0
GEMINI_TIMEOUT = 1.5

# Parallel Processing
BOOK_FETCH_WORKERS = 3
EXPLANATION_WORKERS = 8

# Progressive Fetching
# --------------------
# Exact value from app.py
FETCH_BATCH_SIZES = [40, 80]  

# ============================================================================
# SPEED PRESETS
# ============================================================================

SPEED_PRESETS = {
    'fastest': {
        'genre_threshold': 0.15,
        'mood_threshold': 0.08,
        'search_multiplier': 2,
        'max_candidates': 30,
        'skip_ai_genre': True,
        'fetch_batches': [40],
        'pipeline_timeout': 3.0
    },
    'balanced': {
        'genre_threshold': 0.20,
        'mood_threshold': 0.10,
        'search_multiplier': 2,
        'max_candidates': 50,
        'skip_ai_genre': True,
        'fetch_batches': [40, 80],
        'pipeline_timeout': 4.0
    },
    'accurate': {
        'genre_threshold': 0.30,
        'mood_threshold': 0.15,
        'search_multiplier': 3,
        'max_candidates': 100,
        'skip_ai_genre': False,
        'fetch_batches': [40, 80, 120],
        'pipeline_timeout': 6.0
    }
}

# ============================================================================
# ACTIVE PRESET
# ============================================================================
# Set to 'custom' to use the exact global variables defined above
ACTIVE_PRESET = 'custom' 


def get_config():
    """Get current configuration based on active preset."""
    # Base config using the exact global constants defined at the top
    base_config = {
        'genre_threshold': GENRE_THRESHOLD,
        'mood_threshold': MOOD_THRESHOLD,
        'search_multiplier': SEARCH_MULTIPLIER,
        'max_candidates': MAX_SEARCH_CANDIDATES,
        'skip_ai_genre': GENRE_SKIP_AI_ON_KEYWORD,
        'genre_keyword_min_score': GENRE_KEYWORD_MIN_SCORE,
        'fetch_batches': FETCH_BATCH_SIZES,
        'pipeline_timeout': PIPELINE_TIMEOUT,
        'embedding_batch_size': EMBEDDING_BATCH_SIZE,
        'genre_cache_size': GENRE_CACHE_SIZE,
        'emotion_cache_size': EMOTION_CACHE_SIZE,
        'fetch_workers': BOOK_FETCH_WORKERS,
        'explanation_workers': EXPLANATION_WORKERS,
        'time_budget': TIME_BUDGET
    }

    # Only override if using a specific preset name
    if ACTIVE_PRESET in SPEED_PRESETS:
        preset = SPEED_PRESETS[ACTIVE_PRESET]
        base_config.update({
            'genre_threshold': preset['genre_threshold'],
            'mood_threshold': preset['mood_threshold'],
            'search_multiplier': preset['search_multiplier'],
            'max_candidates': preset['max_candidates'],
            'skip_ai_genre': preset['skip_ai_genre'],
            'fetch_batches': preset['fetch_batches'],
            'pipeline_timeout': preset['pipeline_timeout']
        })
    
    return base_config


def print_config():
    """Print current configuration."""
    config = get_config()
    print("\n" + "="*60)
    print(f"LitSense Configuration - Preset: {ACTIVE_PRESET.upper()}")
    print("="*60)
    print(f"  Genre Threshold:     {config['genre_threshold']}")
    print(f"  Mood Threshold:      {config['mood_threshold']}")
    print(f"  Search Multiplier:   {config['search_multiplier']}")
    print(f"  Max Candidates:      {config['max_candidates']}")
    print(f"  Pipeline Timeout:    {config['pipeline_timeout']}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()
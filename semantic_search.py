import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict, Optional
import warnings
import hashlib
import time
from functools import lru_cache
import re
from config import get_config, GENRE_CACHE_SIZE, EMOTION_CACHE_SIZE

warnings.filterwarnings('ignore')


class SemanticBookRecommender:
    def __init__(self):
        print("Loading models...")
        
        # INTEGRATION: Load configuration
        self.config = get_config()

        # Lightweight, fast model
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embeddings model loaded")

        # Load classifiers lazily (only when needed)
        self._zero_shot_classifier = None
        self._emotion_classifier = None
        
        self.index = None
        self.books = []
        self.embeddings = None
        
        self._last_books_hash = None
        
        # CRITICAL OPTIMIZATION: Comprehensive genre keywords
        self.genre_keywords = {
            'fiction': {
                'primary': ['fiction', 'novel', 'story', 'tale'],
                'exclude': ['non-fiction', 'nonfiction']
            },
            'non-fiction': {
                'primary': ['non-fiction', 'nonfiction', 'essay', 'memoir', 'true story', 'biography', 'autobiography'],
                'exclude': ['fiction', 'novel']
            },
            'mystery': {
                'primary': ['mystery', 'detective', 'crime', 'whodunit', 'murder', 'investigation', 'sleuth'],
                'exclude': []
            },
            'romance': {
                'primary': ['romance', 'love', 'romantic', 'relationship', 'dating', 'marriage'],
                'exclude': []
            },
            'science fiction': {
                'primary': ['science fiction', 'sci-fi', 'scifi', 'cyberpunk', 'space', 'alien', 'futuristic', 'dystopia', 'utopia'],
                'exclude': []
            },
            'fantasy': {
                'primary': ['fantasy', 'magic', 'wizard', 'dragon', 'epic', 'quest', 'medieval', 'sorcery', 'enchant'],
                'exclude': []
            },
            'thriller': {
                'primary': ['thriller', 'suspense', 'tension', 'chase', 'conspiracy'],
                'exclude': []
            },
            'horror': {
                'primary': ['horror', 'scary', 'terror', 'ghost', 'haunted', 'supernatural', 'zombie', 'vampire'],
                'exclude': []
            },
            'biography': {
                'primary': ['biography', 'memoir', 'autobiography', 'life story', 'life of'],
                'exclude': []
            },
            'history': {
                'primary': ['history', 'historical', 'war', 'ancient', 'century', 'revolution', 'empire'],
                'exclude': []
            },
            'self-help': {
                'primary': ['self-help', 'self help', 'improvement', 'motivational', 'personal development', 'mindfulness', 'habits'],
                'exclude': []
            },
            'business': {
                'primary': ['business', 'management', 'leadership', 'entrepreneur', 'startup', 'corporate', 'strategy'],
                'exclude': []
            },
            'philosophy': {
                'primary': ['philosophy', 'philosophical', 'ethics', 'existential', 'metaphysics'],
                'exclude': []
            },
            'poetry': {
                'primary': ['poetry', 'poems', 'verse', 'sonnet', 'haiku'],
                'exclude': []
            },
            'adventure': {
                'primary': ['adventure', 'quest', 'journey', 'exploration', 'expedition', 'voyage'],
                'exclude': []
            }
        }
        
        # Compile regex patterns for faster matching
        self.genre_patterns = {}
        for genre, keywords in self.genre_keywords.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords['primary'])
            self.genre_patterns[genre] = re.compile(pattern, re.IGNORECASE)

        print("Core models ready!\n")

    @property
    def zero_shot_classifier(self):
        """Lazy load genre classifier - ONLY when absolutely needed."""
        if self._zero_shot_classifier is None:
            print("⚠️ Loading heavy genre classifier (this should rarely happen)...")
            self._zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
        return self._zero_shot_classifier

    @property
    def emotion_classifier(self):
        """Lazy load emotion classifier."""
        if self._emotion_classifier is None:
            print("Loading emotion classifier...")
            self._emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1
            )
        return self._emotion_classifier

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching."""
        return hashlib.md5(text[:512].encode()).hexdigest()

    def build_index(self, books: List[Dict]):
        """
        OPTIMIZED: Fast index building with caching and larger batches.
        """
        if not books:
            return

        # Check if same books
        books_hash = hashlib.md5(str([b['title'] for b in books][:20]).encode()).hexdigest()
        if self._last_books_hash == books_hash:
            return  # Use cached index

        start_time = time.time()
        self.books = books
        descriptions = [f"{b['title']} {b.get('description', '')[:300]}" for b in books]

        # INTEGRATION: Use batch size from config
        self.embeddings = self.embedder.encode(
            descriptions,
            batch_size=self.config['embedding_batch_size'],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Build index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        self._last_books_hash = books_hash
        elapsed = time.time() - start_time
        print(f"⚡ Index built in {elapsed:.2f}s for {len(books)} books")

    def search_books(self, query: str, top_k: int = 10) -> List[Dict]:
        """Fast semantic search."""
        if self.index is None or not self.books:
            return []

        query_embedding = self.embedder.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.books))
        )

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.books):
                book = self.books[idx].copy()
                book['similarity_score'] = float(1 / (1 + distance))
                results.append(book)

        return results

    # INTEGRATION: Use constant from config for decorator
    @lru_cache(maxsize=GENRE_CACHE_SIZE)
    def _classify_genre_with_ai(self, text_hash: str, genre_filter: str, text_snippet: str) -> float:
        """
        CACHED AI classification - only called when keyword matching fails.
        """
        # INTEGRATION: Check config to skip AI
        if self.config.get('skip_ai_genre', True):
            return 0.0
            
        try:
            print(f"⚠️ Falling back to AI for genre '{genre_filter}' (cache miss)")
            result = self.zero_shot_classifier(
                text_snippet[:512],
                [genre_filter, "other"],
                multi_label=True
            )
            score = dict(zip(result['labels'], result['scores'])).get(genre_filter, 0)
            return score
        except Exception as e:
            print(f"❌ AI classification error: {e}")
            return 0.0

    def classify_genre_fast(self, book: Dict, genre_filter: str) -> float:
        """
        CRITICAL OPTIMIZATION: Ultra-fast genre matching with minimal AI usage.
        """
        genre_lower = genre_filter.lower()
        
        # OPTIMIZATION 1: Check categories first (instant - no processing)
        categories = book.get('categories', [])
        for cat in categories:
            cat_lower = cat.lower()
            if genre_lower in cat_lower:
                return 0.95  # Very high confidence
            # Check for partial matches
            if any(word in cat_lower for word in genre_lower.split()):
                return 0.85  # High confidence

        # OPTIMIZATION 2: Fast regex pattern matching
        genre_key = genre_lower.replace('-', ' ')
        if genre_key in self.genre_patterns:
            text_to_check = f"{book['title']} {book.get('description', '')[:400]}".lower()
            
            # Check for matches
            pattern = self.genre_patterns[genre_key]
            matches = pattern.findall(text_to_check)
            
            if matches:
                # Check for exclude keywords
                keywords_data = self.genre_keywords.get(genre_key, {})
                exclude_keywords = keywords_data.get('exclude', [])
                
                has_exclusion = any(excl in text_to_check for excl in exclude_keywords)
                
                if has_exclusion:
                    return 0.30  # Low confidence if exclusion found
                
                # Calculate confidence based on number of matches
                num_matches = len(matches)
                confidence = min(0.70 + (num_matches * 0.05), 0.90)
                return confidence

        # OPTIMIZATION 3: Fallback keyword matching (no regex)
        text_to_check = f"{book['title']} {book.get('description', '')[:400]}".lower()
        
        # Simple contains check
        if genre_lower in text_to_check:
            return 0.65  # Medium-high confidence
        
        # Check individual words from genre
        genre_words = genre_lower.split()
        word_matches = sum(1 for word in genre_words if len(word) > 3 and word in text_to_check)
        if word_matches > 0:
            confidence = 0.40 + (word_matches * 0.10)
            return min(confidence, 0.70)
        
        # OPTIMIZATION 4: Check if we need AI or just fail
        # INTEGRATION: Use config value if needed, but logic remains to default low
        return 0.10

    # INTEGRATION: Use constant from config for decorator
    @lru_cache(maxsize=EMOTION_CACHE_SIZE)
    def _classify_emotion_cached(self, text_hash: str, text_snippet: str) -> tuple:
        """Cached emotion classification."""
        try:
            result = self.emotion_classifier(text_snippet[:512])
            if isinstance(result, list) and len(result) > 0:
                # Return as tuple for caching (dict not hashable)
                items = tuple((item['label'], item['score']) for item in result[0])
                return items
        except Exception as e:
            print(f"⚠️ Emotion classification error: {e}")
        return ()

    def classify_emotion_fast(self, text: str) -> Dict:
        """Fast emotion classification with LRU caching."""
        if not text:
            return {}
            
        text_hash = self._get_text_hash(text)
        
        # Get cached result
        result_tuple = self._classify_emotion_cached(text_hash, text[:512])
        
        if result_tuple:
            return dict(result_tuple)
        return {}

    def filter_by_genre(self, books: List[Dict], genre_filter: str, threshold: Optional[float] = None) -> List[Dict]:
        """
        OPTIMIZED: Fast genre filtering with aggressive keyword matching.
        """
        # INTEGRATION: Use config value as default
        if threshold is None:
            threshold = self.config['genre_threshold']

        start_time = time.time()
        filtered = []
        ai_calls = 0
        keyword_matches = 0
        
        for book in books:
            score = self.classify_genre_fast(book, genre_filter)
            if score >= threshold:
                book['genre_score'] = score
                filtered.append(book)
                
                # Track how we matched (for debugging)
                if score >= 0.65:
                    keyword_matches += 1
                else:
                    ai_calls += 1
        
        elapsed = time.time() - start_time
        print(f"⚡ Genre filter: {len(filtered)}/{len(books)} matches in {elapsed:.2f}s "
              f"(keyword={keyword_matches}, ai={ai_calls})")
        
        # Sort by genre score for better results
        filtered.sort(key=lambda x: x.get('genre_score', 0), reverse=True)
        return filtered

    def filter_by_mood_batch(self, books: List[Dict], mood_filter: str, threshold: Optional[float] = None) -> List[Dict]:
        """
        OPTIMIZED: Batch emotion classification with better caching.
        """
        # INTEGRATION: Use config value as default
        if threshold is None:
            threshold = self.config['mood_threshold']

        start_time = time.time()
        mood_map = {
            "Joy": "joy", "Calm": "neutral", "Sadness": "sadness",
            "Excitement": "surprise", "Anger": "anger", 
            "Fear": "fear", "Surprise": "surprise"
        }
        
        emotion_key = mood_map.get(mood_filter, mood_filter.lower())
        filtered = []
        
        # Process all books with caching
        for book in books:
            text = book.get('description', '')[:512]
            if not text:
                continue
            
            emotion_scores = self.classify_emotion_fast(text)
            if emotion_scores.get(emotion_key, 0) >= threshold:
                book['mood_score'] = emotion_scores[emotion_key]
                filtered.append(book)
        
        elapsed = time.time() - start_time
        print(f"⚡ Mood filter: {len(filtered)}/{len(books)} matches in {elapsed:.2f}s")
        
        # Sort by mood score
        filtered.sort(key=lambda x: x.get('mood_score', 0), reverse=True)
        return filtered

    def filter_by_mood(self, books: List[Dict], mood_filter: str, threshold: Optional[float] = None) -> List[Dict]:
        """Wrapper for batch mood filtering."""
        return self.filter_by_mood_batch(books, mood_filter, threshold)

    def recommend_books(
        self,
        query: str,
        mood_filter: Optional[str] = None,
        genre_filter: Optional[str] = None,
        top_k: int = 10,
        timeout: float = 5.0
    ) -> List[Dict]:
        """
        OPTIMIZED: Get recommendations with timeout protection and smart filtering.
        """
        start_time = time.time()
        
        # INTEGRATION: Use search multiplier and cap from config
        multiplier = self.config['search_multiplier']
        max_candidates = self.config['max_candidates']
        
        search_k = min(top_k * multiplier, len(self.books), max_candidates)
        results = self.search_books(query, top_k=search_k)
        
        print(f"⚡ Semantic search: {len(results)} candidates in {time.time()-start_time:.2f}s")
        
        # OPTIMIZATION: Apply genre filter first (can be very fast with keywords)
        if genre_filter and genre_filter != "Any":
            if time.time() - start_time > timeout * 0.6:
                return results[:top_k]  # Return what we have
            # INTEGRATION: Explicitly calling without threshold uses the default which now pulls from config
            results = self.filter_by_genre(results, genre_filter)
        
        # OPTIMIZATION: Apply mood filter second
        if mood_filter and mood_filter != "Any":
            if time.time() - start_time > timeout * 0.85:
                return results[:top_k]  # Return what we have
            # INTEGRATION: Explicitly calling without threshold uses the default which now pulls from config
            results = self.filter_by_mood(results, mood_filter)
        
        # Sort by similarity score and return
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        elapsed = time.time() - start_time
        print(f"⚡ Total recommendation time: {elapsed:.2f}s")
        
        return results[:top_k]
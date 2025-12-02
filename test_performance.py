"""
Performance Testing & Profiling Script for LitSense
Run this to identify bottlenecks and measure optimization improvements.
"""

import time
import cProfile
import pstats
import io
from typing import Dict, List
import sys
# INTEGRATION: Import config
from config import get_config

# Add your project to path
sys.path.insert(0, '.')

from semantic_search import SemanticBookRecommender
from utils import fetch_books_progressive, explain_recommendations_batch


class PerformanceProfiler:
    """Profile and benchmark the recommendation system."""
    
    def __init__(self):
        self.recommender = SemanticBookRecommender()
        self.results = {}
        # INTEGRATION: Load config for assertions
        self.config = get_config()
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    
    def test_book_fetching(self, query: str = "python programming", max_results: int = 40):
        """Test book fetching performance."""
        print("\n" + "="*80)
        print("TEST 1: Book Fetching Performance")
        print("="*80)
        
        # Test with cache miss (first run)
        print(f"\n📥 Fetching {max_results} books for '{query}' (no cache)...")
        books, elapsed = self.time_function(fetch_books_progressive, query, max_results)
        print(f"✓ Fetched {len(books)} books in {elapsed:.2f}s")
        self.results['fetch_no_cache'] = elapsed
        
        # Test with cache hit (second run)
        print(f"\n📥 Fetching same query (with cache)...")
        books, elapsed = self.time_function(fetch_books_progressive, query, max_results)
        print(f"✓ Fetched {len(books)} books in {elapsed:.2f}s")
        self.results['fetch_with_cache'] = elapsed
        
        speedup = self.results['fetch_no_cache'] / self.results['fetch_with_cache'] if self.results['fetch_with_cache'] > 0 else 0
        print(f"\n⚡ Cache speedup: {speedup:.1f}x faster")
        
        return books
    
    def test_index_building(self, books: List[Dict]):
        """Test FAISS index building."""
        print("\n" + "="*80)
        print("TEST 2: Index Building Performance")
        print("="*80)
        
        print(f"\n🔨 Building index for {len(books)} books...")
        _, elapsed = self.time_function(self.recommender.build_index, books)
        print(f"✓ Index built in {elapsed:.2f}s")
        self.results['index_build'] = elapsed
        
        # Test cached index
        print(f"\n🔨 Rebuilding same index (should use cache)...")
        _, elapsed = self.time_function(self.recommender.build_index, books)
        print(f"✓ Index rebuild in {elapsed:.2f}s")
        self.results['index_rebuild'] = elapsed
    
    def test_semantic_search(self, query: str = "motivational books about success"):
        """Test semantic search performance."""
        print("\n" + "="*80)
        print("TEST 3: Semantic Search Performance")
        print("="*80)
        
        print(f"\n🔍 Searching for '{query}'...")
        results, elapsed = self.time_function(self.recommender.search_books, query, top_k=20)
        print(f"✓ Found {len(results)} results in {elapsed:.2f}s")
        self.results['semantic_search'] = elapsed
        
        return results
    
    def test_genre_filtering(self, books: List[Dict], genre: str = "Fiction"):
        """Test genre classification performance."""
        print("\n" + "="*80)
        print("TEST 4: Genre Filtering Performance")
        print("="*80)
        
        print(f"\n🎭 Filtering {len(books)} books for genre '{genre}'...")
        filtered, elapsed = self.time_function(
            self.recommender.filter_by_genre, 
            books, 
            genre
        )
        print(f"✓ Filtered to {len(filtered)} books in {elapsed:.2f}s")
        print(f"  Per-book: {elapsed/len(books)*1000:.1f}ms")
        self.results['genre_filter'] = elapsed
        
        # Test with cache
        print(f"\n🎭 Re-filtering same books (with cache)...")
        filtered, elapsed = self.time_function(
            self.recommender.filter_by_genre, 
            books, 
            genre
        )
        print(f"✓ Filtered to {len(filtered)} books in {elapsed:.2f}s")
        self.results['genre_filter_cached'] = elapsed
        
        return filtered
    
    def test_mood_filtering(self, books: List[Dict], mood: str = "Joy"):
        """Test emotion classification performance."""
        print("\n" + "="*80)
        print("TEST 5: Mood Filtering Performance")
        print("="*80)
        
        print(f"\n😊 Filtering {len(books)} books for mood '{mood}'...")
        filtered, elapsed = self.time_function(
            self.recommender.filter_by_mood, 
            books, 
            mood
        )
        print(f"✓ Filtered to {len(filtered)} books in {elapsed:.2f}s")
        print(f"  Per-book: {elapsed/len(books)*1000:.1f}ms")
        self.results['mood_filter'] = elapsed
        
        return filtered
    
    def test_explanations(self, books: List[Dict], query: str = "motivational books"):
        """Test AI explanation generation."""
        print("\n" + "="*80)
        print("TEST 6: AI Explanation Generation")
        print("="*80)
        
        # Test with 5 books
        test_books = books[:5]
        print(f"\n💬 Generating explanations for {len(test_books)} books...")
        # INTEGRATION: Use timeout relative to pipeline timeout
        timeout = self.config['pipeline_timeout'] * self.config['time_budget']['explain'] * 2 # Giving generous buffer for test
        explanations, elapsed = self.time_function(
            explain_recommendations_batch,
            test_books,
            query,
            timeout=max(timeout, 3.0) 
        )
        print(f"✓ Generated {len(explanations)} explanations in {elapsed:.2f}s")
        print(f"  Per-book: {elapsed/len(test_books)*1000:.0f}ms")
        self.results['explanations'] = elapsed
        
        # Show one example
        if explanations and explanations[0]:
            print(f"\n  Example: {explanations[0][:100]}...")
    
    def test_full_pipeline(self, query: str = "uplifting fantasy books", 
                          genre: str = "Fantasy", mood: str = "Joy",
                          num_results: int = 5):
        """Test complete end-to-end pipeline."""
        print("\n" + "="*80)
        print("TEST 7: Complete Pipeline (End-to-End)")
        print("="*80)
        
        print(f"\n🚀 Full pipeline test:")
        print(f"   Query: '{query}'")
        print(f"   Genre: {genre}, Mood: {mood}")
        print(f"   Results: {num_results}")
        
        start_time = time.time()
        
        # Step 1: Fetch books
        print("\n  [1/5] Fetching books...")
        books = fetch_books_progressive(query, max_results=40)
        step1_time = time.time() - start_time
        print(f"        ✓ {len(books)} books ({step1_time:.2f}s)")
        
        # Step 2: Build index
        print("  [2/5] Building search index...")
        self.recommender.build_index(books)
        step2_time = time.time() - start_time
        print(f"        ✓ Index ready ({step2_time - step1_time:.2f}s)")
        
        # Step 3: Search
        print("  [3/5] Semantic search...")
        results = self.recommender.search_books(query, top_k=20)
        step3_time = time.time() - start_time
        print(f"        ✓ {len(results)} matches ({step3_time - step2_time:.2f}s)")
        
        # Step 4: Filter
        print("  [4/5] Applying filters...")
        if genre != "Any":
            results = self.recommender.filter_by_genre(results, genre)
        if mood != "Any":
            results = self.recommender.filter_by_mood(results, mood)
        step4_time = time.time() - start_time
        print(f"        ✓ {len(results)} filtered ({step4_time - step3_time:.2f}s)")
        
        # Step 5: Explanations
        print("  [5/5] Generating explanations...")
        final_results = results[:num_results]
        
        # INTEGRATION: Use config-based budget for test
        timeout = self.config['pipeline_timeout'] * self.config['time_budget']['explain']
        
        explanations = explain_recommendations_batch(final_results, query, timeout=max(timeout, 2.0))
        step5_time = time.time() - start_time
        print(f"        ✓ {len(explanations)} explanations ({step5_time - step4_time:.2f}s)")
        
        total_time = time.time() - start_time
        self.results['full_pipeline'] = total_time
        
        print(f"\n  🎯 TOTAL TIME: {total_time:.2f}s")
        
        # INTEGRATION: Check against configured timeout
        target_time = self.config['pipeline_timeout']
        if total_time < target_time + 1.0: # Allow 1s buffer for test overhead
            print(f"  ✅ SUCCESS! Near {target_time}-second target")
        else:
            print(f"  ⚠️  EXCEEDS target by {total_time - target_time:.2f}s")
        
        # Breakdown
        print(f"\n  Time breakdown:")
        print(f"    Fetch:        {step1_time:.2f}s ({step1_time/total_time*100:.0f}%)")
        print(f"    Index:        {step2_time-step1_time:.2f}s ({(step2_time-step1_time)/total_time*100:.0f}%)")
        print(f"    Search:       {step3_time-step2_time:.2f}s ({(step3_time-step2_time)/total_time*100:.0f}%)")
        print(f"    Filter:       {step4_time-step3_time:.2f}s ({(step4_time-step3_time)/total_time*100:.0f}%)")
        print(f"    Explain:      {step5_time-step4_time:.2f}s ({(step5_time-step4_time)/total_time*100:.0f}%)")
    
    def profile_with_cprofile(self, query: str = "science fiction"):
        """Profile code with cProfile to find bottlenecks."""
        print("\n" + "="*80)
        print("TEST 8: Detailed cProfile Analysis")
        print("="*80)
        
        print("\n🔬 Running cProfile (this may take a moment)...")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run a typical workflow
        books = fetch_books_progressive(query, max_results=40)
        self.recommender.build_index(books)
        results = self.recommender.recommend_books(
            query=query,
            genre_filter="Science Fiction",
            mood_filter="Excitement",
            top_k=5
        )
        explain_recommendations_batch(results[:3], query, timeout=3.0)
        
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        
        print("\n📊 Top 30 functions by cumulative time:")
        print(s.getvalue())
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to display.")
            return
        
        print("\n⏱️  Individual Component Times:")
        for name, time_val in sorted(self.results.items()):
            status = "✅" if time_val < 2.0 else "⚠️" if time_val < 5.0 else "❌"
            print(f"  {status} {name:25s}: {time_val:6.2f}s")
        
        # Calculate estimated total
        key_components = ['fetch_no_cache', 'semantic_search', 'genre_filter', 
                         'mood_filter', 'explanations']
        estimated_total = sum(self.results.get(k, 0) for k in key_components)
        
        print(f"\n📈 Estimated Total (sum of components): {estimated_total:.2f}s")
        
        if 'full_pipeline' in self.results:
            print(f"🎯 Actual Full Pipeline: {self.results['full_pipeline']:.2f}s")
            
            # INTEGRATION: Check against configured timeout
            target = self.config['pipeline_timeout']
            if self.results['full_pipeline'] < target + 0.5:
                print("\n✅ PERFORMANCE TARGET MET!")
            else:
                print(f"\n⚠️  Need {self.results['full_pipeline'] - target:.2f}s improvement")
        
        print("\n💡 Recommendations:")
        if self.results.get('fetch_no_cache', 0) > 2.0:
            print("  • Optimize book fetching (use parallel requests)")
        if self.results.get('genre_filter', 0) > 1.0:
            print("  • Add keyword-based genre matching before AI classification")
        if self.results.get('mood_filter', 0) > 1.0:
            print("  • Batch emotion classification or add caching")
        if self.results.get('explanations', 0) / 5 > 0.4:
            print("  • Use parallel explanation generation with ThreadPoolExecutor")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("🚀 LitSense Performance Testing Suite")
    print("="*80)
    print("\nThis will run comprehensive performance tests.")
    print("Estimated time: 1-2 minutes\n")
    
    profiler = PerformanceProfiler()
    
    # Run tests
    try:
        # Basic component tests
        books = profiler.test_book_fetching()
        profiler.test_index_building(books)
        results = profiler.test_semantic_search()
        filtered_genre = profiler.test_genre_filtering(results[:10])
        filtered_mood = profiler.test_mood_filtering(results[:10])
        profiler.test_explanations(results[:5])
        
        # Full pipeline test
        profiler.test_full_pipeline()
        
        # Detailed profiling (optional - can be slow)
        print("\n" + "="*80)
        print("Run detailed cProfile analysis? (This may take 1-2 minutes)")
        response = input("Enter 'y' to continue, or press Enter to skip: ")
        if response.lower() == 'y':
            profiler.profile_with_cprofile()
        
        # Print summary
        profiler.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        profiler.print_summary()
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        profiler.print_summary()


if __name__ == "__main__":
    run_all_tests()
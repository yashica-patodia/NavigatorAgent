import os
# Set this at the very beginning of your script
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # Updated import
import tiktoken
from dotenv import load_dotenv

# Load environment variables (including OpenAI API key)
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Updated client initialization
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

class NavigatorAgent:
    def __init__(self, vectors_dir='vectors', data_dir='data'):
        self.vectors_dir = vectors_dir
        self.data_dir = data_dir
        
        # Load FAISS index
        self.index = faiss.read_index(f'{vectors_dir}/faiss_index.bin')
        
        # Load embeddings
        self.embeddings = np.load(f'{vectors_dir}/embeddings.npy')
        
        # Load chunks and mapping
        with open(f'{vectors_dir}/chunks.json', 'r') as f:
            self.chunks = json.load(f)
        
        with open(f'{vectors_dir}/chunk_map.json', 'r') as f:
            self.chunk_map = json.load(f)
        
        # Load original articles
        with open(f'{data_dir}/articles.json', 'r') as f:
            self.articles = json.load(f)
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def count_tokens(self, text, model="gpt-4o-mini"):
        """Count the number of tokens in the text for the given model"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximate counting if tiktoken fails
            return len(text) // 4  # Rough approximation
    
    def search(self, query, k=10):
        """Search for relevant content based on query with source diversification"""
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Get more results than needed to allow for diversification
        k_search = min(k * 3, len(self.chunks))  # Get 3x results but don't exceed total chunks
        distances, indices = self.index.search(query_embedding.astype('float32'), k_search)
        
        # Group results by source URL
        results_by_url = {}
        
        for i, idx in enumerate(indices[0]):
            chunk_text = self.chunks[idx]
            chunk_info = self.chunk_map[idx]
            url = chunk_info['url']
            
            result = {
                'chunk': chunk_text,
                'title': chunk_info['title'],
                'url': url,
                'is_title': chunk_info['is_title'],
                'relevance_score': 1.0 - (distances[0][i] / 10.0)  # Normalize distance to 0-1 relevance
            }
            
            # Group by URL
            if url not in results_by_url:
                results_by_url[url] = []
            results_by_url[url].append(result)
        
        # Select diverse results - take the best chunk from each source first
        # then take second-best from each source, etc. until we have k results
        diversified_results = []
        
        # Sort URLs by the highest relevance score of any chunk from that URL
        sorted_urls = sorted(
            results_by_url.keys(),
            key=lambda url: max(r['relevance_score'] for r in results_by_url[url]),
            reverse=True
        )
        
        # Take chunks in rounds to ensure diversity
        for round_num in range(max(len(results_by_url[url]) for url in sorted_urls)):
            for url in sorted_urls:
                if round_num < len(results_by_url[url]):
                    # Sort chunks within this URL by relevance
                    sorted_chunks = sorted(
                        results_by_url[url],
                        key=lambda r: r['relevance_score'],
                        reverse=True
                    )
                    # Add the chunk at the current round index
                    diversified_results.append(sorted_chunks[round_num])
                    
                    # Break if we've reached our target k
                    if len(diversified_results) >= k:
                        break
            
            # Break the outer loop if we have enough results
            if len(diversified_results) >= k:
                break
        
        # Trim to exactly k results if we have more
        return diversified_results[:k]
    
    def optimize_context(self, search_results, max_tokens=3000):
        """Optimize context to fit within token limits and deduplicate sources"""
        context = ""
        context_sources = []
        seen_urls = set()  # Track URLs we've already added
        
        tokens_used = 0
        for result in search_results:
            source_text = f"Source: {result['title']}\nURL: {result['url']}\nContent: {result['chunk']}\n\n"
            source_tokens = self.count_tokens(source_text)
            
            if tokens_used + source_tokens <= max_tokens:
                context += source_text
                
                # Only add to sources if we haven't seen this URL before
                if result['url'] not in seen_urls:
                    context_sources.append({
                        'title': result['title'],
                        'url': result['url']
                    })
                    seen_urls.add(result['url'])
                    
                tokens_used += source_tokens
            else:
                # If full context doesn't fit, try adding just the title and URL
                minimal_text = f"Source: {result['title']}\nURL: {result['url']}\n\n"
                minimal_tokens = self.count_tokens(minimal_text)
                
                if tokens_used + minimal_tokens <= max_tokens:
                    context += minimal_text
                    
                    # Only add to sources if we haven't seen this URL before
                    if result['url'] not in seen_urls:
                        context_sources.append({
                            'title': result['title'],
                            'url': result['url']
                        })
                        seen_urls.add(result['url'])
                        
                    tokens_used += minimal_tokens
        
        return context, context_sources
    
    def generate_response(self, query, context, sources):
        """Generate response using ChatGPT based on the context with improved citation instructions"""
        system_prompt = """You are a specialized AI assistant for Accel's SeedToScale platform, a premier knowledge hub for entrepreneurs and startup founders.

Your purpose is to provide relevant advice from SeedToScale's content library.

RULES:
1. ONLY use information provided in the context to answer questions
2. If the context doesn't contain enough information, acknowledge this limitation
3. Always cite specific articles when providing information (e.g., "According to [Article Title]...")
4. Use a conversational, helpful tone appropriate for entrepreneurial advice
5. Focus on actionable insights rather than generic advice
6. Organize complex responses with clear structure when needed
7. When suggesting resources, reference specific SeedToScale articles by name
8. Represent Accel's expertise in the venture capital and startup space accurately
9. IMPORTANT: When citing sources, mention each source only ONCE in your response, even if you reference information from it multiple times
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
                Please help answer this question based ONLY on the information provided in the context:
                
                USER QUESTION: {query}
                
                CONTEXT FROM SEEDTOSCALE:
                {context}
                
                Please provide a helpful response that directly answers the question using only the information from the context. 
                If the context doesn't have enough information to provide a complete answer, acknowledge this limitation.
                Cite sources when appropriate by mentioning the article title.
            """}
        ]
        
        try:
            # Updated API call for OpenAI >= 1.0.0
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-3.5-turbo for a more affordable option
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            return {
                'answer': response.choices[0].message.content,  # Updated response structure
                'sources': sources
            }
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                'answer': f"I encountered an error while generating a response. Please try again.",
                'sources': sources
            }
    
    def answer_question(self, query):
        """Answer questions based on the relevant content"""
        # Retrieve relevant content
        search_results = self.search(query, k=10)  # Get more results initially for better context
        
        # Optimize context to fit within token limits
        context, sources = self.optimize_context(search_results)
        
        # Generate answer using ChatGPT
        response = self.generate_response(query, context, sources)
        
        return response

if __name__ == "__main__":
    # Test the navigator
    navigator = NavigatorAgent()
    query = "What advice does SeedToScale provide for Series A fundraising?"
    result = navigator.answer_question(query)
    print(f"Query: {query}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for source in result['sources']:
        print(f"- {source['title']}: {source['url']}")
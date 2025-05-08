import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

class Vectorizer:
    def __init__(self, input_dir='data', output_dir='vectors'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use a more powerful model
    
    def load_articles(self):
        """Load articles from the JSON file"""
        try:
            with open(f'{self.input_dir}/articles.json', 'r') as f:
                articles = json.load(f)
            return articles
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            return []
    
    def chunk_text(self, text, max_length=512):
        """Split text into chunks of specified max length"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_embeddings(self):
        """Create embeddings for all articles"""
        articles = self.load_articles()
        if not articles:
            print("No articles found to vectorize")
            return
        
        all_chunks = []
        chunk_to_article_map = []
        
        # Process each article and create chunks
        for article_id, article in enumerate(tqdm(articles, desc="Chunking articles")):
            title = article['title']
            content = article['content']
            
            # Create chunks from the content
            chunks = self.chunk_text(content)
            
            # Add title as a separate chunk with higher importance
            all_chunks.append(title)
            chunk_to_article_map.append({
                'article_id': article_id,
                'chunk_id': 0,
                'is_title': True,
                'url': article['url'],
                'title': title
            })
            
            # Add content chunks
            for chunk_id, chunk in enumerate(chunks, 1):
                all_chunks.append(chunk)
                chunk_to_article_map.append({
                    'article_id': article_id,
                    'chunk_id': chunk_id,
                    'is_title': False,
                    'url': article['url'],
                    'title': title
                })
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Creating embeddings"):
            batch = all_chunks[i:i+batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(all_embeddings.astype('float32'))
        
        # Save FAISS index
        faiss.write_index(index, f'{self.output_dir}/faiss_index.bin')
        
        # Save embeddings, chunks and mapping
        np.save(f'{self.output_dir}/embeddings.npy', all_embeddings)
        
        with open(f'{self.output_dir}/chunk_map.json', 'w') as f:
            json.dump(chunk_to_article_map, f)
        
        with open(f'{self.output_dir}/chunks.json', 'w') as f:
            json.dump(all_chunks, f)
        
        print(f"Created and saved {len(all_chunks)} chunks with embeddings")
        return all_embeddings, all_chunks, chunk_to_article_map

if __name__ == "__main__":
    vectorizer = Vectorizer()
    vectorizer.create_embeddings()
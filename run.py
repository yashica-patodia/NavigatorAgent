import os
import argparse
from scraper import SeedToScaleScraper
from vectorizer import Vectorizer
from navigator import NavigatorAgent

def main():
    parser = argparse.ArgumentParser(description='SeedToScale Navigator Tool')
    parser.add_argument('--scrape', action='store_true', help='Scrape the SeedToScale website')
    parser.add_argument('--vectorize', action='store_true', help='Create vector embeddings')
    parser.add_argument('--query', type=str, help='Test a query')
    
    args = parser.parse_args()
    
    if args.scrape:
        print("Starting to scrape SeedToScale...")
        scraper = SeedToScaleScraper()
        scraper.scrape_all_content()
    
    if args.vectorize:
        print("Creating vector embeddings...")
        vectorizer = Vectorizer()
        vectorizer.create_embeddings()
    
    if args.query:
        print(f"Testing query: {args.query}")
        navigator = NavigatorAgent()
        result = navigator.answer_question(args.query)
        print("\nAnswer:")
        print(result['answer'])
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source['title']}: {source['url']}")

if __name__ == "__main__":
    main()
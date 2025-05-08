import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SeedToScaleScraper:
    def __init__(self, output_dir='data'):
        self.base_url = "https://www.seedtoscale.com"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
    
    def get_sitemap_urls(self):
        """Extract all URLs from the sitemap"""
        try:
            # First, try to get the sitemap
            sitemap_url = f"{self.base_url}/sitemap.xml"
            response = requests.get(sitemap_url)
            if response.status_code != 200:
                print(f"Failed to fetch sitemap: {response.status_code}")
                return []
            
            # Use lxml parser explicitly
            soup = BeautifulSoup(response.text, 'lxml-xml')
            urls = [loc.text for loc in soup.find_all('loc')]
            print(f"Found {len(urls)} URLs in sitemap")
            return urls
        except Exception as e:
            print(f"Error fetching sitemap: {str(e)}")
            return []
    
    def get_all_content_urls(self):
        """Get all content URLs from the website"""
        urls = self.get_sitemap_urls()
        if not urls:
            print("Fallback to manual URL discovery...")
            # Fallback to manually browsing the site sections
            sections = ['/blog', '/podcasts', '/resources', '/playbooks']
            all_urls = []
            
            driver = webdriver.Chrome(options=self.chrome_options)
            
            try:
                for section in sections:
                    section_url = f"{self.base_url}{section}"
                    print(f"Scraping section: {section_url}")
                    driver.get(section_url)
                    time.sleep(3)  # Allow page to load
                    
                    # Scroll a few times to load more content if it's lazy-loaded
                    for i in range(5):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(1)
                        print(f"Scroll {i+1}/5 completed")
                    
                    # Find all article links
                    links = driver.find_elements(By.CSS_SELECTOR, "a[href^='/']")
                    section_urls = [link.get_attribute('href') for link in links if link.get_attribute('href')]
                    
                    # Filter for content URLs
                    content_section_urls = [url for url in section_urls if any(s in url for s in ['/blog/', '/podcast/', '/resources/', '/playbook/'])]
                    
                    print(f"Found {len(content_section_urls)} content URLs in {section}")
                    all_urls.extend(content_section_urls)
                
                # Remove duplicates
                all_urls = list(set(all_urls))
                print(f"Total unique content URLs found: {len(all_urls)}")
                
                return all_urls
            finally:
                driver.quit()
        
        return urls
    
    def scrape_article(self, url):
        """Scrape content from a single article URL"""
        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            
            # Wait for content to load (max 10 seconds)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "h1"))
                )
            except:
                print(f"Timeout waiting for content to load: {url}")
                driver.quit()
                return None
            
            # Extract title
            try:
                title = driver.find_element(By.TAG_NAME, "h1").text.strip()
            except:
                title = "Unknown Title"
                print(f"Could not extract title for {url}")
            
            # Extract content - try different selectors
            content_element = None
            for selector in [".post-content", "article", ".main-content", ".content-container", "main", ".entry-content"]:
                try:
                    content_element = driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not content_element:
                print(f"Could not find content element for {url}")
                # Try getting any text from the page as a fallback
                try:
                    content = driver.find_element(By.TAG_NAME, "body").text
                except:
                    content = ""
                driver.quit()
                
                if not content:
                    return None
            else:
                content = content_element.text
            
            # Extract author if available
            author = ""
            for author_selector in [".author", ".byline", ".meta-author", "[rel='author']"]:
                try:
                    author_element = driver.find_element(By.CSS_SELECTOR, author_selector)
                    author = author_element.text.strip()
                    if author:
                        break
                except:
                    pass
            
            # Extract date if available
            date = ""
            for date_selector in ["time", ".date", ".meta-date", ".published"]:
                try:
                    date_element = driver.find_element(By.CSS_SELECTOR, date_selector)
                    date = date_element.text.strip() or date_element.get_attribute("datetime")
                    if date:
                        break
                except:
                    pass
            
            # Extract categories/tags if available
            categories = []
            for category_selector in [".category", ".tag", ".topic", ".post-categories"]:
                try:
                    category_elements = driver.find_elements(By.CSS_SELECTOR, category_selector)
                    if category_elements:
                        categories = [element.text.strip() for element in category_elements]
                        break
                except:
                    pass
            
            # Get HTML content
            html_content = driver.page_source
            
            driver.quit()
            
            result = {
                'url': url,
                'title': title,
                'content': content,
                'author': author,
                'date': date,
                'categories': categories,
                'html': html_content
            }
            
            print(f"Successfully scraped article: {title}")
            return result
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            if 'driver' in locals():
                driver.quit()
            return None
    
    def scrape_all_content(self):
        """Scrape all content from the website"""
        urls = self.get_all_content_urls()
        print(f"Starting to scrape {len(urls)} URLs")
        
        articles = []
        for i, url in enumerate(urls):
            print(f"Scraping {i+1}/{len(urls)}: {url}")
            article = self.scrape_article(url)
            if article:
                articles.append(article)
                # Save progress incrementally
                if (i+1) % 5 == 0 or (i+1) == len(urls):
                    self.save_data(articles)
            
            # Respect the website by not hammering it with requests
            time.sleep(2)
        
        return articles
    
    def save_data(self, articles):
        """Save the scraped data to files"""
        # Save as JSON
        with open(f'{self.output_dir}/articles.json', 'w') as f:
            json.dump(articles, f, indent=2)
        
        # Save as CSV (excluding HTML content which would make CSV unwieldy)
        df = pd.DataFrame([{k: v for k, v in article.items() if k != 'html'} for article in articles])
        df.to_csv(f'{self.output_dir}/articles.csv', index=False)
        
        print(f"Saved {len(articles)} articles to {self.output_dir}")

if __name__ == "__main__":
    scraper = SeedToScaleScraper()
    scraper.scrape_all_content()
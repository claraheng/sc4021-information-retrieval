import os
import json
import time
import random
import uuid
from datetime import datetime
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. TRUSTPILOT SCRAPER
# ---------------------------------------------------------
def scrape_trustpilot(page):
    print("\n--- Starting Trustpilot Scraper ---")
    companies = [
        "tesla.com", "byd.com", "polestar.com", "vinfastauto.com", # Manufacturers
        "electroverse.com", "shellrecharge.com", "chargepoint.com", # Networks
        "crackingenergy.com", "cord-ev.com", "evheroes.co.uk"      # Installers/Hardware
    ]
    reviews =[]
    
    for company in companies:
        for star in range(1, 6):
            url = f"https://www.trustpilot.com/review/{company}?stars={star}"
            page.goto(url)
            time.sleep(random.uniform(2, 4))
            
            soup = BeautifulSoup(page.content(), "html.parser")
            cards = soup.find_all("article", attrs={"data-service-review-card-paper": "true"})
            
            for card in cards:
                text_el = card.find("p", attrs={"data-service-review-text-typography": "true"})
                if not text_el: continue
                
                # Create a UNIQUE ID for every single record
                doc_id = f"tp_{uuid.uuid4().hex[:8]}" 
                
                reviews.append({
                    "doc_id": doc_id,
                    "platform": "Trustpilot",
                    "source_target": company,
                    "text": text_el.get_text(strip=True),
                    "stars": star
                })

    print(f"Extracted {len(reviews)} reviews from Trustpilot.")
    
    # Save isolated JSON
    with open(f"{OUTPUT_DIR}/trustpilot.json", "w") as f:
        json.dump(reviews, f, indent=4)
    return reviews

# ---------------------------------------------------------
# 2. SGCARMART
# ---------------------------------------------------------
# def scrape_sgcarmart(page):
#     print("\n--- Starting sgCarMart Scraper ---")
#     reviews =[]
    
#     urls =[
#         "https://www.sgcarmart.com/new-cars/info/21839/maxus-mifa-7-electric/reviews"
#     ]
    
#     for url in urls:
#         page.goto(url)
#         time.sleep(random.uniform(2, 4))
        
#         soup = BeautifulSoup(page.content(), "html.parser")
        
#         # Target the user reviews section
#         review_blocks = soup.find_all("div", class_="review_box") # Example class name
        
#         for block in review_blocks:
#             text_el = block.find("div", class_="review_text")
#             if text_el:
#                 reviews.append({
#                     "doc_id": f"sgcm_{uuid.uuid4().hex[:8]}",
#                     "platform": "sgCarMart",
#                     "source_target": "Maxus Mifa",
#                     "text": text_el.get_text(strip=True),
#                     "stars": None
#                 })
                
#     with open(f"{OUTPUT_DIR}/sgcarmart.json", "w") as f:
#         json.dump(reviews, f, indent=4)
#     return reviews

# ---------------------------------------------------------
# 3. TEAM-BHP SCRAPER
# ---------------------------------------------------------
def scrape_teambhp(page):
    print("\n--- Starting Team-BHP Scraper ---")
    reviews = []
    
    # --- Step 1: Discover Thread URLs ---
    unique_threads = set()
    # Scrape first 10 pages of the electric cars subforum
    for page_num in range(1, 11):
        if page_num == 1:
            forum_index_url = "https://www.team-bhp.com/forum/electric-cars/"
        else:
            forum_index_url = f"https://www.team-bhp.com/forum/electric-cars/index{page_num}.html"
            
        print(f"Fetching forum index page {page_num}: {forum_index_url}")
        page.goto(forum_index_url)
        time.sleep(random.uniform(4, 6)) # Allow time for CloudFlare and page load
        
        soup = BeautifulSoup(page.content(), "html.parser")
        links = soup.find_all("a", href=True)
        
        # Filter for relevant thread links in the electric-cars subforum
        thread_links = [l for l in links if "electric-cars/" in l["href"] and ".html" in l["href"]]
        
        for link in thread_links:
            href = link["href"]
            # Filter out index pages, specific pages of a thread, lastpost, etc.
            if "index" not in href.split("/")[-1] and "page" not in href and "lastpost" not in href and "newpost" not in href and "#post" not in href:
                href = href.split("?")[0]
                if not href.startswith("http"):
                    if href.startswith("/"):
                        href = "https://www.team-bhp.com" + href
                    else:
                        href = "https://www.team-bhp.com/forum/" + href
                unique_threads.add(href)
                
    # Convert to list and take top N threads to not take forever but get >500 records.
    # 50 threads * ~15 posts = ~750 records
    unique_threads = list(unique_threads)[:50]
    print(f"Found unique threads. Scraping {len(unique_threads)} threads...")
    
    # --- Step 2: Scrape Discovered Threads ---
    for url in unique_threads:
        print(f"Fetching Team-BHP Thread: {url}")
        page.goto(url)
        time.sleep(random.uniform(3, 5)) # Human delay
        
        soup = BeautifulSoup(page.content(), "html.parser")
        
        # Grab the thread title
        title_el = soup.find("title")
        title = title_el.get_text(strip=True).replace(" - Team-BHP", "") if title_el else "Team-BHP Discussion"
        
        # The main content for forum posts are div elements with id starting with post_message_
        posts = soup.find_all("div", attrs={"id": lambda x: x and str(x).startswith("post_message_")})
            
        for post in posts:
            text_content = post.get_text(separator=" ", strip=True)
            if text_content:
                reviews.append({
                    "doc_id": f"tbhp_{uuid.uuid4().hex[:8]}",
                    "platform": "Team-BHP",
                    "source_target": title, 
                    "text": text_content,
                    "stars": None # Forums don't have stars! We will auto-label this via ML later
                })
            
    output_file = f"{OUTPUT_DIR}/teambhp.json"
    existing_reviews = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_reviews = json.load(f)
            except json.JSONDecodeError:
                pass
                
    existing_reviews.extend(reviews)
    
    with open(output_file, "w") as f:
        json.dump(existing_reviews, f, indent=4)
        
    print(f"Extracted {len(reviews)} new discussions from Team-BHP. Total records now {len(existing_reviews)}.")
    return reviews

# ---------------------------------------------------------
# 4. EDMUNDS
# ---------------------------------------------------------
# def scrape_edmunds(page):
#     print("\n--- Starting Edmunds Scraper ---")
#     reviews =[]
    
#     # Pre-defined list of EVs for the year 2025
#     ev_models =[
#         {"make": "tesla", "model": "model-3"},
#         {"make": "tesla", "model": "model-y"},
#         {"make": "hyundai", "model": "ioniq-5"},
#         {"make": "kia", "model": "ev6"},
#         {"make": "ford", "model": "f-150-lightning"}
#     ]
#     year = "2025"
    
#     for car in ev_models:
#         url = f"https://www.edmunds.com/{car['make']}/{car['model']}/{year}/consumer-reviews/"
#         print(f"Fetching Edmunds: {url}")
        
#         try:
#             page.goto(url)
#             # Wait for review blocks to load. Edmunds uses 'review-item' or similar classes.
#             page.wait_for_selector("div.review-item", timeout=10000)
#         except Exception:
#             print(f"No reviews found for {car['make']} {car['model']} (or hit anti-bot). Skipping.")
#             continue
            
#         time.sleep(random.uniform(2, 5))
        
#         soup = BeautifulSoup(page.content(), "html.parser")
#         review_cards = soup.find_all("div", class_=lambda x: x and "review-item" in x)
        
#         for card in review_cards:
#             # Extract text
#             text_el = card.find("div", class_=lambda x: x and "review-text" in x)
#             if not text_el: continue
            
#             # Extract stars (Edmunds usually has a visually hidden text or span with the rating)
#             rating_el = card.find("span", class_=lambda x: x and "rating-stars" in x)
#             stars = 3 # Default neutral
#             if rating_el and rating_el.get_text():
#                 try:
#                     stars = int(rating_el.get_text()[0]) # e.g., "5 out of 5" -> 5
#                 except:
#                     pass
                    
#             reviews.append({
#                 "doc_id": f"edm_{uuid.uuid4().hex[:8]}",
#                 "platform": "Edmunds",
#                 "source_target": f"{car['make'].title()} {car['model'].title()}",
#                 "text": text_el.get_text(strip=True),
#                 "stars": stars
#             })
            
#     with open(f"{OUTPUT_DIR}/edmunds.json", "w") as f:
#         json.dump(reviews, f, indent=4)
        
#     print(f"Extracted {len(reviews)} reviews from Edmunds.")
#     return reviews

# ---------------------------------------------------------
# 5. CARS.COM
# ---------------------------------------------------------
# def scrape_carscom(page):
#     print("\n--- Starting Cars.com Scraper ---")
#     reviews =[]
    
#     ev_models =[
#         {"make": "kia", "model": "ev6"},
#         {"make": "chevrolet", "model": "blazer-ev"}
#     ]
#     year = "2025"
    
#     for car in ev_models:
#         url = f"https://www.cars.com/research/{car['make']}-{car['model']}-{year}/consumer-reviews/"
#         print(f"Fetching Cars.com: {url}")
        
#         try:
#             page.goto(url)
#             time.sleep(random.uniform(3, 6))
            
#             soup = BeautifulSoup(page.content(), "html.parser")
#             # Cars.com uses classes like 'review-card'
#             cards = soup.find_all("div", class_="review-card")
            
#             for card in cards:
#                 text_el = card.find("p", class_="review-body")
#                 if not text_el: continue
                
#                 reviews.append({
#                     "doc_id": f"cars_{uuid.uuid4().hex[:8]}",
#                     "platform": "Cars.com",
#                     "source_target": f"{car['make']} {car['model']}",
#                     "text": text_el.get_text(strip=True),
#                     "stars": None # Will parse later if available
#                 })
#         except Exception as e:
#             print(f"Failed to scrape {url}: {e}")
            
#     with open(f"{OUTPUT_DIR}/carscom.json", "w") as f:
#         json.dump(reviews, f, indent=4)
#     return reviews

# ---------------------------------------------------------
# 6. HARDWAREZONE
# ---------------------------------------------------------
# def scrape_hardwarezone(page):
#     print("\n--- Starting HardwareZone Scraper ---")
#     reviews =[]
    
#     tag_url = "https://www.hardwarezone.com.sg/tag/electric-car"
#     page.goto(tag_url)
#     time.sleep(random.uniform(2, 4))
    
#     soup = BeautifulSoup(page.content(), "html.parser")
    
#     # 1. Collect article links
#     links =[]
#     for a_tag in soup.find_all("a", href=True):
#         if "/review/" in a_tag['href'] or "/feature/" in a_tag['href']:
#             full_link = "https://www.hardwarezone.com.sg" + a_tag['href'] if not a_tag['href'].startswith("http") else a_tag['href']
#             if full_link not in links:
#                 links.append(full_link)
                
#     # Limit to 50 results as you suggested
#     links = links[:50]
    
#     # 2. Visit each link and extract text
#     for link in links:
#         page.goto(link)
#         time.sleep(random.uniform(2, 4))
#         article_soup = BeautifulSoup(page.content(), "html.parser")
        
#         # Extract main article body (HardwareZone usually uses an article tag or a specific content div)
#         content_div = article_soup.find("div", class_="article-content")
#         if content_div:
#             reviews.append({
#                 "doc_id": f"hwz_{uuid.uuid4().hex[:8]}",
#                 "platform": "HardwareZone",
#                 "source_target": "Editorial Review",
#                 "text": content_div.get_text(separator=" ", strip=True),
#                 "stars": None
#             })
            
#     with open(f"{OUTPUT_DIR}/hardwarezone.json", "w") as f:
#         json.dump(reviews, f, indent=4)
#     return reviews

# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        # Sequentially run scrapers
        # tp_data = scrape_trustpilot(page)
        # sg_data = scrape_sgcarmart(page)
        tbhp_data = scrape_teambhp(page)
        # hwz_data = scrape_hardwarezone(page)
        # ed_data = scrape_edmunds(page)
        # carscom_data = scrape_carscom(page)
        
        # Teammates' data (Twitter/Reddit) will also end up in the OUTPUT_DIR
        
        print(f"\nFinished scraping. Files saved to {OUTPUT_DIR}")
        browser.close()

if __name__ == "__main__":
    main()
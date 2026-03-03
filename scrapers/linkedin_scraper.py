# Fallback if scraped content from reddit and X APIs are insufficient
# LinkedIn API is not easy to scrape
import time
import uuid
import pysolr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. Connect to our Dockerized Solr instance
# 'opinion_engine' is the core we created in the docker-compose file
SOLR_URL = 'http://localhost:8983/solr/opinion_engine'
solr = pysolr.Solr(SOLR_URL, always_commit=True)

# 2. Set up Selenium Webdriver
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Uncomment this later to run in the background
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def login_to_linkedin(username, password):
    print("Logging into LinkedIn...")
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    
    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.ID, "password").send_keys(Keys.RETURN)
    time.sleep(5) # Wait for login to complete

def scrape_and_index_topic(topic):
    print(f"Searching for {topic}...")
    # Navigate to LinkedIn post search for the specific topic
    search_url = f"https://www.linkedin.com/search/results/content/?keywords={topic}"
    driver.get(search_url)
    time.sleep(5) # Wait for results to load
    
    # Scroll down to load a few posts
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    
    # Find post elements (LinkedIn classes change often, this is a common wrapper)
    # Note: LinkedIn changes their CSS classes frequently. You may need to inspect 
    # the page and update this class name.
    posts = driver.find_elements(By.CSS_SELECTOR, ".feed-shared-update-v2")
    
    documents_to_index =[]
    
    for post in posts:
        try:
            # Extract text (Update CSS selectors based on current LinkedIn layout)
            text_element = post.find_element(By.CSS_SELECTOR, ".feed-shared-update-v2__commentary")
            post_text = text_element.text
            
            # Create the document schema
            doc = {
                "id": f"linkedin_{uuid.uuid4()}", # Generate a unique ID
                "source": "LinkedIn",
                "topic": topic,
                "text": post_text,
                "date_scraped": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Default values for MVP - you can extract real values later
                "engagement_score": 1 
            }
            
            documents_to_index.append(doc)
            print(f"Scraped: {post_text[:50]}...")
            
        except Exception as e:
            # Skip ads or posts formatted differently
            continue

    # 3. Push data to Solr
    if documents_to_index:
        print(f"Indexing {len(documents_to_index)} documents into Solr...")
        solr.add(documents_to_index)
        print("Done!")
    else:
        print("No documents found. LinkedIn might have changed their CSS classes.")

# --- RUN THE MVP ---
if __name__ == "__main__":
    # IMPORTANT: Use a dummy LinkedIn account. LinkedIn bans accounts that scrape.
    LINKEDIN_USER = "your_dummy_email@gmail.com"
    LINKEDIN_PASS = "your_dummy_password"
    
    try:
        login_to_linkedin(LINKEDIN_USER, LINKEDIN_PASS)
        scrape_and_index_topic("Electric Vehicles")
    finally:
        driver.quit()
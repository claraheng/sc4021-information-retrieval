import asyncio
from twikit import Client
from datetime import datetime

async def scrape_x_to_solr(topic, limit=50):
    print(f"Connecting to X using twikit for topic: {topic}...")
    
    client = Client('en-US')
    
    # Login using a dummy X account
    await client.login(
        auth_info_1='your_dummy_username', 
        auth_info_2='your_dummy_email@gmail.com', 
        password='your_dummy_password'
    )
    
    # Save cookies so you don't have to login every time (highly recommended!)
    client.save_cookies('cookies.json')
    # client.load_cookies('cookies.json') # Use this on subsequent runs instead of login()

    documents =[]
    
    # Search for latest tweets containing the topic
    tweets = await client.search_tweet(topic, product='Latest', count=limit)
    
    for tweet in tweets:
        # Convert Twitter's date format to Solr's required format
        # Twikit usually returns standard string dates, we parse and reformat
        try:
            parsed_date = datetime.strptime(tweet.created_at, '%a %b %d %H:%M:%S %z %Y')
            solr_date = parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        except:
            solr_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        doc = {
            "id": f"x_{tweet.id}",
            "source": "X",
            "topic": topic,
            "text": tweet.text,
            "date_posted": solr_date,
            "engagement_score": tweet.favorite_count, # Likes
            "author": tweet.user.name
        }
        documents.append(doc)
        
    print(f"Extracted {len(documents)} posts from X.")
    return documents
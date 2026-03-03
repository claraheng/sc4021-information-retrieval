# Seems like Reddit API needs approval before use
# Until then, this is just a template
import praw
import time

def scrape_reddit_to_solr(topic, limit=50):
    print(f"Connecting to Reddit API for topic: {topic}...")
    
    # Initialize PRAW with your credentials
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="OpinionEngine_Bot_v1.0 (by u/your_username)"
    )
    
    documents =[]
    
    # Search all of Reddit for the topic
    for submission in reddit.subreddit("all").search(topic, limit=limit):
        # We combine title and post text to get the full context
        full_text = f"{submission.title} - {submission.selftext}"
        
        # Skip image-only posts with no text
        if not full_text.strip():
            continue
            
        doc = {
            "id": f"reddit_{submission.id}",
            "source": "Reddit",
            "topic": topic,
            "text": full_text,
            "date_posted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(submission.created_utc)),
            "engagement_score": submission.score, # Upvotes minus downvotes
            "author": str(submission.author) if submission.author else "Deleted"
        }
        documents.append(doc)
        
    print(f"Extracted {len(documents)} posts from Reddit.")
    return documents
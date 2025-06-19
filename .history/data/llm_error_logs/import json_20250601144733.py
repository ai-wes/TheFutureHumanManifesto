import json
from datetime import datetime
import re

# Get input data from n8n
inputs = _input.all()

# Initialize results and stats
parsed_trends = []
source_stats = {
    'google_trends': 0,
    'reddit': 0,
    'news_api': 0,
    'hacker_news': 0,
    'unknown': 0,
    'errors': 0
}

def convert_unix_time(unix_time):
    """Convert Unix timestamp to ISO format"""
    if unix_time:
        try:
            return datetime.fromtimestamp(unix_time).isoformat()
        except:
            return datetime.now().isoformat()
    return datetime.now().isoformat()

def generate_id(text1, text2=""):
    """Generate unique ID from text"""
    combined = str(text1) + str(text2)
    return f"id_{hash(combined) % 100000}"

def estimate_engagement_score(source, score, comments=0, additional_metrics=None):
    """Calculate unified engagement score across platforms"""
    base_score = int(score or 0)
    comment_boost = int(comments or 0) * 0.5
    
    # Source-specific multipliers
    multipliers = {
        'reddit': 1.0,
        'hacker_news': 2.0,  # HN points are more valuable
        'news_api': 0.5,     # News articles get base score
        'google_trends': 1.5  # Trending searches are valuable
    }
    
    multiplier = multipliers.get(source, 1.0)
    final_score = (base_score + comment_boost) * multiplier
    
    return int(final_score)

def detect_source_type(data):
    """Detect data source from structure"""
    # Check for Reddit structure
    if isinstance(data, dict):
        if 'kind' in data and data.get('kind') == 'Listing':
            return 'reddit'
        elif any(key in data for key in ['subreddit', 'subreddit_name_prefixed', 'ups', 'upvote_ratio']):
            return 'reddit'
        elif any(key in data for key in ['story_id', 'hn_url']) and 'score' in data:
            return 'hacker_news'
        elif any(key in data for key in ['articles', 'totalResults', 'publishedAt']):
            return 'news_api'
        elif 'source' in data and data['source'].get('name'):
            return 'news_api'
        elif any(key in data for key in ['trend_breakdown', 'trendingSearches']):
            return 'google_trends'
    
    return 'unknown'

def parse_reddit_data(data):
    """Parse Reddit API response"""
    posts = []
    
    # Handle Reddit Listing format
    if data.get('kind') == 'Listing' and 'data' in data:
        children = data['data'].get('children', [])
        for child in children:
            if child.get('kind') == 't3' and 'data' in child:
                post_data = child['data']
                posts.append(parse_single_reddit_post(post_data))
    
    # Handle direct Reddit post data
    elif 'subreddit' in data or 'ups' in data:
        posts.append(parse_single_reddit_post(data))
    
    return posts

def parse_single_reddit_post(post_data):
    """Parse individual Reddit post"""
    return {
        'source': 'reddit',
        'id': post_data.get('id', generate_id(post_data.get('title', ''))),
        'title': post_data.get('title', 'No title'),
        'url': post_data.get('url') or post_data.get('url_overridden_by_dest'),
        'discussion_url': f"https://reddit.com{post_data.get('permalink', '')}",
        'score': int(post_data.get('ups', 0)) or int(post_data.get('score', 0)),
        'author': post_data.get('author'),
        'comments_count': int(post_data.get('num_comments', 0)),
        'time_posted': convert_unix_time(post_data.get('created_utc')),
        'content_type': 'social_post',
        'engagement_score': estimate_engagement_score(
            'reddit', 
            post_data.get('ups', 0), 
            post_data.get('num_comments', 0)
        ),
        'platform_specific': {
            'subreddit': post_data.get('subreddit_name_prefixed') or post_data.get('subreddit'),
            'upvote_ratio': float(post_data.get('upvote_ratio', 0)),
            'awards': int(post_data.get('total_awards_received', 0)),
            'is_video': post_data.get('is_video', False),
            'thumbnail': post_data.get('thumbnail'),
            'domain': post_data.get('domain')
        }
    }

def parse_hacker_news_data(data):
    """Parse Hacker News data (already well structured)"""
    return {
        'source': 'hacker_news',
        'id': str(data.get('story_id') or data.get('id')),
        'title': data.get('title', 'No title'),
        'url': data.get('external_url') or data.get('url'),
        'discussion_url': data.get('hn_url') or f"https://news.ycombinator.com/item?id={data.get('id', '')}",
        'score': int(data.get('score', 0)),
        'author': data.get('author') or data.get('by'),
        'comments_count': int(data.get('comments_count', 0)) or int(data.get('descendants', 0)),
        'time_posted': data.get('time_posted') or convert_unix_time(data.get('time')),
        'content_type': 'tech_story',
        'engagement_score': estimate_engagement_score(
            'hacker_news',
            data.get('score', 0),
            data.get('comments_count', 0)
        ),
        'platform_specific': {
            'story_type': data.get('story_type', 'story'),
            'comment_ids': data.get('comment_ids', []) or data.get('kids', [])
        }
    }

def parse_news_api_data(data):
    """Parse News API data"""
    articles = []
    
    # Handle News API response with articles array
    if 'articles' in data:
        for article in data['articles']:
            articles.append(parse_single_news_article(article))
    
    # Handle single article
    elif 'title' in data and ('url' in data or 'source' in data):
        articles.append(parse_single_news_article(data))
    
    return articles

def parse_single_news_article(article):
    """Parse individual news article"""
    return {
        'source': 'news_api',
        'id': generate_id(article.get('url', ''), article.get('title', '')),
        'title': article.get('title', 'No title'),
        'url': article.get('url'),
        'discussion_url': article.get('url'),
        'score': estimate_news_score(article),
        'author': article.get('author'),
        'comments_count': 0,  # News API doesn't provide comments
        'time_posted': article.get('publishedAt', datetime.now().isoformat()),
        'content_type': 'news_article',
        'engagement_score': estimate_news_score(article),
        'platform_specific': {
            'description': article.get('description'),
            'source_name': article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else str(article.get('source', '')),
            'source_id': article.get('source', {}).get('id') if isinstance(article.get('source'), dict) else None,
            'image_url': article.get('urlToImage'),
            'content_preview': (article.get('content', '') or '')[:200] + '...' if article.get('content') else None
        }
    }

def estimate_news_score(article):
    """Estimate engagement score for news articles"""
    score = 50  # Base score
    
    # Boost for major news sources
    source_name = ''
    if isinstance(article.get('source'), dict):
        source_name = article.get('source', {}).get('name', '').lower()
    else:
        source_name = str(article.get('source', '')).lower()
    
    major_sources = ['reuters', 'ap', 'bbc', 'wsj', 'nyt', 'cnn', 'fox', 'bloomberg']
    if any(source in source_name for source in major_sources):
        score += 30
    
    # Boost for recent articles
    if article.get('publishedAt'):
        try:
            pub_time = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
            hours_ago = (datetime.now() - pub_time.replace(tzinfo=None)).total_seconds() / 3600
            if hours_ago < 6:
                score += 25
            elif hours_ago < 24:
                score += 15
        except:
            pass
    
    return score

def parse_google_trends_data(data):
    """Parse Google Trends data"""
    trends = []
    
    # Handle trend_breakdown format
    if 'trend_breakdown' in data and isinstance(data['trend_breakdown'], list):
        for i, trend in enumerate(data['trend_breakdown']):
            trends.append({
                'source': 'google_trends',
                'id': f"trend_{generate_id(str(trend))}",
                'title': str(trend),
                'url': f"https://trends.google.com/trends/explore?q={str(trend).replace(' ', '%20')}",
                'discussion_url': f"https://trends.google.com/trends/explore?q={str(trend).replace(' ', '%20')}",
                'score': max(100 - (i * 2), 10),  # Position-based scoring
                'author': 'Google Trends',
                'comments_count': 0,
                'time_posted': datetime.now().isoformat(),
                'content_type': 'search_trend',
                'engagement_score': estimate_engagement_score('google_trends', 100 - (i * 2), 0),
                'platform_specific': {
                    'position': i + 1,
                    'trend_type': 'trending_search',
                    'region': data.get('geo', 'US')
                }
            })
    
    # Handle other Google Trends formats
    elif 'trendingSearches' in data:
        for trend_item in data['trendingSearches']:
            query = trend_item.get('title', {}).get('query', 'Unknown trend') if isinstance(trend_item.get('title'), dict) else str(trend_item.get('title', 'Unknown'))
            traffic = trend_item.get('formattedTraffic', '1K+')
            
            trends.append({
                'source': 'google_trends',
                'id': f"trend_{generate_id(query)}",
                'title': query,
                'url': f"https://trends.google.com/trends/explore?q={query.replace(' ', '%20')}",
                'discussion_url': f"https://trends.google.com/trends/explore?q={query.replace(' ', '%20')}",
                'score': parse_traffic_number(traffic),
                'author': 'Google Trends',
                'comments_count': 0,
                'time_posted': datetime.now().isoformat(),
                'content_type': 'search_trend',
                'engagement_score': estimate_engagement_score('google_trends', parse_traffic_number(traffic), 0),
                'platform_specific': {
                    'traffic': traffic,
                    'articles': trend_item.get('articles', [])[:3],
                    'related_queries': trend_item.get('relatedQueries', [])[:3]
                }
            })
    
    return trends

def parse_traffic_number(traffic_str):
    """Parse traffic strings like '1K+', '50K+' to numbers"""
    if not traffic_str:
        return 50
    
    # Extract number and multiplier
    match = re.search(r'(\d+)([KMB]?)', str(traffic_str).upper())
    if match:
        num, multiplier = match.groups()
        num = int(num)
        
        multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        return (num * multipliers.get(multiplier, 1)) // 1000  # Scale down for scoring
    
    return 50



def main():
    # Main processing logic
    for item in inputs:
        try:
            # Extract data (handle n8n json wrapper)
            data = item.get('json', item)
            print(data)
            # Skip empty data
            if not data:
                continue
            
            # Detect source and parse accordingly
            source_type = detect_source_type(data)
            
            if source_type == 'reddit':
                parsed_items = parse_reddit_data(data)
                print("Reddit Data", parsed_items)
                if isinstance(parsed_items, list):
                    parsed_trends.extend(parsed_items)
                    source_stats['reddit'] += len(parsed_items)
                else:
                    parsed_trends.append(parsed_items)
                    source_stats['reddit'] += 1
            
            elif source_type == 'hacker_news':
                parsed_item = parse_hacker_news_data(data)
                print("Hacker News Data", parsed_item)
                parsed_trends.append(parsed_item)
                source_stats['hacker_news'] += 1
            
            elif source_type == 'news_api':
                parsed_items = parse_news_api_data(data)
                print("News Api Data", parsed_item)

                if isinstance(parsed_items, list):
                    parsed_trends.extend(parsed_items)
                    source_stats['news_api'] += len(parsed_items)
                else:
                    parsed_trends.append(parsed_items)
                    source_stats['news_api'] += 1
            
            elif source_type == 'google_trends':
                parsed_items = parse_google_trends_data(data)
                print("Google Data", parsed_items)

                if isinstance(parsed_items, list):
                    parsed_trends.extend(parsed_items)
                    source_stats['google_trends'] += len(parsed_items)
                else:
                    parsed_trends.append(parsed_items)
                    source_stats['google_trends'] += 1
            
            else:
                parsed_item = {'source': 'unknown',
                    'id': generate_id(str(data)),
                    'title': str(data.get('title', data.get('query', str(data)[:50]))),
                    'url': data.get('url'),
                    'discussion_url': data.get('url'),
                    'score': int(data.get('score', 0)),
                    'author': data.get('author'),
                    'comments_count': int(data.get('comments_count', 0)),
                    'time_posted': datetime.now().isoformat(),
                    'content_type': 'unknown',
                    'engagement_score': int(data.get('score', 0)),
                    'platform_specific': data
                }
                parsed_trends.append(parsed_item)
                source_stats['unknown'] += 1
        
        except Exception as e:
            # Add error item but continue processing
            error_item = {
                'source': 'error',
                'id': f"error_{len(parsed_trends)}",
                'title': f'Error processing item: {str(e)[:100]}',
                'url': None,
                'discussion_url': None,
                'score': 0,
                'author': 'System',
                'comments_count': 0,
                'time_posted': datetime.now().isoformat(),
                'content_type': 'error',
                'engagement_score': 0,
                'platform_specific': {'error': str(e), 'original_data': str(item)[:200]}
            }
            parsed_trends.append(error_item)
            source_stats['errors'] += 1

    # Sort by engagement score and take top trends
    parsed_trends.sort(key=lambda x: x.get('engagement_score', 0), reverse=True)
    top_trends = parsed_trends[:25]  # Top 25 trends

    # Add metadata to each trend
    total_processed = len(parsed_trends)
    processing_time = datetime.now().isoformat()

    for trend in top_trends:
        trend.update({
            'total_items_processed': total_processed,
            'source_breakdown': source_stats,
            'processed_at': processing_time,
            'rank': top_trends.index(trend) + 1
        })

    # Return in n8n expected format
    result = []
    for trend in top_trends:
        result.append({"json": trend})

    # If no trends found, return summary item
    if not result:
        result = [{"json": {
            "message": "No trends processed",
            "total_inputs": len(inputs),
            "source_stats": source_stats,
            "processing_errors": source_stats['errors'],
            "processed_at": datetime.now().isoformat()
        }}]

    return result

if __name__ == "__main__":
    main()

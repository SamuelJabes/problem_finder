#!/usr/bin/env python3
"""
Reddit Pain Point Analyzer - A tool to discover business opportunities through pain point analysis
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
from collections import defaultdict
import typer
from rich.console import Console
from rich.progress import Progress, TaskID, track
from rich.table import Table
from rich import print as rprint
import httpx
from parsel import Selector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from schemas.classify_schema import ClassifySchema
from schemas.redditPost_schema import RedditPost
from schemas.redditComment_schema import RedditComment
from schemas.painPoint_schema import PainPoint
from schemas.clusterAnalyzer_schema import ClusterAnalysisSchema

# Load environment variables
load_dotenv()

app = typer.Typer(no_args_is_help=True)
console = Console()

# Configuração de diretórios (ADICIONAR ISTO!)
DATA_DIR = Path("data")
SUBREDDITS_DIR = DATA_DIR / "subreddits"
COMMENTS_DIR = DATA_DIR / "comments"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLUSTERS_DIR = DATA_DIR / "clusters"

# Configuration
SUBREDDITS = [
    "Entrepreneur",
    "SaaS", 
    "NoStupidQuestions",
    "personalfinance",
    "smallbusiness",
    "socialmedia",
    "askatherapist",
    "productivity",
    "Accounting"
]

DATA_DIR = Path("data")
COMMENTS_DIR = DATA_DIR / "comments"

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=3072,
            api_key=api_key
        )
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        # Filter out None and empty texts
        clean_texts = [t for t in texts if t and t.strip()]
        
        console.print(f"📊 Generating embeddings for {len(clean_texts)} texts...")
        
        try:
            # Generate embeddings in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in track(range(0, len(clean_texts), batch_size), description="Generating embeddings..."):
                batch = clean_texts[i:i + batch_size]
                batch_embeddings = await asyncio.to_thread(
                    self.embeddings.embed_documents,
                    batch
                )
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

class ClusterAnalyzer:
    def __init__(self, api_key: str):
        self.embeddings_generator = EmbeddingGenerator(api_key)
        self.llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            api_key=api_key
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a business analyst expert in identifying market opportunities from customer pain points.
               
               Your task is to analyze a cluster of similar pain points/problems and provide business insights.
               
               For each cluster, provide:
               - main_theme: The central theme/problem of this cluster
               - common_problems: List of specific common problems mentioned
               - business_opportunity: A concrete business opportunity to address these problems
               - target_audience: Who would be the target customers
               - urgency: How urgent are these problems (1-10, where 10 = extremely urgent)
               - market_size: Estimated market size (small/medium/large)
               - solution_complexity: How complex would the solution be (simple/medium/complex)
            """),
            ("human", """
               Analyze this cluster of pain points:
               
               Pain Points:
               {pain_points}
               
               Provide a comprehensive business analysis.
            """)
        ])
        
        self.analysis_chain = self.analysis_prompt | self.llm.with_structured_output(ClusterAnalysisSchema)
    
    async def perform_clustering(self, embeddings: List[List[float]], n_clusters: int = 7) -> np.ndarray:
        """Perform clustering on embeddings"""
        console.print(f"🔍 Performing clustering with {n_clusters} clusters...")
        
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_array)
        
        console.print(f"✅ Clustering completed! Found {len(set(labels))} clusters")
        return labels
    
    async def analyze_cluster(self, pain_points: List[str]) -> ClusterAnalysisSchema:
        """Analyze a cluster of pain points with LLM"""
        try:
            # Limit to first 20 pain points to avoid token limits
            sample_points = pain_points[:20]
            pain_points_text = "\n".join([f"- {point}" for point in sample_points])
            
            if len(pain_points) > 20:
                pain_points_text += f"\n... and {len(pain_points) - 20} more similar pain points"
            
            analysis = await asyncio.to_thread(
                self.analysis_chain.invoke,
                {"pain_points": pain_points_text}
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            # Return a fallback analysis
            return ClusterAnalysisSchema(
                main_theme="Analysis unavailable",
                common_problems=["Various problems"],
                business_opportunity="Opportunity needs further analysis",
                target_audience="To be determined",
                urgency=5,
                market_size="medium",
                solution_complexity="medium"
            )

class RedditScraper:
    def __init__(self):
        self.client = httpx.AsyncClient(
            http2=True,
            headers={
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Cookie": "intl_splash=false"
            },
            follow_redirects=True
        )

    def parse_subreddit(self, response: httpx.Response) -> Dict:
        """Parse subreddit data from HTML"""
        selector = Selector(response.text)
        url = str(response.url)
        
        info = {
            "id": url.split("/r")[-1].replace("/", ""),
            "description": selector.xpath("//shreddit-subreddit-header/@description").get(),
            "url": url
        }
        
        post_data = []
        for box in selector.xpath("//article"):
            link = box.xpath(".//a/@href").get()
            author = box.xpath(".//shreddit-post/@author").get()
            title = box.xpath("./@aria-label").get()
            upvotes = box.xpath(".//shreddit-post/@score").get()
            comment_count = box.xpath(".//shreddit-post/@comment-count").get()
            
            post_data.append({
                "title": title,
                "link": "https://www.reddit.com" + link if link else None,
                "author": author,
                "upvotes": int(upvotes) if upvotes else None,
                "comment_count": int(comment_count) if comment_count else None,
                "publish_date": box.xpath(".//shreddit-post/@created-timestamp").get(),
            })
        
        cursor_id = selector.xpath("//shreddit-post/@more-posts-cursor").get()
        return {"post_data": post_data, "info": info, "cursor": cursor_id}

    async def scrape_subreddit(self, subreddit_id: str, max_pages: int = 2) -> Dict:
        """Scrape posts from a subreddit"""
        base_url = f"https://www.reddit.com/r/{subreddit_id}/"
        
        try:
            response = await self.client.get(base_url)
            data = self.parse_subreddit(response)
            subreddit_data = {
                "info": data["info"],
                "posts": data["post_data"]
            }
            
            cursor = data["cursor"]
            pages_scraped = 1
            
            while cursor and pages_scraped < max_pages:
                pagination_url = f"https://www.reddit.com/svc/shreddit/community-more-posts/hot/?after={cursor}%3D%3D&t=DAY&name={subreddit_id}&feedLength=3&sort=new"
                response = await self.client.get(pagination_url)
                data = self.parse_subreddit(response)
                cursor = data["cursor"]
                subreddit_data["posts"].extend(data["post_data"])
                pages_scraped += 1
                
            logger.success(f"Scraped {len(subreddit_data['posts'])} posts from r/{subreddit_id}")
            return subreddit_data
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_id}: {e}")
            return {"info": {"id": subreddit_id}, "posts": []}

    def parse_post_comments(self, response: httpx.Response) -> List[Dict]:
        """Parse comments from a Reddit post"""
        def parse_comment(parent_selector) -> Dict:
            author = parent_selector.xpath("./@data-author").get()
            comment_body = parent_selector.xpath(".//div[@class='md']/p/text()").get()
            upvotes = parent_selector.xpath(".//span[contains(@class, 'likes')]/@title").get()
            
            return {
                "author": author,
                "body": comment_body,
                "upvotes": int(upvotes) if upvotes else None,
                "publish_date": parent_selector.xpath(".//time/@datetime").get(),
            }

        selector = Selector(response.text)
        comments = []
        
        for item in selector.xpath("//div[@class='sitetable nestedlisting']/div[@data-type='comment']"):
            comment_data = parse_comment(item)
            if comment_data["body"]:  # Only add comments with actual content
                comments.append(comment_data)
                
        return comments

    async def scrape_post_comments(self, post_url: str, max_comments: int = 100) -> List[Dict]:
        """Scrape comments from a specific post"""
        try:
            # Convert to old.reddit.com for easier parsing
            old_reddit_url = post_url.replace("www.reddit.com", "old.reddit.com") + f"?limit={max_comments}"
            response = await self.client.get(old_reddit_url)
            comments = self.parse_post_comments(response)
            logger.info(f"Scraped {len(comments)} comments from {post_url}")
            return comments
        except Exception as e:
            logger.error(f"Error scraping comments from {post_url}: {e}")
            return []

    async def close(self):
        await self.client.aclose()

class PainPointClassifier:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional analyst of Reddit comments and posts specialized in identifying business opportunities through pain point analysis.
               
               Your task is to analyze comments/posts and identify if they express any type of pain, frustration, or problem that could represent a business opportunity.
               
               Classification criteria:
               - has_problem: 'YES' if text expresses a clear pain/problem/frustration, 'NO' otherwise
               - confidence: Rate your confidence from 1-10 (10 = very confident in classification)
               - category: Classify the problem type (business, financial, technical, personal, professional, health, productivity, marketing, customer_service, legal, unknown)
               - intensity: Rate the pain intensity from 1-10 (10 = extreme frustration/urgent problem, 1 = minor annoyance)
               
               Focus on problems that could lead to business solutions, not just personal complaints.
            """),
            ("human", """
               Given this Reddit comment/post:
               
               Text: "{text}"
               
               Analyze if this reveals a user problem/pain point with business opportunity potential.
            """)
        ])
        
        self.chain = self.prompt | self.llm.with_structured_output(ClassifySchema)

    async def classify_text(self, text: str) -> ClassifySchema:
        """Classify a single text for pain points"""
        try:
            result = await asyncio.to_thread(
                self.chain.invoke, 
                {"text": text}
            )
            return result
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return ClassifySchema(has_problem="NO", confidence=1, category="unknown")

    async def classify_batch(self, texts: List[str], progress_callback=None) -> List[Dict]:
        """Classify multiple texts for pain points"""
        results = []
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 10:  # Skip very short texts
                continue
                
            classification = await self.classify_text(text)
            
            if classification.has_problem == "YES":
                results.append({
                    "text": text,
                    "has_problem": classification.has_problem,
                    "confidence": classification.confidence,
                    "category": classification.category
                })
                
            if progress_callback:
                progress_callback(i + 1, len(texts))
                
        return results

def setup_directories():
    """Create necessary directories"""
    DATA_DIR.mkdir(exist_ok=True)
    SUBREDDITS_DIR.mkdir(exist_ok=True)
    COMMENTS_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    CLUSTERS_DIR.mkdir(exist_ok=True)
    for subreddit in SUBREDDITS:
        (SUBREDDITS_DIR / subreddit).mkdir(exist_ok=True)

def save_json(data: Dict, filepath: Path):
    """Save data to JSON file"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: Path) -> Dict:
    """Load data from JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@app.command("scrape")
def scrape_reddit(
    subreddits: Optional[str] = typer.Option(None, help="Comma-separated list of subreddits"),
    max_pages: int = typer.Option(2, help="Maximum pages to scrape per subreddit"),
    max_posts: int = typer.Option(5, help="Maximum posts to get comments from"),
    include_comments: bool = typer.Option(True, help="Whether to scrape comments from posts")
):
    """
    Scrape Reddit data from specified subreddits
    """
    setup_directories()
    target_subreddits = subreddits.split(",") if subreddits else SUBREDDITS
    
    async def run_scraper():
        all_texts = await run_scraper_logic(target_subreddits, max_pages, max_posts, include_comments)
        
        console.print(f"✅ Scraping completed!")
        console.print(f"📊 Total texts collected: {len(all_texts)}")
        console.print(f"💾 Data saved to: {DATA_DIR}")
        
    asyncio.run(run_scraper())

@app.command("classify")
def classify_pain_points(
    input_file: str = typer.Option("data/comments/all_texts.json", help="Input JSON file with texts"),
    output_file: str = typer.Option("data/comments/pain_points.json", help="Output file for classified pain points"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
):
    """
    Classify texts to identify pain points using LLM
    """
    
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    # Load texts
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"❌ Input file not found: {input_file}")
        raise typer.Exit(1)
    
    texts = load_json(input_path)
    console.print(f"📄 Loaded {len(texts)} texts for classification")
    
    async def run_classifier():
        pain_points = await run_classifier_logic(texts, openai_key)
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(pain_points, output_path)
        
        # Show summary
        console.print(f"✅ Classification completed!")
        console.print(f"🎯 Pain points found: {len(pain_points)}")
        console.print(f"📊 Conversion rate: {len(pain_points)/len(texts)*100:.1f}%")
        console.print(f"💾 Results saved to: {output_path}")
        
        # Show top categories
        categories = {}
        intensities = []
        
        for pp in pain_points:
            cat = pp.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            if pp.get("intensity"):
                intensities.append(pp["intensity"])
        
        if categories:
            table = Table(title="Top Pain Point Categories")
            table.add_column("Category", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Avg Intensity", style="yellow")
            
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                # Calculate average intensity for this category
                cat_intensities = [pp.get("intensity", 0) for pp in pain_points if pp.get("category") == cat and pp.get("intensity")]
                avg_intensity = sum(cat_intensities) / len(cat_intensities) if cat_intensities else 0
                
                table.add_row(cat, str(count), f"{avg_intensity:.1f}")
                
            console.print(table)
            
        # Show intensity distribution
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            high_intensity = len([i for i in intensities if i >= 7])
            console.print(f"\n📊 Pain Intensity Analysis:")
            console.print(f"   Average intensity: {avg_intensity:.1f}/10")
            console.print(f"   High intensity problems (7+): {high_intensity} ({high_intensity/len(intensities)*100:.1f}%)")
    
    asyncio.run(run_classifier())

async def run_scraper_logic(target_subreddits: List[str], max_pages: int, max_posts: int, include_comments: bool = True):
    """Internal function to run scraper logic"""
    scraper = RedditScraper()
    all_texts = []
    
    with Progress() as progress:
        task = progress.add_task("[green]Scraping subreddits...", total=len(target_subreddits))
        
        for subreddit in target_subreddits:
            progress.update(task, description=f"[green]Scraping r/{subreddit}...")
            
            # Scrape subreddit posts
            subreddit_data = await scraper.scrape_subreddit(subreddit, max_pages)
            
            # Save subreddit data in subreddits folder
            subreddit_path = SUBREDDITS_DIR / subreddit / "subreddit.json"
            save_json(subreddit_data, subreddit_path)
            
            # Collect post titles
            for post in subreddit_data["posts"]:
                if post["title"]:
                    all_texts.append(post["title"])
            
            # Scrape comments if requested
            if include_comments:
                posts_to_scrape = subreddit_data["posts"][:max_posts]
                
                for post in posts_to_scrape:
                    if post["link"]:
                        comments = await scraper.scrape_post_comments(post["link"])
                        for comment in comments:
                            if comment["body"]:
                                all_texts.append(comment["body"])
            
            progress.advance(task)
    
    # Save all collected texts
    texts_path = COMMENTS_DIR / "all_texts.json"
    save_json(all_texts, texts_path)
    
    await scraper.close()
    return all_texts

async def run_classifier_logic(texts: List[str], openai_key: str):
    """Internal function to run classifier logic"""
    classifier = PainPointClassifier(openai_key)
    
    with Progress() as progress:
        task = progress.add_task("[blue]Classifying texts...", total=len(texts))
        
        def update_progress(current, total):
            progress.update(task, completed=current)
        
        pain_points = await classifier.classify_batch(texts, update_progress)
    
    return pain_points

@app.command("pipeline")
def run_full_pipeline(
    subreddits: Optional[str] = typer.Option(None, help="Comma-separated list of subreddits"),
    max_pages: int = typer.Option(2, help="Maximum pages to scrape per subreddit"),
    max_posts: int = typer.Option(3, help="Maximum posts to get comments from"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key")
):
    """
    Run the complete pipeline: scrape Reddit data and classify pain points
    """
    console.print("🚀 Starting full pipeline...")
    
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    setup_directories()
    target_subreddits = subreddits.split(",") if subreddits else SUBREDDITS
    
    async def run_pipeline():
        # Step 1: Scrape Reddit
        console.print("\n📡 Step 1: Scraping Reddit data...")
        all_texts = await run_scraper_logic(target_subreddits, max_pages, max_posts, True)
        
        console.print(f"✅ Scraping completed!")
        console.print(f"📊 Total texts collected: {len(all_texts)}")
        console.print(f"💾 Data saved to: {DATA_DIR}")
        
        # Step 2: Classify pain points
        console.print("\n🤖 Step 2: Classifying pain points...")
        pain_points = await run_classifier_logic(all_texts, openai_key)
        
        # Save results
        output_path = COMMENTS_DIR / "pain_points.json"
        save_json(pain_points, output_path)
        
        # Show summary
        console.print(f"✅ Classification completed!")
        console.print(f"🎯 Pain points found: {len(pain_points)}")
        console.print(f"📊 Conversion rate: {len(pain_points)/len(all_texts)*100:.1f}%")
        console.print(f"💾 Results saved to: {output_path}")
        
        # Show top categories
        categories = {}
        intensities = []
        
        for pp in pain_points:
            cat = pp.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            if pp.get("intensity"):
                intensities.append(pp["intensity"])
        
        if categories:
            table = Table(title="Top Pain Point Categories")
            table.add_column("Category", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Avg Intensity", style="yellow")
            
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                # Calculate average intensity for this category
                cat_intensities = [pp.get("intensity", 0) for pp in pain_points if pp.get("category") == cat and pp.get("intensity")]
                avg_intensity = sum(cat_intensities) / len(cat_intensities) if cat_intensities else 0
                
                table.add_row(cat, str(count), f"{avg_intensity:.1f}")
                
            console.print(table)
            
        # Show intensity distribution
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            high_intensity = len([i for i in intensities if i >= 7])
            console.print(f"\n📊 Pain Intensity Analysis:")
            console.print(f"   Average intensity: {avg_intensity:.1f}/10")
            console.print(f"   High intensity problems (7+): {high_intensity} ({high_intensity/len(intensities)*100:.1f}%)")
    
    asyncio.run(run_pipeline())
    console.print("\n🎉 Pipeline completed successfully!")

@app.command("info")
def show_info():
    """
    Show information about the Reddit Pain Point Analyzer
    """
    console.print("🔍 Reddit Pain Point Analyzer")
    console.print("📋 Discovers business opportunities through pain point analysis")
    console.print(f"🎯 Target subreddits: {', '.join(SUBREDDITS)}")
    console.print(f"💾 Data directory: {DATA_DIR}")
    console.print("\n📚 Available Commands:")
    console.print("   [bold cyan]Basic Commands:[/bold cyan]")
    console.print("   • scrape: Collect Reddit data")
    console.print("   • classify: Identify pain points with LLM")
    console.print("   • insights: Show individual high-intensity pain points")
    console.print("   • stats: Show data statistics")
    console.print("\n   [bold green]Advanced Clustering:[/bold green]")
    console.print("   • embeddings: Generate embeddings for clustering")
    console.print("   • cluster: Perform clustering and business analysis")
    console.print("   • show-clusters: View detailed cluster information")
    console.print("   • extract: Extract detailed business insights (original approach)")
    console.print("\n   [bold yellow]Complete Pipelines:[/bold yellow]")
    console.print("   • pipeline: Run scrape + classify")
    console.print("   • full-pipeline: Run complete analysis (all steps + extraction)")
    console.print("   • continue: Resume from existing scraped data")
    console.print("\n   [bold blue]Information:[/bold blue]")
    console.print("   • info: Show this information")
    console.print("\n💡 [bold]Typical Workflow:[/bold]")
    console.print("   1. python reddit_analyzer.py full-pipeline  # Complete analysis")
    console.print("   2. python reddit_analyzer.py show-clusters  # View results")
    console.print("   3. python reddit_analyzer.py show-clusters --cluster-id 0  # Details")

@app.command("continue")
def continue_pipeline(
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key")
):
    """
    Continue pipeline from existing scraped data (skip scraping, only classify)
    """
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    # Check if scraped data exists
    texts_file = COMMENTS_DIR / "all_texts.json"
    if not texts_file.exists():
        console.print("❌ No scraped data found. Run scraping first:")
        console.print("   python reddit_analyzer.py scrape")
        raise typer.Exit(1)
    
    console.print("🔄 Continuing pipeline from existing data...")
    console.print("🤖 Classifying pain points...")
    
    # Load and classify
    texts = load_json(texts_file)
    console.print(f"📄 Found {len(texts)} texts to classify")
    
    async def run_classification():
        pain_points = await run_classifier_logic(texts, openai_key)
        
        # Save results
        output_path = COMMENTS_DIR / "pain_points.json"
        save_json(pain_points, output_path)
        
        # Show summary
        console.print(f"✅ Classification completed!")
        console.print(f"🎯 Pain points found: {len(pain_points)}")
        console.print(f"📊 Conversion rate: {len(pain_points)/len(texts)*100:.1f}%")
        console.print(f"💾 Results saved to: {output_path}")
        
        # Show top categories
        categories = {}
        intensities = []
        
        for pp in pain_points:
            cat = pp.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            if pp.get("intensity"):
                intensities.append(pp["intensity"])
        
        if categories:
            table = Table(title="Top Pain Point Categories")
            table.add_column("Category", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Avg Intensity", style="yellow")
            
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                # Calculate average intensity for this category
                cat_intensities = [pp.get("intensity", 0) for pp in pain_points if pp.get("category") == cat and pp.get("intensity")]
                avg_intensity = sum(cat_intensities) / len(cat_intensities) if cat_intensities else 0
                
                table.add_row(cat, str(count), f"{avg_intensity:.1f}")
                
            console.print(table)
            
        # Show intensity distribution
        if intensities:
            avg_intensity = sum(intensities) / len(intensities)
            high_intensity = len([i for i in intensities if i >= 7])
            console.print(f"\n📊 Pain Intensity Analysis:")
            console.print(f"   Average intensity: {avg_intensity:.1f}/10")
            console.print(f"   High intensity problems (7+): {high_intensity} ({high_intensity/len(intensities)*100:.1f}%)")
        
        console.print("\n🎉 Pipeline completed successfully!")
        console.print("\n💡 Next steps:")
        console.print("   • View opportunities: python reddit_analyzer.py insights")
        console.print("   • See statistics: python reddit_analyzer.py stats")
    
    asyncio.run(run_classification())

@app.command("embeddings")
def generate_embeddings(
    input_file: str = typer.Option("data/comments/pain_points.json", help="Input file with pain points"),
    output_file: str = typer.Option("data/embeddings/reddit_embeddings.json", help="Output file for embeddings"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key")
):
    """
    Generate embeddings for pain points
    """
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    # Setup directories
    setup_directories()
    
    # Load pain points
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"❌ Input file not found: {input_file}")
        console.print("💡 Run classification first: python reddit_analyzer.py classify")
        raise typer.Exit(1)
    
    pain_points = load_json(input_path)
    
    # Extract texts from pain points
    texts = []
    if isinstance(pain_points, list):
        if pain_points and isinstance(pain_points[0], dict):
            # New format with metadata
            texts = [pp.get("text", "") for pp in pain_points if pp.get("text")]
        else:
            # Old format - just text strings
            texts = [str(pp) for pp in pain_points if pp]
    
    console.print(f"📄 Loaded {len(texts)} pain points for embedding generation")
    
    async def generate():
        generator = EmbeddingGenerator(openai_key)
        embeddings = await generator.generate_embeddings(texts)
        
        # Save embeddings
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(embeddings, output_path)
        
        console.print(f"✅ Embeddings generated successfully!")
        console.print(f"📊 Total embeddings: {len(embeddings)}")
        console.print(f"📐 Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        console.print(f"💾 Saved to: {output_path}")
    
    asyncio.run(generate())

@app.command("cluster")
def perform_clustering(
    embeddings_file: str = typer.Option("data/embeddings/reddit_embeddings.json", help="Embeddings file"),
    pain_points_file: str = typer.Option("data/comments/pain_points.json", help="Pain points file"),
    output_file: str = typer.Option("data/clusters/clusters.json", help="Output clusters file"),
    n_clusters: int = typer.Option(7, help="Number of clusters"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key for analysis")
):
    """
    Perform clustering on pain point embeddings
    """
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    # Load embeddings
    embeddings_path = Path(embeddings_file)
    if not embeddings_path.exists():
        console.print(f"❌ Embeddings file not found: {embeddings_file}")
        console.print("💡 Generate embeddings first: python reddit_analyzer.py embeddings")
        raise typer.Exit(1)
    
    # Load pain points
    pain_points_path = Path(pain_points_file)
    if not pain_points_path.exists():
        console.print(f"❌ Pain points file not found: {pain_points_file}")
        console.print("💡 Run classification first: python reddit_analyzer.py classify")
        raise typer.Exit(1)
    
    embeddings = load_json(embeddings_path)
    pain_points = load_json(pain_points_path)
    
    # Extract texts from pain points
    texts = []
    if isinstance(pain_points, list):
        if pain_points and isinstance(pain_points[0], dict):
            # New format with metadata
            texts = [pp.get("text", "") for pp in pain_points if pp.get("text")]
        else:
            # Old format - just text strings
            texts = [str(pp) for pp in pain_points if pp]
    
    console.print(f"📊 Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    
    async def cluster_and_analyze():
        analyzer = ClusterAnalyzer(openai_key)
        
        # Perform clustering
        labels = await analyzer.perform_clustering(embeddings, n_clusters)
        
        # Organize clusters
        clusters_dict = defaultdict(list)
        for label, text in zip(labels, texts):
            if text:  # Only add non-empty texts
                clusters_dict[int(label)].append(text)
        
        # Convert to normal dict
        clusters_dict = dict(clusters_dict)
        
        # Analyze each cluster with LLM
        console.print("\n🤖 Analyzing clusters with LLM...")
        cluster_analyses = {}
        
        for cluster_id, cluster_texts in clusters_dict.items():
            console.print(f"   Analyzing cluster {cluster_id} ({len(cluster_texts)} items)...")
            
            analysis = await analyzer.analyze_cluster(cluster_texts)
            cluster_analyses[cluster_id] = {
                "texts": cluster_texts,
                "analysis": analysis.model_dump(),
                "count": len(cluster_texts)
            }
        
        # Save results
        results = {
            "clusters": cluster_analyses,
            "metadata": {
                "n_clusters": n_clusters,
                "total_items": len(texts),
                "embeddings_dimensions": len(embeddings[0]) if embeddings else 0
            }
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, output_path)
        
        # Display results
        console.print(f"\n✅ Clustering and analysis completed!")
        console.print(f"📊 Created {len(cluster_analyses)} clusters")
        console.print(f"💾 Results saved to: {output_path}")
        
        # Show cluster summary
        table = Table(title="Cluster Analysis Summary")
        table.add_column("Cluster", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Main Theme", style="green")
        table.add_column("Urgency", style="yellow")
        table.add_column("Market Size", style="blue")
        
        for cluster_id, data in cluster_analyses.items():
            analysis = data["analysis"]
            table.add_row(
                str(cluster_id),
                str(data["count"]),
                analysis["main_theme"][:50] + "..." if len(analysis["main_theme"]) > 50 else analysis["main_theme"],
                f"{analysis['urgency']}/10",
                analysis["market_size"]
            )
        
        console.print(table)
        
        # Show top opportunities
        sorted_clusters = sorted(
            cluster_analyses.items(),
            key=lambda x: x[1]["analysis"]["urgency"] * x[1]["count"],
            reverse=True
        )
        
        console.print("\n🚀 Top Business Opportunities:")
        for i, (cluster_id, data) in enumerate(sorted_clusters[:3], 1):
            analysis = data["analysis"]
            console.print(f"\n📋 #{i} - Cluster {cluster_id} ({data['count']} pain points)")
            console.print(f"   Theme: {analysis['main_theme']}")
            console.print(f"   Opportunity: {analysis['business_opportunity']}")
            console.print(f"   Target: {analysis['target_audience']}")
            console.print(f"   Urgency: {analysis['urgency']}/10")
    
    asyncio.run(cluster_and_analyze())

@app.command("show-clusters")
def show_clusters(
    clusters_file: str = typer.Option("data/clusters/clusters.json", help="Clusters file"),
    cluster_id: Optional[int] = typer.Option(None, help="Show specific cluster (optional)"),
    limit: int = typer.Option(10, help="Maximum examples to show per cluster")
):
    """
    Show detailed cluster information and examples
    """
    clusters_path = Path(clusters_file)
    if not clusters_path.exists():
        console.print(f"❌ Clusters file not found: {clusters_file}")
        console.print("💡 Run clustering first: python reddit_analyzer.py cluster")
        raise typer.Exit(1)
    
    results = load_json(clusters_path)
    clusters = results.get("clusters", {})
    
    if not clusters:
        console.print("❌ No clusters found in file")
        return
    
    if cluster_id is not None:
        # Show specific cluster
        if str(cluster_id) not in clusters:
            console.print(f"❌ Cluster {cluster_id} not found")
            return
        
        cluster_data = clusters[str(cluster_id)]
        analysis = cluster_data["analysis"]
        
        console.print(f"🔍 [bold cyan]Cluster {cluster_id} Details[/bold cyan]")
        console.print(f"📊 Count: {cluster_data['count']} pain points")
        console.print(f"🎯 Main Theme: {analysis['main_theme']}")
        console.print(f"🚀 Business Opportunity: {analysis['business_opportunity']}")
        console.print(f"👥 Target Audience: {analysis['target_audience']}")
        console.print(f"⚡ Urgency: {analysis['urgency']}/10")
        console.print(f"📈 Market Size: {analysis['market_size'].title()}")
        console.print(f"🔧 Solution Complexity: {analysis['solution_complexity'].title()}")
        
        console.print(f"\n📝 Common Problems:")
        for i, problem in enumerate(analysis.get('common_problems', []), 1):
            console.print(f"   {i}. {problem}")
        
        console.print(f"\n💬 Example Pain Points:")
        for i, text in enumerate(cluster_data['texts'][:limit], 1):
            # Truncate long texts
            display_text = text[:200] + "..." if len(text) > 200 else text
            console.print(f"   {i}. \"{display_text}\"")
        
        if len(cluster_data['texts']) > limit:
            console.print(f"   ... and {len(cluster_data['texts']) - limit} more")
    
    else:
        # Show all clusters summary
        console.print("🔍 [bold cyan]All Clusters Overview[/bold cyan]")
        
        table = Table(title="Detailed Cluster Analysis")
        table.add_column("ID", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Main Theme", style="green")
        table.add_column("Business Opportunity", style="yellow")
        table.add_column("Urgency", style="red")
        table.add_column("Market", style="blue")
        
        for cluster_id, data in clusters.items():
            analysis = data["analysis"]
            
            # Truncate long texts for table display
            theme = analysis["main_theme"][:30] + "..." if len(analysis["main_theme"]) > 30 else analysis["main_theme"]
            opportunity = analysis["business_opportunity"][:40] + "..." if len(analysis["business_opportunity"]) > 40 else analysis["business_opportunity"]
            
            table.add_row(
                cluster_id,
                str(data["count"]),
                theme,
                opportunity,
                f"{analysis['urgency']}/10",
                analysis["market_size"]
            )
        
        console.print(table)
        
        console.print(f"\n💡 Use --cluster-id N to see detailed information for a specific cluster")
        console.print(f"💡 Example: python reddit_analyzer.py show-clusters --cluster-id 0")

@app.command("full-pipeline")
def run_complete_pipeline(
    subreddits: Optional[str] = typer.Option(None, help="Comma-separated list of subreddits"),
    max_pages: int = typer.Option(2, help="Maximum pages to scrape per subreddit"),
    max_posts: int = typer.Option(3, help="Maximum posts to get comments from"),
    n_clusters: int = typer.Option(7, help="Number of clusters for analysis"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key")
):
    """
    Run the complete pipeline: scrape → classify → embed → cluster → analyze → extract insights
    """
    console.print("🚀 Starting COMPLETE pipeline (all steps)...")
    
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    try:
        # Step 1: Scrape
        console.print("\n📡 Step 1: Scraping Reddit data...")
        setup_directories()
        target_subreddits = subreddits.split(",") if subreddits else SUBREDDITS
        
        async def run_complete():
            # Scrape data
            all_texts = await run_scraper_logic(target_subreddits, max_pages, max_posts, True)
            console.print(f"✅ Scraping: {len(all_texts)} texts collected")
            
            # Classify pain points
            console.print("\n🤖 Step 2: Classifying pain points...")
            pain_points = await run_classifier_logic(all_texts, openai_key)
            pain_points_path = COMMENTS_DIR / "pain_points.json"
            save_json(pain_points, pain_points_path)
            console.print(f"✅ Classification: {len(pain_points)} pain points found")
            
            # Generate embeddings
            console.print("\n🧠 Step 3: Generating embeddings...")
            generator = EmbeddingGenerator(openai_key)
            texts = [pp.get("text", "") for pp in pain_points if pp.get("text")]
            embeddings = await generator.generate_embeddings(texts)
            embeddings_path = EMBEDDINGS_DIR / "reddit_embeddings.json"
            save_json(embeddings, embeddings_path)
            console.print(f"✅ Embeddings: {len(embeddings)} embeddings generated")
            
            # Perform clustering
            console.print(f"\n🔍 Step 4: Clustering into {n_clusters} groups...")
            analyzer = ClusterAnalyzer(openai_key)
            labels = await analyzer.perform_clustering(embeddings, n_clusters)
            
            # Organize and analyze clusters
            clusters_dict = defaultdict(list)
            for label, text in zip(labels, texts):
                if text:
                    clusters_dict[int(label)].append(text)
            
            clusters_dict = dict(clusters_dict)
            
            console.print("\n🤖 Step 5: Analyzing clusters with LLM...")
            cluster_analyses = {}
            
            for cluster_id, cluster_texts in clusters_dict.items():
                analysis = await analyzer.analyze_cluster(cluster_texts)
                cluster_analyses[cluster_id] = {
                    "texts": cluster_texts,
                    "analysis": analysis.model_dump(),
                    "count": len(cluster_texts)
                }
            
            # Save final results
            results = {
                "clusters": cluster_analyses,
                "metadata": {
                    "n_clusters": n_clusters,
                    "total_items": len(texts),
                    "embeddings_dimensions": len(embeddings[0]) if embeddings else 0
                }
            }
            
            clusters_path = CLUSTERS_DIR / "clusters.json"
            save_json(results, clusters_path)
            
            return cluster_analyses
        
        cluster_analyses = asyncio.run(run_complete())
        
        # Show final results
        console.print(f"\n🎉 COMPLETE PIPELINE FINISHED!")
        console.print(f"📊 Final Results:")
        console.print(f"   • Texts collected: {len(asyncio.run(run_complete()))} (estimated)")
        console.print(f"   • Pain points: Available in data/comments/pain_points.json")
        console.print(f"   • Clusters: {len(cluster_analyses)} business opportunities identified")
        console.print(f"   • Analysis: Available in data/clusters/clusters.json")
        console.print(f"   • Detailed insights: Available in data/clusters/extracted_insights.txt")
        
        # Show top opportunities
        sorted_clusters = sorted(
            cluster_analyses.items(),
            key=lambda x: x[1]["analysis"]["urgency"] * x[1]["count"],
            reverse=True
        )
        
        console.print("\n🚀 TOP 3 BUSINESS OPPORTUNITIES:")
        for i, (cluster_id, data) in enumerate(sorted_clusters[:3], 1):
            analysis = data["analysis"]
            console.print(f"\n🥇 #{i} - {analysis['main_theme']}")
            console.print(f"   💡 Opportunity: {analysis['business_opportunity']}")
            console.print(f"   🎯 Target: {analysis['target_audience']}")
            console.print(f"   ⚡ Urgency: {analysis['urgency']}/10 | 📊 Items: {data['count']}")
        
        console.print(f"\n💡 Next steps:")
        console.print(f"   • Detailed view: python reddit_analyzer.py show-clusters")
        console.print(f"   • Specific cluster: python reddit_analyzer.py show-clusters --cluster-id 0")
        console.print(f"   • Business insights: cat data/clusters/extracted_insights.txt")
        console.print(f"   • Re-extract insights: python reddit_analyzer.py extract")
        
    except Exception as e:
        console.print(f"❌ Pipeline failed: {e}")
        logger.error(f"Complete pipeline error: {e}")
        raise typer.Exit(1)
    
@app.command("extract")
def extract_insights(
    clusters_file: str = typer.Option("data/clusters/clusters.json", help="Input clusters file"),
    output_file: str = typer.Option("data/clusters/extracted_insights.txt", help="Output file for insights"),
    api_key: Optional[str] = typer.Option(None, help="OpenAI API key")
):
    """
    Extract detailed business insights from clustered pain points (original approach)
    """
    # Get API key
    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        console.print("❌ OpenAI API key not found. Set OPENAI_API_KEY env var or use --api-key option.")
        raise typer.Exit(1)
    
    # Load clusters
    clusters_path = Path(clusters_file)
    if not clusters_path.exists():
        console.print(f"❌ Clusters file not found: {clusters_file}")
        console.print("💡 Run clustering first: python reddit_analyzer.py cluster")
        raise typer.Exit(1)
    
    clusters_data = load_json(clusters_path)
    clusters = clusters_data.get("clusters", {})
    
    if not clusters:
        console.print("❌ No clusters found in file")
        return
    
    console.print(f"🧠 Extracting detailed insights from {len(clusters)} clusters...")
    
    # Create the extraction prompt (based on your Jupyter notebook)
    key_topic_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um assistente pessoal que vai ler vários comentários e posts do Reddit que estão em um mesmo cluster semântico.
               Os comentários/posts revelam algum teor problemático para o usuário que comenta/posta. Dentro desse cluster, a ideia é que as dores dos usuários sejam similares.
               Assim, a sua tarefa é extrair as principais dores (os principais problemas) que os usuários estão enfrentando e que são comuns entre eles. Além disso, você deve
               também extrair um tópico geral e abrangente que resuma o cluster, que diga qual cluster é esse, pois eles foram obtidos a partir de um algoritmo de clusterização como K-Means, logo,
               precisamos dar um significado para esse cluster.
               
               Use o padrão abaixo para responder:
               
               [Nome do cluster]
               [Início da lista de tópicos]
                1. Tópico 1
                    - Descrição do tópico 1
                2. Tópico 2
                    - Descrição do tópico 2
                ...
                [Fim da lista de tópicos]
                
                OBS1: Vale salientar que alguns comentários/posts não vão ter sentido, ou seja, não vão revelar nenhum problema. Ainda sim, extraia as dores comuns.
                OBS2: Seria importante ter até 5 tópicos.
            """,
        ),
        (
            "human",
            """
            Aqui estão alguns comentários/posts do Reddit que revelam algum ou alguns problemas que os usuários estão enfrentando ou já enfrentaram:
            
            Comentários/posts: {comentarios_posts}
            
            (Cluster referente: {cluster_number})
            
            Por favor, me ajude a identificar os problemas comuns (os principais problemas) entre esses comentários/posts. Estou querendo encontrar alguma oportunidade para empreender e criar um produto
            ou serviço que resolva algum desses problemas.
            """,
        ),
    ])
    
    # Create LLM and chain
    llm = ChatOpenAI(
        model="gpt-4o-2024-08-06",
        api_key=openai_key
    )
    
    chain = key_topic_prompt | llm | StrOutputParser()
    
    async def extract_all_insights():
        all_results = []
        
        for cluster_id, cluster_data in track(clusters.items(), description="Extracting insights..."):
            cluster_texts = cluster_data.get("texts", [])
            
            if not cluster_texts:
                continue
            
            # Limit to avoid token limits (use first 30 texts)
            sample_texts = cluster_texts[:30]
            comentarios_posts_text = "\n".join([f"- {text}" for text in sample_texts])
            
            if len(cluster_texts) > 30:
                comentarios_posts_text += f"\n... e mais {len(cluster_texts) - 30} comentários similares"
            
            try:
                result = await asyncio.to_thread(
                    chain.invoke,
                    {
                        "comentarios_posts": comentarios_posts_text,
                        "cluster_number": cluster_id
                    }
                )
                
                all_results.append(f"{'='*80}")
                all_results.append(f"CLUSTER {cluster_id} - ({len(cluster_texts)} pain points)")
                all_results.append(f"{'='*80}")
                all_results.append(result)
                all_results.append("\n")
                
                console.print(f"✅ Cluster {cluster_id} processed ({len(cluster_texts)} items)")
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id}: {e}")
                all_results.append(f"❌ Erro ao processar cluster {cluster_id}: {e}\n")
        
        return all_results
    
    # Extract insights
    results = asyncio.run(extract_all_insights())
    
    # Save results to text file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("REDDIT PAIN POINT ANALYZER - EXTRACTED INSIGHTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated by extract command\n")
        f.write(f"Total clusters analyzed: {len(clusters)}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(result + "\n")
    
    console.print(f"\n✅ Extraction completed!")
    console.print(f"📊 Processed {len(clusters)} clusters")
    console.print(f"💾 Insights saved to: {output_path}")
    console.print(f"📖 View results: cat {output_path}")

@app.command("insights")
def show_insights(
    pain_points_file: str = typer.Option("data/comments/pain_points.json", help="Pain points JSON file"),
    min_intensity: int = typer.Option(7, help="Minimum pain intensity (1-10)"),
    min_confidence: int = typer.Option(8, help="Minimum classification confidence (1-10)"),
    limit: int = typer.Option(10, help="Maximum number of insights to show")
):
    """
    Show the most promising business opportunities (high intensity + high confidence pain points)
    """
    
    pain_points_path = Path(pain_points_file)
    if not pain_points_path.exists():
        console.print(f"❌ Pain points file not found: {pain_points_file}")
        console.print("💡 Run classification first: python reddit_analyzer.py classify")
        raise typer.Exit(1)
    
    pain_points = load_json(pain_points_path)
    
    # Filter high-potential opportunities
    opportunities = []
    for pp in pain_points:
        intensity = pp.get("intensity", 0)
        confidence = pp.get("confidence", 0)
        
        if intensity >= min_intensity and confidence >= min_confidence:
            opportunities.append(pp)
    
    if not opportunities:
        console.print(f"❌ No high-potential opportunities found.")
        console.print(f"   Try lowering --min-intensity ({min_intensity}) or --min-confidence ({min_confidence})")
        return
    
    # Sort by intensity * confidence score
    opportunities.sort(key=lambda x: (x.get("intensity", 0) * x.get("confidence", 0)), reverse=True)
    
    console.print(f"🚀 Top Business Opportunities Found: {len(opportunities)}")
    console.print(f"   Criteria: Intensity ≥ {min_intensity}, Confidence ≥ {min_confidence}")
    
    for i, opp in enumerate(opportunities[:limit], 1):
        console.print(f"\n📋 Opportunity #{i}")
        console.print(f"   Category: {opp.get('category', 'unknown').title()}")
        console.print(f"   Intensity: {opp.get('intensity', 0)}/10")
        console.print(f"   Confidence: {opp.get('confidence', 0)}/10")
        
        # Truncate long texts
        text = opp.get("text", "")
        if len(text) > 200:
            text = text[:200] + "..."
        console.print(f"   Pain Point: \"{text}\"")
        
    # Show category distribution for opportunities
    opp_categories = {}
    for opp in opportunities:
        cat = opp.get("category", "unknown")
        opp_categories[cat] = opp_categories.get(cat, 0) + 1
    
    if opp_categories:
        console.print(f"\n📊 Opportunity Categories:")
        for cat, count in sorted(opp_categories.items(), key=lambda x: x[1], reverse=True):
            console.print(f"   {cat.title()}: {count}")

@app.command("stats")
def show_stats():
    """
    Show statistics about collected data
    """
    if not DATA_DIR.exists():
        console.print("❌ No data found. Run scraping first.")
        return
    
    table = Table(title="Data Statistics")
    table.add_column("Subreddit", style="cyan")
    table.add_column("Posts", style="magenta")
    table.add_column("Status", style="green")
    
    total_posts = 0
    
    for subreddit in SUBREDDITS:
        subreddit_file = SUBREDDITS_DIR / subreddit / "subreddit.json"
        if subreddit_file.exists():
            data = load_json(subreddit_file)
            post_count = len(data.get("posts", []))
            total_posts += post_count
            table.add_row(subreddit, str(post_count), "✅")
        else:
            table.add_row(subreddit, "0", "❌")
    
    console.print(table)
    console.print(f"\n📊 Total posts collected: {total_posts}")
    
    # Check for pain points data
    pain_points_file = COMMENTS_DIR / "pain_points.json"
    if pain_points_file.exists():
        pain_points = load_json(pain_points_file)
        console.print(f"🎯 Pain points identified: {len(pain_points)}")
    
    # Check for clustering data
    clusters_file = CLUSTERS_DIR / "clusters.json"
    if clusters_file.exists():
        clusters_data = load_json(clusters_file)
        clusters = clusters_data.get("clusters", {})
        console.print(f"🔍 Business clusters analyzed: {len(clusters)}")
    
    # Show directory structure
    console.print(f"\n📁 Data Structure:")
    console.print(f"   📂 {DATA_DIR}")
    console.print(f"   ├── 📂 subreddits/")
    for subreddit in SUBREDDITS:
        subreddit_file = SUBREDDITS_DIR / subreddit / "subreddit.json"
        status = "✅" if subreddit_file.exists() else "❌"
        console.print(f"   │   ├── 📂 {subreddit}/ {status}")
    console.print(f"   ├── 📂 comments/")
    console.print(f"   ├── 📂 embeddings/")
    console.print(f"   └── 📂 clusters/")

if __name__ == "__main__":
    app()
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

class WebSearcher:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search web using SerpAPI
        Returns list of search results with title, snippet and link
        """
        try:
            search = GoogleSearch({
                "q": query + " site:hoanghamobile.com OR site:thegioididong.com OR site:fptshop.com.vn",
                "hl": "vi",
                "gl": "vn", 
                "num": num_results,
                "api_key": self.api_key
            })
            results = search.get_dict()
            
            if "error" in results:
                print(f"SerpAPI error: {results['error']}")
                return []
                
            organic_results = results.get("organic_results", [])
            
            formatted_results = []
            for result in organic_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", "")
                })
                
            return formatted_results

        except Exception as e:
            print(f"Error in web search: {e}")
            return []

    def format_results(self, results: List[Dict]) -> str:
        """Format search results into readable text"""
        if not results:
            return ""
            
        formatted = "Thông tin tìm được từ web:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   Nguồn: {result['link']}\n\n"
            
        return formatted
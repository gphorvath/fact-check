import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import requests
import textwrap
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys (You'll need to obtain these)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Paid: Google Cloud Console
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # Paid: OpenAI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")    # Paid: Google AI Studio  
SERPER_API_KEY = os.getenv("SERPER_API_KEY")    # Paid: SerperAPI for search results

# Configure APIs
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

class YouTubeFactChecker:
    def __init__(self, youtube_url: str, llm_provider: str = "openai"):
        self.youtube_url = youtube_url
        self.video_id = self._extract_video_id(youtube_url)
        self.transcript: Optional[List[Dict[str, Any]]] = None
        self.video_info: Optional[Dict[str, Any]] = None
        self.claims: List[Dict[str, Any]] = []
        self.llm_provider = llm_provider
        
    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if video_id_match:
            return video_id_match.group(1)
        else:
            raise ValueError("Invalid YouTube URL format")
    
    def fetch_video_info(self) -> Dict[str, Any]:
        """Get basic video information using YouTube API."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=self.video_id
        )
        response = request.execute()
        
        if response.get('items'):
            self.video_info = response['items'][0]
            return self.video_info
        else:
            raise ValueError(f"No video found with ID: {self.video_id}")
    
    def fetch_transcript(self) -> List[Dict[str, Any]]:
        """Fetch video transcript using YouTube Transcript API."""
        try:
            self.transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            return self.transcript
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            print("Attempting to fetch auto-generated transcript...")
            try:
                self.transcript = YouTubeTranscriptApi.get_transcript(self.video_id, languages=['en'])
                return self.transcript
            except Exception as e:
                raise ValueError(f"Could not retrieve transcript. Video may not have captions. Error: {e}")
    
    def _combine_transcript_segments(self) -> str:
        """Combine transcript segments into a full text."""
        if not self.transcript:
            self.fetch_transcript()
        
        if not self.transcript:
            return ""  # Return empty string if transcript is still None
            
        full_text = " ".join([segment.get('text', '') for segment in self.transcript])
        return full_text
    
    def identify_claims(self, max_claims: int = 10) -> List[Dict[str, Any]]:
        """Use LLM to identify factual claims in the transcript."""
        full_transcript = self._combine_transcript_segments()
        
        # Initialize claims to empty list by default
        claims: List[Dict[str, Any]] = []
        
        if not full_transcript:
            print("Warning: Empty transcript, cannot identify claims")
            self.claims = claims
            return claims
        
        prompt = f"""
        Analyze the following transcript from a YouTube video and identify up to {max_claims} 
        significant factual claims made in it. Focus on claims that:
        
        1. Make assertions about factual matters (not opinions)
        2. Are specific enough to be verified
        3. Could potentially be misleading if incorrect
        
        For each claim, extract:
        1. The exact claim as stated
        2. The context surrounding the claim
        3. The approximate timestamp (if you can determine it)
        
        Format your response as a JSON object with a "claims" array. Each claim in the array must have a "claim" field containing the exact claim text.
        
        Example format:
        {{
          "claims": [
            {{
              "claim": "The exact claim text here",
              "context": "The surrounding context",
              "timestamp": "Approximate timestamp"
            }},
            ...
          ]
        }}
        
        TRANSCRIPT:
        {full_transcript}
        """
        
        if self.llm_provider == "openai":
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    response_json = json.loads(response.choices[0].message.content)
                    response_claims = response_json.get("claims", [])
                    
                    # Validate each claim has the required 'claim' field
                    validated_claims = []
                    for claim in response_claims:
                        if isinstance(claim, dict) and "claim" in claim and claim["claim"]:
                            validated_claims.append(claim)
                        else:
                            print(f"Skipping invalid claim format: {claim}")
                    
                    claims = validated_claims
                else:
                    print("Error: OpenAI response does not have content")
            except (KeyError, json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing OpenAI response: {e}. Falling back to default format.")
            except Exception as e:
                print(f"Unexpected error with OpenAI: {e}")
        
        elif self.llm_provider == "gemini":
            try:
                model = genai.GenerativeModel('gemini-pro')
                gemini_response = model.generate_content(prompt)
                
                if hasattr(gemini_response, 'text') and gemini_response.text:
                    response_text = gemini_response.text
                    response_json = json.loads(response_text)
                    response_claims = response_json.get("claims", [])
                    
                    # Validate each claim has the required 'claim' field
                    validated_claims = []
                    for claim in response_claims:
                        if isinstance(claim, dict) and "claim" in claim and claim["claim"]:
                            validated_claims.append(claim)
                        else:
                            print(f"Skipping invalid claim format: {claim}")
                    
                    claims = validated_claims
                else:
                    print("Error: Gemini response does not have text content")
            except (KeyError, json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing Gemini response: {e}. Falling back to default format.")
            except Exception as e:
                print(f"Unexpected error with Gemini: {e}")
        
        self.claims = claims
        return claims
    
    def verify_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single claim using search API and LLM analysis."""
        if "claim" not in claim or not claim["claim"]:
            raise ValueError("Invalid claim format: missing or empty 'claim' field")
            
        claim_text = claim['claim']
        
        # Step 1: Search for relevant information
        search_results = self._search_web(claim_text)
        
        # Step 2: Use LLM to analyze search results and determine veracity
        verification_prompt = f"""
        I need to verify this claim from a YouTube video: "{claim_text}"
        
        Here are search results related to this claim:
        
        {search_results}
        
        Analyze these search results and determine:
        1. Whether the claim is TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE
        2. A detailed explanation supporting your determination
        3. Sources that support or contradict the claim
        
        Format your response as a JSON object with keys: "verdict", "explanation", "sources"
        """
        
        # Default verification result in case of errors
        verification = {
            "verdict": "ERROR",
            "explanation": "Failed to analyze verification results",
            "sources": []
        }
        
        if self.llm_provider == "openai":
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": verification_prompt}],
                    response_format={"type": "json_object"}
                )
                
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    verification_result = json.loads(response.choices[0].message.content)
                    if isinstance(verification_result, dict):
                        verification = {
                            "verdict": verification_result.get("verdict", "ERROR"),
                            "explanation": verification_result.get("explanation", "No explanation provided"),
                            "sources": verification_result.get("sources", [])
                        }
                else:
                    print("Error: OpenAI verification response does not have content")
            except json.JSONDecodeError as e:
                print(f"Error parsing OpenAI verification response: {e}")
            except Exception as e:
                print(f"Unexpected error with OpenAI verification: {e}")
        
        elif self.llm_provider == "gemini":
            try:
                model = genai.GenerativeModel('gemini-pro')
                gemini_response = model.generate_content(verification_prompt)
                
                if hasattr(gemini_response, 'text') and gemini_response.text:
                    verification_result = json.loads(gemini_response.text)
                    if isinstance(verification_result, dict):
                        verification = {
                            "verdict": verification_result.get("verdict", "ERROR"),
                            "explanation": verification_result.get("explanation", "No explanation provided"),
                            "sources": verification_result.get("sources", [])
                        }
                else:
                    print("Error: Gemini verification response does not have text content")
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Error parsing Gemini verification response: {e}")
            except Exception as e:
                print(f"Unexpected error with Gemini verification: {e}")
        
        # Add verification results to the claim
        claim.update({
            "verification": verification["verdict"],
            "explanation": verification["explanation"],
            "sources": verification["sources"]
        })
        
        return claim
    
    def _search_web(self, query: str) -> str:
        """Search the web for information about the claim."""
        if not SERPER_API_KEY:
            return "Error: SERPER_API_KEY not configured"
            
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        payload = json.dumps({
            "q": query,
            "num": 5
        })
        
        try:
            response = requests.post(
                'https://google.serper.dev/search', 
                headers=headers, 
                data=payload
            )
            
            if response.status_code == 200:
                search_data = response.json()
                
                # Format search results as text
                formatted_results = ""
                
                # Organic search results
                if 'organic' in search_data:
                    for i, result in enumerate(search_data['organic'][:5], 1):
                        title = result.get('title', 'No title')
                        link = result.get('link', 'No link')
                        snippet = result.get('snippet', 'No snippet')
                        
                        formatted_results += f"Result {i}:\n"
                        formatted_results += f"Title: {title}\n"
                        formatted_results += f"URL: {link}\n"
                        formatted_results += f"Snippet: {snippet}\n\n"
                
                return formatted_results
            else:
                return f"Error searching the web: {response.status_code}"
        except Exception as e:
            return f"Error searching the web: {str(e)}"
    
    def analyze_video(self) -> Dict[str, Any]:
        """Run the full fact-checking pipeline on the video."""
        # Get video metadata
        self.fetch_video_info()
        
        # Get transcript
        self.fetch_transcript()
        
        # Identify claims
        self.identify_claims()
        
        # Verify each claim
        verified_claims = []
        if not self.claims:
            print("Warning: No claims were identified in the transcript.")
        
        for claim in self.claims:
            try:
                if "claim" not in claim or not claim["claim"]:
                    print(f"Skipping claim with invalid format: {claim}")
                    continue
                verified_claim = self.verify_claim(claim)
                verified_claims.append(verified_claim)
            except Exception as e:
                print(f"Error verifying claim: {e}")
                # Add the error to the claim for debugging
                claim_with_error = claim.copy()
                claim_with_error["verification"] = "ERROR"
                claim_with_error["explanation"] = f"Error during verification: {str(e)}"
                claim_with_error["sources"] = []
                verified_claims.append(claim_with_error)
        
        # Ensure video_info exists and has the expected structure
        if not self.video_info:
            raise ValueError("Video information not available. Call fetch_video_info() first.")
            
        # Safely extract video metadata
        video_title = self.video_info.get("snippet", {}).get("title", "Unknown Title")
        channel_title = self.video_info.get("snippet", {}).get("channelTitle", "Unknown Channel")
        publish_date = self.video_info.get("snippet", {}).get("publishedAt", "Unknown Date")
        view_count = self.video_info.get("statistics", {}).get("viewCount", "0")
        
        # Compile final analysis
        analysis = {
            "video_id": self.video_id,
            "video_title": video_title,
            "channel": channel_title,
            "publish_date": publish_date,
            "view_count": view_count,
            "verified_claims": verified_claims,
            "summary": self._generate_summary(verified_claims)
        }
        
        return analysis
    
    def _generate_summary(self, verified_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of fact-checking results."""
        # Count verdicts
        verdicts = {
            "TRUE": 0,
            "FALSE": 0,
            "PARTIALLY TRUE": 0,
            "UNVERIFIABLE": 0,
            "ERROR": 0
        }
        
        for claim in verified_claims:
            verdict = claim.get("verification", "ERROR")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        # Calculate overall reliability score (simple version)
        total_claims = len(verified_claims) if verified_claims else 1
        true_weight = 1.0
        partially_true_weight = 0.5
        false_weight = 0.0
        unverifiable_weight = 0.5
        
        reliability_score = (
            (verdicts["TRUE"] * true_weight) +
            (verdicts["PARTIALLY TRUE"] * partially_true_weight) +
            (verdicts["FALSE"] * false_weight) +
            (verdicts["UNVERIFIABLE"] * unverifiable_weight)
        ) / total_claims
        
        # Generate text summary using LLM
        claims_text = ""
        for i, claim in enumerate(verified_claims[:5], 1):  # Limit to first 5 for brevity
            try:
                claims_text += f"Claim {i}: {claim.get('claim', 'Unknown claim')}\n"
                claims_text += f"Verdict: {claim.get('verification', 'ERROR')}\n\n"
            except Exception as e:
                claims_text += f"Claim {i}: Error formatting claim - {str(e)}\n\n"
        
        # Ensure video_info exists and has the expected structure
        if not self.video_info:
            video_title = "Unknown Title"
            channel_title = "Unknown Channel"
        else:
            video_title = self.video_info.get("snippet", {}).get("title", "Unknown Title")
            channel_title = self.video_info.get("snippet", {}).get("channelTitle", "Unknown Channel")
        
        summary_prompt = f"""
        Summarize the fact-checking results for this YouTube video:
        
        Video: {video_title}
        Channel: {channel_title}
        
        Claims verified: {len(verified_claims)}
        True claims: {verdicts["TRUE"]}
        Partially true claims: {verdicts["PARTIALLY TRUE"]}
        False claims: {verdicts["FALSE"]}
        Unverifiable claims: {verdicts["UNVERIFIABLE"]}
        
        Sample claims:
        {claims_text}
        
        Provide a brief assessment of the video's overall factual reliability and what viewers should be aware of.
        """
        
        # Default summary text
        summary_text = "Unable to generate summary."
        
        if self.llm_provider == "openai":
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                    summary_text = response.choices[0].message.content
            except Exception as e:
                print(f"Error generating summary with OpenAI: {e}")
        
        elif self.llm_provider == "gemini":
            try:
                model = genai.GenerativeModel('gemini-pro')
                gemini_response = model.generate_content(summary_prompt)
                if hasattr(gemini_response, 'text') and gemini_response.text:
                    summary_text = gemini_response.text
            except Exception as e:
                print(f"Error generating summary with Gemini: {e}")
        
        return {
            "verdict_counts": verdicts,
            "reliability_score": reliability_score,
            "text_summary": summary_text
        }

def main():
    parser = argparse.ArgumentParser(description='Fact-check claims in a YouTube video')
    parser.add_argument('url', type=str, help='YouTube video URL')
    parser.add_argument('--llm', type=str, choices=['openai', 'gemini'], default='openai',
                      help='LLM provider to use (default: openai)')
    parser.add_argument('--output', type=str, help='Output file path (JSON)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed error messages')
    
    args = parser.parse_args()
    
    print(f"Analyzing video: {args.url}")
    fact_checker = YouTubeFactChecker(args.url, llm_provider=args.llm)
    
    try:
        analysis = fact_checker.analyze_video()
        
        # Print summary to console
        print("\n" + "="*50)
        print(f"FACT CHECK RESULTS: {analysis['video_title']}")
        print("="*50)
        print(f"Channel: {analysis['channel']}")
        print(f"Published: {analysis['publish_date']}")
        print(f"Views: {analysis['view_count']}")
        print("\nSUMMARY:")
        print(textwrap.fill(analysis['summary']['text_summary'], width=80))
        print("\nRELIABILITY SCORE:", f"{analysis['summary']['reliability_score']:.2f}/1.00")
        print("\nCLAIMS ANALYZED:")
        
        for i, claim in enumerate(analysis['verified_claims'], 1):
            try:
                print(f"\n{i}. CLAIM: {claim.get('claim', 'Unknown claim')}")
                print(f"   VERDICT: {claim.get('verification', 'ERROR')}")
                print(f"   EXPLANATION: {textwrap.fill(claim.get('explanation', 'No explanation available'), width=80, initial_indent='   ', subsequent_indent='   ')}")
            except Exception as e:
                print(f"\n{i}. Error displaying claim: {str(e)}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nDetailed results saved to {args.output}")
            
    except Exception as e:
        if args.debug:
            import traceback
            print(f"Error analyzing video: {e}")
            print("\nDetailed error information:")
            traceback.print_exc()
        else:
            print(f"Error analyzing video: {e}")
            print("Run with --debug flag for more detailed error information.")

if __name__ == "__main__":
    main()

import os
import time
import traceback
from dotenv import load_dotenv
from typing import List, Optional
import openai
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_theme_summary(
    theme: str,
    snippets: List[str],
    query_context: Optional[str] = None
) -> str:
    """
    Generate a GPT-powered summary of themes found across documents.
    
    Args:
        theme: The identified theme to summarize
        snippets: List of text snippets supporting the theme
        query_context: Original user query for context
    
    Returns:
        str: Generated summary or error message
    """
    # Build the prompt with enhanced instructions
    prompt = f"""
    Analyze the following document excerpts related to the theme: "{theme}"
    {f"in the context of the query: '{query_context}'" if query_context else ""}

    Excerpts:
    {chr(10).join(f"- {s}" for s in snippets)}

    Task: Create a concise 3-5 sentence summary that:
    1. Identifies the common thread across excerpts
    2. Notes any significant variations or contradictions
    3. Maintains academic tone while being accessible
    4. Cites document sources where relevant
    """

    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Updated model
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific research assistant. "
                            "Provide accurate, well-cited summaries without speculation."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.3,  # More factual output
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Post-processing to ensure quality
            if len(summary.split()) < 10:  # Very short response
                raise ValueError("Summary too brief")
                
            return summary

        except Exception as e:
            error_str = str(e).lower()
            
            # Rate limit handling
            if any(kw in error_str for kw in ["rate limit", "quota", "429", "overloaded"]):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return (
                    "Summary delayed due to high demand. "
                    "Please try again shortly or check your API quota."
                )
            
            # Other errors
            traceback.print_exc()
            if attempt == max_retries - 1:  # Final attempt failed
                return (
                    "Unable to generate summary due to technical issues. "
                    "The raw excerpts are still available for review."
                )

    return "Summary generation failed after multiple attempts."
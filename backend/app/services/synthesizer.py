from openai import AsyncOpenAI
from typing import List

client = AsyncOpenAI()

async def generate_synthesis(themes: List[str], documents: List[dict]):
    prompt = f"""
    Analyze these themes across {len(documents)} documents:
    {chr(10).join(themes)}
    
    Create a executive summary highlighting:
    1. Consensus findings
    2. Key disagreements  
    3. Most cited evidence
    """
    
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
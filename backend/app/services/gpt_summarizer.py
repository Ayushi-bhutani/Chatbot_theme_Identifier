import os
import time
import traceback
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_theme_summary(theme: str, snippets: list[str]) -> str:
    prompt = (
        f"Summarize the following scientific literature snippets related to the theme '{theme}' "
        f"into a concise, informative paragraph:\n\n"
        + "\n\n".join(snippets)
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert scientific summarizer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return "Summary could not be generated due to quota limits. Please try again later."
            else:
                traceback.print_exc()
                return "Summary could not be generated at this time."

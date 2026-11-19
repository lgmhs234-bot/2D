import random
import time
import pandas as pd
from google import genai
from google.genai import types

# Load full cleaned dataset (all 292 rows remain available inside `df`)
df = pd.read_csv("cleaned_2d_sector_dataset.csv")


def run_pandas_analysis(code: str) -> str:
    """
    Executes pandas/python code against dataframe `df` and returns the result.
    Always assign your final output to a variable named `result`.
    """
    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "Code executed, but no `result` variable was set."))
    except Exception as e:
        return f"Error executing code: {e}"


# Keep system prompt compact (schema + 3 sample rows) to avoid heavy payloads
digest_cols = [
    "shortCode", "ownerUsername", "views", "likes", "comments",
    "outlier_index", "performance_tier", "discussion_rate_per_1k",
    "videoDuration", "caption", "top_comments"
]
sample_preview = df[digest_cols].head(3).to_csv(index=False)

SYSTEM_PROMPT = f"""
You are an expert 2D Animation Sector Intelligence AI advising the Instagram channel @vichardesk.
You have access to a pandas DataFrame `df` with {len(df)} Reels across {df['ownerUsername'].nunique()} benchmark creators.

COLUMNS AVAILABLE IN `df`:
{list(df.columns)}

METRICS DEFINITIONS:
- `outlier_index`: Video Views / Channel Median Views.
- `discussion_rate_per_1k`: (Comments / Views) * 1000.

RULES:
1. ALWAYS use the `run_pandas_analysis` tool for numbers, rankings, averages, filters, text searches, or correlations across `df`, and assign the final output to `result`.
2. Combine exact quantitative results from `df` with concise qualitative takeaways.

SAMPLE ROWS (first 3 rows for format reference):
{sample_preview}
"""

MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


def ask_sector_analyst(user_question: str, api_key: str) -> str:
    clean_key = (api_key or "").strip()
    if not clean_key:
        return "Please enter your API key in the sidebar first."

    client = genai.Client(api_key=clean_key)
    errors = []

    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_question,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=[run_pandas_analysis],
                        temperature=0.2,
                    ),
                )
                if response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                errors.append(f"{model_name} (try {attempt + 1}): {err_str}")
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.5 * (attempt + 1) + random.uniform(0.1, 0.5))
                    continue
                break

    return (
        "All Gemini models hit errors or high demand:\n\n"
        + "\n\n".join(errors[-4:])
    )

import pandas as pd
from google import genai
from google.genai import types

# Load cleaned data
df = pd.read_csv("cleaned_2d_sector_dataset.csv")

def run_pandas_analysis(code: str) -> str:
    """
    Executes pandas/python code against dataframe `df` and returns printed results.
    Always assign your final output to a variable named `result`.
    """
    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "Code executed, but no `result` variable was set."))
    except Exception as e:
        return f"Error executing code: {e}"

# Build a text digest of the entire dataset for qualitative context
digest_cols = [
    "shortCode", "ownerUsername", "views", "likes", "comments",
    "outlier_index", "performance_tier", "discussion_rate_per_1k",
    "videoDuration", "caption", "top_comments"
]
dataset_digest = df[digest_cols].to_csv(index=False)

SYSTEM_PROMPT = f"""
You are an expert 2D Animation Sector Intelligence AI advising the Instagram channel @vichardesk.
You have access to 292 cleaned Reels across 12 benchmark creators in the 2D explainer sector.

METRICS DEFINITIONS:
- `outlier_index`: Video Views / Channel Median Views.
- `discussion_rate_per_1k`: (Comments / Views) * 1000.

RULES:
1. For any request involving numbers, rankings, averages, filters, or correlations, ALWAYS invoke the `run_pandas_analysis` tool.
2. Combine exact quantitative results with qualitative insights.

FULL DATASET CONTEXT SUMMARY:
{dataset_digest}
"""

def ask_sector_analyst(user_question: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[run_pandas_analysis],
            temperature=0.2,
        ),
    )
    return response.text
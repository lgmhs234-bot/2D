import random
import time
import pandas as pd
from google import genai
from google.genai import types

# Load full cleaned dataset (all rows stay in `df` for analysis)
df = pd.read_csv("cleaned_2d_sector_dataset.csv")


def run_pandas_analysis(code: str) -> str:
    """Executes pandas code against dataframe `df` and returns `result`.

    Args:
        code: Python snippet using `df` and `pd` that assigns output to `result`.
    """
    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "Code executed, but no `result` variable was set."))
    except Exception as e:
        return f"Error executing code: {e}"


digest_cols = [
    "shortCode",
    "ownerUsername",
    "views",
    "likes",
    "comments",
    "outlier_index",
    "performance_tier",
    "discussion_rate_per_1k",
    "videoDuration",
    "caption",
    "top_comments",
]
available_digest_cols = [c for c in digest_cols if c in df.columns]
sample_preview = df[available_digest_cols].head(3).to_csv(index=False)

creator_count = df["ownerUsername"].nunique() if "ownerUsername" in df.columns else "multiple"

SYSTEM_PROMPT = f"""
You are a data analyst for the Instagram channel @vichardesk covering {len(df)} Reels across {creator_count} creators.
Columns in `df`: {list(df.columns)}
- `outlier_index`: Video Views / Channel Median Views
- `discussion_rate_per_1k`: (Comments / Views) * 1000

Call `run_pandas_analysis` once or twice at most when numbers, filters, rankings, or text searches on `df` are needed, assign final output to `result`, and then give a clear answer.

Sample rows:
{sample_preview}
"""

MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def ask_sector_analyst(user_question: str, api_key: str) -> str:
    clean_key = (api_key or "").strip()
    if not clean_key:
        return "Please enter your API key in the sidebar first."

    client = genai.Client(
        api_key=clean_key,
        http_options=types.HttpOptions(timeout=25_000),
    )
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
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=2
                        ),
                        temperature=0.2,
                    ),
                )
                if response.text:
                    return response.text
                errors.append(f"{model_name} (try {attempt + 1}): Empty text response")
            except Exception as e:
                err_str = str(e)
                errors.append(f"{model_name} (try {attempt + 1}): {err_str}")
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.0 * (attempt + 1) + random.uniform(0.1, 0.4))
                    continue
                break

    return "Could not complete analysis:\n\n" + "\n\n".join(errors[-4:])

import os
# Set the API key as an environment variable
os.environ["SERPER_API_KEY"] = "d0ca813b05f69b3f99b1557bfd7ca33fd39285ab"
from search_client import run_query

# Example: run a query using the configured search provider (Cerebras by default)
os.environ.setdefault("SEARCH_PROVIDER", "cerebras")

result = run_query("What is the capital of France?", provider=os.environ.get("SEARCH_PROVIDER"))
print(result)
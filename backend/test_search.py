import os
from pinecone import Pinecone
from google import genai
from embeddings import search

# The repository namespace where we stored your repo contents
namespace = "harshh06_my-journal"

# We can search for literally anything semantic!
query = "What is this journal project about?"

print(f"Searching Pinecone for '{query}' inside '{namespace}'...")
results = search(query, namespace, top_k=2)

print("\n--- TOP RESULTS ---")
for i, r in enumerate(results):
    print(f"\n[Result {i+1}]")
    print(f"File: {r['file_path']} (Starting at line {r['start_line']})")
    print("-" * 40)
    print(r['content'][:300] + "\n[...]")

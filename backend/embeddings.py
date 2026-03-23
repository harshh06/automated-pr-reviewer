# embeddings.py
from pinecone import Pinecone
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# We only instantiate if the key exists to avoid hard crashes without env set
pc_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pc_api_key) if pc_api_key else None
index = None
if pc:
    try:
        # This will query Pinecone to resolve the host. Fast fail if the index doesn't exist!
        index = pc.Index("pr-reviewer")
    except Exception as e:
        print(f"WARNING: Could not connect to Pinecone index 'pr-reviewer'. Did you create it in your dashboard? Initializing index as None.")

gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

def get_namespace(repo_url: str) -> str:
    # "https://github.com/user/repo" → "user_repo"
    return repo_url.replace("https://github.com/", "").replace("/", "_")

def embed(texts: list[str]) -> list[list[float]]:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not defined.")
    
    # Send content string arrays to the embeddings API
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=texts,
        config=genai.types.EmbedContentConfig(output_dimensionality=768)
    )
    return [e.values for e in result.embeddings]

def store_chunks(chunks: list[dict], namespace: str):
    if not index:
        return
        
    contents = [c["content"] for c in chunks]
    vectors = embed(contents)
    
    index.upsert(
        vectors=[
            {
                "id": f"{c['file_path']}_{i}",   # unique ID per chunk
                "values": vectors[i],
                "metadata": {
                    "content": c["content"],
                    "file_path": c["file_path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"]
                }
            }
            for i, c in enumerate(chunks)
        ],
        namespace=namespace
    )

def search(query: str, namespace: str, top_k: int = 5) -> list[dict]:
    if not index:
        return []
        
    query_vector = embed([query])[0]
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    
    return [
        {
            "content": m.metadata["content"],
            "file_path": m.metadata["file_path"],
            "start_line": m.metadata.get("start_line", 0)
        }
        for m in results.matches
    ]

def delete_file_chunks(file_path: str, namespace: str):
    """Deletes vectors belonging to the specified file path inside the given namespace."""
    if not index:
        return
    
    try:
        # We delete by filtering purely on the metadata parameter which Pinecone handles natively
        index.delete(
            filter={"file_path": file_path},
            namespace=namespace
        )
    except Exception as e:
        # If the namespace doesn't exist yet, Pinecone throws a 404. We safely ignore it.
        if "Namespace not found" in str(e) or "404" in str(e):
            pass
        else:
            print(f"Warning: Failed to delete chunks for {file_path}: {e}")

def has_namespace(namespace: str) -> bool:
    """Checks if the given namespace already exists in the Pinecone index."""
    if not index:
        return False
    try:
        stats = index.describe_index_stats()
        # namespaces is a dictionary: {"my_namespace": {"vector_count": 100}, ...}
        return namespace in stats.get("namespaces", {})
    except Exception as e:
        print(f"Warning: Could not check namespace stats: {e}")
        return False

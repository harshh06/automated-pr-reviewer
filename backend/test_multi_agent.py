import asyncio
from agent import run_agents_in_parallel

async def debug_orchestrator():
    repo_url = "https://github.com/harshh06/my-journal"
    
    # Simulating a massive PR diff affecting multiple layers
    pr_diff = '''--- a/src/lib/auth.ts
+++ b/src/lib/auth.ts
-// Missing input validation
+function authenticateUser(email) {
+  const query = "SELECT * FROM users WHERE email='" + email + "'"; // SQL Injection!
+  db.execute(query);
+}

--- a/src/components/List.tsx
+++ b/src/components/List.tsx
+  // O(N^2) unoptimized loop!
+  for (let i = 0; i < items.length; i++) {
+    for (let j = 0; j < items.length; j++) {
+      console.log(items[i], items[j]);
+    }
+  }
'''
    print("Testing Multi-Agent Orchestrator...")
    result = await run_agents_in_parallel(repo_url, pr_diff)
    print("\n\n=== FINAL AGGREGATE RESULT ===\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(debug_orchestrator())

import asyncio
from backend.agent import run_agents_in_parallel

async def main():
    repo_url = "https://github.com/test/repo"
    diff = "test diff"
    result = await run_agents_in_parallel(repo_url, diff)
    print("Result:", type(result))

if __name__ == "__main__":
    asyncio.run(main())

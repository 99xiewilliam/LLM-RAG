import asyncio
import aiohttp
import json
from typing import List, Optional
import time

class RAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []

    def log_result(self, test_name: str, success: bool, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "time": timestamp
        })
        status = "✅" if success else "❌"
        print(f"[{timestamp}] {status} {test_name}:")
        print(f"Response: {message}\n")

    async def test_add_document(self):
        """Test adding the Organization table document"""
        try:
            print("\n[{time.strftime('%H:%M:%S')}] Adding Organization table document...")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "files": [],
                    "split_method": "token"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/add_documents",
                    json=payload
                ) as response:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    if result.get("success"):
                        self.log_result(
                            "Add Organization Document", 
                            True, 
                            f"Document added successfully [Time: {elapsed:.2f}s]"
                        )
                    else:
                        self.log_result(
                            "Add Organization Document", 
                            False, 
                            f"Error: {result.get('error', 'Unknown error')}"
                        )
                await asyncio.sleep(2)
        except Exception as e:
            self.log_result("Add Organization Document", False, str(e))

    async def test_query(self, query: str):
        """Test querying the system"""
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Testing query: {query}")
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": query,
                    "target_language": "english"
                }
                
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/query",
                    json=payload
                ) as response:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    if "response" in result:
                        self.log_result(
                            f"Query: {query}", 
                            True, 
                            f"[Time: {elapsed:.2f}s]\n{result['response']}"
                        )
                    else:
                        self.log_result(
                            f"Query: {query}", 
                            False, 
                            f"Error: {result.get('error', 'Unknown error')}"
                        )
                await asyncio.sleep(2)
        except Exception as e:
            self.log_result(f"Query: {query}", False, str(e))

    async def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting APEC Organization RAG System Tests...\n")

        # First add the document
        await self.test_add_document()

        # Test queries about APEC organizations
        test_queries = [

        ]
        
        for query in test_queries:
            await self.test_query(query)
            await asyncio.sleep(2)

        # Print test summary
        print("\n📊 Test Summary:")
        success_count = sum(1 for r in self.test_results if r["success"])
        total_count = len(self.test_results)
        print(f"Total tests: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print(f"Success rate: {(success_count/total_count)*100:.2f}%")

if __name__ == "__main__":
    # Create tester instance
    tester = RAGTester("http://localhost:8000")
    
    # Run tests
    asyncio.run(tester.run_all_tests())
#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working correctly.
"""
import requests
import time
import json
import sys


def test_api_endpoints(base_url="http://localhost:8000"):
    """Test all API endpoints."""
    
    print(f"🔍 Testing PDF Search Engine API at {base_url}")
    print("="*50)
    
    # Test 1: Root endpoint
    try:
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ Root endpoint working")
            print(f"   📄 Response: {response.json()}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Root endpoint error: {e}")
    
    # Test 2: Health check
    try:
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            health_data = response.json()
            print(f"   📊 Status: {health_data.get('status')}")
            print(f"   📚 Index stats: {health_data.get('index_stats', {})}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 3: Index stats
    try:
        print("\n3. Testing index stats...")
        response = requests.get(f"{base_url}/index/stats")
        if response.status_code == 200:
            print("   ✅ Index stats working")
            stats = response.json()
            print(f"   📊 Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   📊 Total documents: {stats.get('total_documents', 0)}")
        else:
            print(f"   ❌ Index stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Index stats error: {e}")
    
    # Test 4: Document count
    try:
        print("\n4. Testing document count...")
        response = requests.get(f"{base_url}/docs-count")
        if response.status_code == 200:
            print("   ✅ Document count working")
            docs = response.json()
            print(f"   📄 PDF files found: {docs.get('count', 0)}")
        else:
            print(f"   ❌ Document count failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Document count error: {e}")
    
    # Test 5: Index documents (POST)
    try:
        print("\n5. Testing document indexing...")
        response = requests.post(f"{base_url}/index", timeout=60)
        if response.status_code == 200:
            print("   ✅ Document indexing working")
            result = response.json()
            print(f"   📚 Indexed documents: {result.get('indexed_documents', 0)}")
            print(f"   📝 Indexed chunks: {result.get('indexed_chunks', 0)}")
        else:
            print(f"   ❌ Document indexing failed: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                print(f"   📄 Error: {response.json()}")
    except Exception as e:
        print(f"   ❌ Document indexing error: {e}")
    
    # Test 6: Search (GET)
    try:
        print("\n6. Testing search (GET)...")
        response = requests.get(f"{base_url}/search?q=machine learning&max_results=5")
        if response.status_code == 200:
            print("   ✅ Search (GET) working")
            results = response.json()
            print(f"   🔍 Query: {results.get('query')}")
            print(f"   📄 Results found: {results.get('total_results', 0)}")
            print(f"   ⏱️ Search time: {results.get('search_time', 0):.3f}s")
        else:
            print(f"   ❌ Search (GET) failed: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                print(f"   📄 Error: {response.json()}")
    except Exception as e:
        print(f"   ❌ Search (GET) error: {e}")
    
    # Test 7: Search (POST)
    try:
        print("\n7. Testing search (POST)...")
        search_data = {
            "query": "machine learning",
            "max_results": 5
        }
        response = requests.post(
            f"{base_url}/search", 
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("   ✅ Search (POST) working")
            results = response.json()
            print(f"   🔍 Query: {results.get('query')}")
            print(f"   📄 Results found: {results.get('total_results', 0)}")
            print(f"   ⏱️ Search time: {results.get('search_time', 0):.3f}s")
            
            # Show first result if available
            if results.get('results'):
                first_result = results['results'][0]
                print(f"   📚 First result: {first_result.get('pdf_name')}")
                print(f"   📄 Page: {first_result.get('page_number')}")
                print(f"   🎯 Confidence: {first_result.get('confidence_score', 0):.3f}")
        else:
            print(f"   ❌ Search (POST) failed: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                print(f"   📄 Error: {response.json()}")
    except Exception as e:
        print(f"   ❌ Search (POST) error: {e}")
    
    print("\n" + "="*50)
    print("🏁 API endpoint testing completed!")


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api_endpoints(base_url)

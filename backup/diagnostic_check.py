#!/usr/bin/env python3
"""
Diagnostic script to check all system components before running queries.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment_variables():
    """Check if all required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = [
        "HACKRX_AUTH_TOKEN",
        "NOMIC_API_KEY",
        "GROQ_API_KEY_1"  # At least one Groq API key
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Set")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def test_nomic_api():
    """Test Nomic API connectivity."""
    print("\n🔍 Testing Nomic API...")
    
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        print("❌ NOMIC_API_KEY not set")
        return False
    
    url = "https://api-atlas.nomic.ai/v1/embedding/text"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": ["test"],
        "model": "nomic-embed-text-v1",
        "task_type": "search_document"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings[0]) == 768:
                print(f"✅ Nomic API working - embedding dimension: {len(embeddings[0])}")
                return True
            else:
                print(f"❌ Unexpected embedding format: {len(embeddings[0]) if embeddings else 0}")
                return False
        else:
            print(f"❌ Nomic API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Nomic API connection failed: {e}")
        return False

def test_groq_api():
    """Test Groq API connectivity."""
    print("\n🔍 Testing Groq API...")
    
    # Check for at least one Groq API key
    groq_keys = []
    for i in range(1, 11):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            groq_keys.append(key)
    
    if not groq_keys:
        single_key = os.getenv("GROQ_API_KEY")
        if single_key:
            groq_keys.append(single_key)
    
    if not groq_keys:
        print("❌ No Groq API keys found")
        return False
    
    print(f"✅ Found {len(groq_keys)} Groq API key(s)")
    
    # Test first key
    try:
        from groq import Groq
        client = Groq(api_key=groq_keys[0])
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'test'"}],
            model="llama-3.1-8b-instant",
            max_tokens=5
        )
        
        if completion.choices and completion.choices[0].message.content:
            print("✅ Groq API working")
            return True
        else:
            print("❌ Groq API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

def test_document_url():
    """Test if the sample document URL is accessible."""
    print("\n🔍 Testing document URL accessibility...")
    
    url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            content_length = response.headers.get('content-length', 'Unknown')
            print(f"✅ Document URL accessible - Size: {content_length} bytes")
            return True
        else:
            print(f"❌ Document URL error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Document URL test failed: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("🏥 Bajaj Finserv HackRx System Diagnostics")
    print("=" * 50)
    
    checks = [
        check_environment_variables,
        test_nomic_api,
        test_groq_api,
        test_document_url
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All diagnostics passed! System should be working correctly.")
    else:
        print("⚠️  Some diagnostics failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()

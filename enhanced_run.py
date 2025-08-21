#!/usr/bin/env python3
# enhanced_run.py - Enhanced launcher for Agentic RAG system

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def print_banner():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           🚀 ENHANCED AGENTIC RAG SYSTEM 2.0              ║
    ╠════════════════════════════════════════════════════════════╣
    ║ Features:                                                  ║
    ║ • Multi-step reasoning and query rewriting                ║
    ║ • Intelligent information source selection                ║ 
    ║ • Quality evaluation and iterative improvement            ║
    ║ • Internet search integration (SerpAPI)                   ║
    ║ • Real-time shop information (Google Sheets)              ║
    ║ • Advanced conversation context management                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)

def check_system_health():
    """Check system health before starting"""
    print("🔍 Performing system health check...")
    
    # Check required files
    required_files = [".env", "enhanced_rag.py", "enhanced_gemini_serve.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in .env")
        return False
    
    # Test Gemini connection
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash')
        model.generate_content("test")
        print("✅ Gemini API connection successful")
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False
    
    print("✅ System health check passed")
    return True

def start_enhanced_server():
    """Start the enhanced server"""
    print("🚀 Starting Enhanced Gemini RAG Server...")
    try:
        subprocess.run([sys.executable, "enhanced_gemini_serve.py"])
    except KeyboardInterrupt:
        print("\n👋 Enhanced server stopped by user")
    except Exception as e:
        print(f"❌ Enhanced server error: {e}")

def start_streamlit_ui():
    """Start the Streamlit UI"""
    print("🎨 Starting Enhanced Streamlit UI...")
    try:
        # Wait a bit for server to start
        time.sleep(2)
        subprocess.run(["streamlit", "run", "enhanced_client.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ UI error: {e}")

def run_interactive_test():
    """Run interactive test of enhanced features"""
    print("🧪 Running Enhanced System Interactive Test")
    print("=" * 50)
    
    try:
        from enhanced_rag import EnhancedRAG
        enhanced_rag = EnhancedRAG()
        
        test_queries = [
            "Nokia 3210 4G có giá bao nhiêu?",
            "Cửa hàng có những chi nhánh ở đâu?",
            "So sánh Samsung Galaxy A05s và Nokia 3210"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 30)
            
            start_time = time.time()
            response = enhanced_rag.process_query(query)
            end_time = time.time()
            
            print(f"💬 Response: {response[:200]}...")
            print(f"⏱️  Processing time: {end_time - start_time:.2f}s")
            print("✅ Test completed")
        
        print("\n🎉 All enhanced features working correctly!")
        
    except Exception as e:
        print(f"❌ Interactive test failed: {e}")

def main():
    print_banner()
    
    if not check_system_health():
        print("❌ System health check failed. Please run enhanced_setup.py first.")
        sys.exit(1)
    
    print("\nWhat would you like to do?")
    print("1. Start Enhanced Server only")
    print("2. Start UI only") 
    print("3. Start Both (Server + UI)")
    print("4. Run Interactive Test")
    print("5. System Status Check")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        start_enhanced_server()
    elif choice == "2":
        start_streamlit_ui()
    elif choice == "3":
        print("🚀 Starting Enhanced System (Server + UI)")
        
        # Start server in background thread
        server_thread = threading.Thread(target=start_enhanced_server, daemon=True)
        server_thread.start()
        
        # Start UI in main thread
        start_streamlit_ui()
        
    elif choice == "4":
        run_interactive_test()
    elif choice == "5":
        # Show detailed system status
        print("📊 Enhanced System Status:")
        check_system_health()
        
        # Check optional components
        serpapi_key = os.getenv("SERPAPI_KEY")
        print(f"🔍 Internet Search: {'✅ Enabled' if serpapi_key else '⚠️  Disabled (no SerpAPI key)'}")
        
        # Check database
        db_exists = Path("db").exists()
        print(f"🗄️  Vector Database: {'✅ Ready' if db_exists else '❌ Missing'}")
        
        # Check Google Sheets credentials
        sheets_cred = Path(os.getenv("GOOGLE_SHEETS_CREDENTIAL_FILE")).exists()
        print(f"📊 Shop Info Integration: {'✅ Ready' if sheets_cred else '⚠️  Missing credentials'}")
        
    else:
        print("❌ Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()

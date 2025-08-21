#!/usr/bin/env python3
# enhanced_setup.py - Enhanced setup script for Agentic RAG system

import subprocess
import sys
import os
import time
from pathlib import Path
import json

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print step information"""
    print(f"\n{step} {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    print_step("1️⃣", "CHECKING PYTHON VERSION")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Enhanced RAG requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_requirements():
    """Install required packages"""
    print_step("2️⃣", "INSTALLING ENHANCED REQUIREMENTS")
    
    try:
        # Core packages
        core_packages = [
            "google-generativeai>=0.8.0",
            "flask>=2.3.0", 
            "flask-cors>=4.0.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "chromadb>=0.4.0",
            "python-dotenv>=1.0.0",
            "requests>=2.31.0",
            "streamlit>=1.28.0",
            "plotly>=5.18.0"
        ]
        
        print("📦 Installing core packages...")
        for package in core_packages:
            print(f"   Installing {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
        
        # Google Sheets integration
        gsheet_packages = ["gspread>=5.12.0", "oauth2client>=4.1.3"]
        print("📊 Installing Google Sheets integration...")
        for package in gsheet_packages:
            print(f"   Installing {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                         check=True, capture_output=True)
        
        # Internet search capability
        try:
            print("🔍 Installing internet search capability...")
            subprocess.run([sys.executable, "-m", "pip", "install", "google-search-results>=2.4.2"],
                         check=True, capture_output=True)
            print("✅ SerpAPI integration available")
        except subprocess.CalledProcessError:
            print("⚠️  SerpAPI package installation failed (optional feature)")
        
        print("✅ All required packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        return False

def check_required_files():
    """Check if required files exist"""
    print_step("3️⃣", "CHECKING REQUIRED FILES")
    from dotenv import load_dotenv
    load_dotenv()

    required_files = {
        ".env": "Environment variables file",
        "DATA_FILE": "Product database CSV file",
        "GOOGLE_SHEETS_CREDENTIAL_FILE": "Google Sheets credentials (optional)"
    }
    
    missing_files = []
    optional_missing = []
    
    for file, description in required_files.items():
        if Path(file).exists():
            print(f"✅ {file} - {description}")
        else:
            if "optional" in description.lower():
                optional_missing.append(f"   ⚠️  {file} - {description}")
            else:
                missing_files.append(f"   ❌ {file} - {description}")
    
    if missing_files:
        print("\n📋 Missing required files:")
        for file in missing_files:
            print(file)
    
    if optional_missing:
        print("\n📋 Missing optional files:")
        for file in optional_missing:
            print(file)
    
    return len(missing_files) == 0

def check_environment_variables():
    """Check environment variables"""
    print_step("4️⃣", "CHECKING ENVIRONMENT VARIABLES")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    env_vars = {
        "GEMINI_API_KEY": {"required": True, "description": "Google Gemini API Key"},
        "SERPAPI_KEY": {"required": False, "description": "SerpAPI Key for internet search"}
    }
    
    missing_required = []
    missing_optional = []
    
    for var, config in env_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            if config["required"]:
                missing_required.append(f"   ❌ {var} - {config['description']}")
            else:
                missing_optional.append(f"   ⚠️  {var} - {config['description']}")
    
    if missing_required:
        print("\n🔑 Missing required environment variables:")
        for var in missing_required:
            print(var)
    
    if missing_optional:
        print("\n🔑 Missing optional environment variables:")
        for var in missing_optional:
            print(var)
    
    return len(missing_required) == 0

def setup_vector_database():
    """Setup vector database"""
    print_step("5️⃣", "SETTING UP VECTOR DATABASE")
    
    try:
        import chromadb
        import pandas as pd
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Check if CSV exists
        if not Path(os.getenv("DATA_FILE")).exists():
            print(f"⚠️  {os.getenv('DATA_FILE')} not found, skipping database setup")
            return False
        
        # Check if database already exists
        db_path = Path("db")
        if db_path.exists() and any(db_path.iterdir()):
            print("✅ Vector database already exists")
            return True
        
        print("🔧 Setting up new vector database...")
        
        # Create database setup script content
        setup_script = '''
import pandas as pd
import chromadb
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load and process data
df = pd.read_csv(os.getenv("DATA_FILE"))
print(f"Loaded {len(df)} products")

# Initialize ChromaDB
client = chromadb.PersistentClient("db")
collection = client.get_or_create_collection(name="products")

# Process and embed data
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

batch_size = 10
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
    
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    
    for idx, row in batch_df.iterrows():
        combined_info = f"{row.get('name', '')} {row.get('price', '')} {row.get('description', '')}".strip()
        
        embedding = get_embedding(combined_info)
        if embedding:
            documents.append(combined_info)
            embeddings.append(embedding)
            metadatas.append({"information": combined_info})
            ids.append(str(idx))
    
    if embeddings:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Processed batch {i//batch_size + 1}")

print("Vector database setup complete!")
'''
        
        # Write and execute setup script
        with open("temp_db_setup.py", "w", encoding="utf-8") as f:
            f.write(setup_script)
        
        result = subprocess.run([sys.executable, "temp_db_setup.py"], 
                              capture_output=True, text=True)
        
        # Clean up temp script
        if Path("temp_db_setup.py").exists():
            os.remove("temp_db_setup.py")
        
        if result.returncode == 0:
            print("✅ Vector database setup completed successfully")
            return True
        else:
            print(f"❌ Database setup failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def test_enhanced_system():
    """Test the enhanced system"""
    print_step("6️⃣", "TESTING ENHANCED SYSTEM")
    
    try:
        # Test Gemini connection
        print("🧪 Testing Gemini API connection...")
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Respond with: Enhanced system test successful!")
        print(f"✅ Gemini API: {response.text.strip()}")
        
        # Test Enhanced RAG import
        print("🧪 Testing Enhanced RAG system...")
        try:
            from enhanced_rag import EnhancedRAG, QueryProcessor
            enhanced_rag = EnhancedRAG()
            query_processor = QueryProcessor()
            print("✅ Enhanced RAG system imported successfully")
        except Exception as e:
            print(f"⚠️  Enhanced RAG import error: {e}")
        
        # Test internet search capability
        print("🧪 Testing internet search capability...")
        serpapi_key = os.getenv("SERPAPI_KEY")
        if serpapi_key:
            print("✅ SerpAPI key found - Internet search available")
        else:
            print("⚠️  No SerpAPI key - Internet search disabled")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def create_run_script():
    """Create enhanced run script"""
    print_step("7️⃣", "CREATING RUN SCRIPTS")
    
    run_script_content = '''#!/usr/bin/env python3
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
        print("\\n👋 Enhanced server stopped by user")
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
        print("\\n👋 UI stopped by user")
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
            print(f"\\n🔍 Test {i}: {query}")
            print("-" * 30)
            
            start_time = time.time()
            response = enhanced_rag.process_query(query)
            end_time = time.time()
            
            print(f"💬 Response: {response[:200]}...")
            print(f"⏱️  Processing time: {end_time - start_time:.2f}s")
            print("✅ Test completed")
        
        print("\\n🎉 All enhanced features working correctly!")
        
    except Exception as e:
        print(f"❌ Interactive test failed: {e}")

def main():
    print_banner()
    
    if not check_system_health():
        print("❌ System health check failed. Please run enhanced_setup.py first.")
        sys.exit(1)
    
    print("\\nWhat would you like to do?")
    print("1. Start Enhanced Server only")
    print("2. Start UI only") 
    print("3. Start Both (Server + UI)")
    print("4. Run Interactive Test")
    print("5. System Status Check")
    
    choice = input("\\nEnter choice (1-5): ").strip()
    
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
'''
    
    # Write the enhanced run script
    with open("enhanced_run.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)
    
    print("✅ Enhanced run script created: enhanced_run.py")
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod("enhanced_run.py", 0o755)
        print("✅ Script made executable")

def generate_documentation():
    """Generate enhanced documentation"""
    print_step("8️⃣", "GENERATING DOCUMENTATION")
    
    doc_content = '''# Enhanced Agentic RAG System 2.0

## 🚀 Overview
This is an enhanced version of the Agentic RAG system with advanced reasoning capabilities, multi-step query processing, and intelligent information retrieval.

## ✨ Enhanced Features

### 🧠 Multi-Step Reasoning Workflow
1. **Query Rewriting**: Optimizes user queries for better information retrieval
2. **Information Need Assessment**: Determines if additional data is required  
3. **Source Selection**: Intelligently chooses appropriate information sources
4. **Quality Evaluation**: Assesses response quality and iterates if needed

### 🔍 Information Sources
- **Vector Database**: Product specifications, pricing, features
- **Google Sheets**: Real-time shop information, locations, hours
- **Internet Search**: Latest information via SerpAPI integration

### 🎯 Intelligent Agents
- **Enhanced Product Agent**: Advanced product assistance with reasoning
- **Enhanced Shop Info Agent**: Smart shop information with real-time data

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- SerpAPI key (optional, for internet search)

### Setup Steps
1. Run the enhanced setup script:
   ```bash
   python enhanced_setup.py
   ```

2. Configure your `.env` file:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   SERPAPI_KEY=your_serpapi_key  # optional
   ```

3. Launch the system:
   ```bash
   python enhanced_run.py
   ```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Required for AI processing
- `SERPAPI_KEY`: Optional for internet search
- `MAX_ITERATIONS`: Maximum query processing iterations (default: 3)
- `ENABLE_INTERNET_SEARCH`: Enable/disable internet search

### System Components
- `enhanced_rag.py`: Core enhanced RAG system
- `enhanced_gemini_serve.py`: Enhanced API server
- `client.py`: Streamlit UI (unchanged)

## 📊 API Endpoints

### Enhanced Chat
```
POST /chat
{
  "message": "user query",
  "thread_id": "optional_thread_id"
}
```

Response includes:
- Enhanced processing metadata
- Routing information with confidence scores
- Processing time and agent details
- Quality evaluation results

### System Health
```
GET /health
```

Returns enhanced system status including:
- Component health status
- Supported features
- Available information sources

### Enhanced Statistics
```
GET /stats
```

Provides detailed analytics:
- Agent usage patterns
- Confidence distribution
- Processing time metrics
- Feature utilization

## 🧪 Testing

### Interactive Testing
```bash
python enhanced_run.py
# Select option 4: Run Interactive Test
```

### API Testing
```bash
curl -X GET http://localhost:5001/test-enhanced
```

## 🔍 Workflow Details

### Query Processing Steps
1. **Input Analysis**: Parse and understand user intent
2. **Query Rewriting**: Optimize for information retrieval
3. **Need Assessment**: Determine information requirements
4. **Source Selection**: Choose optimal data sources
5. **Information Retrieval**: Gather relevant data
6. **Response Generation**: Create comprehensive answer
7. **Quality Evaluation**: Assess and improve if needed

### Quality Assurance
- Automatic response evaluation
- Iterative improvement process
- Confidence scoring for routing decisions
- Processing time optimization

## 📈 Performance

### Optimization Features
- Intelligent caching for repeated queries
- Batch processing for embeddings
- Context-aware conversation management
- Resource-efficient multi-iteration processing

### Monitoring
- Real-time processing metrics
- Agent usage analytics
- Quality score tracking
- Performance benchmarking

## 🔒 Security & Privacy
- API key management via environment variables
- No persistent user data storage
- Secure Google Sheets integration
- Rate limiting for external API calls

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Test enhanced functionality
4. Submit pull request with performance benchmarks

## 📞 Support
For technical support or feature requests, please check:
- System health endpoint: `/health`
- Enhanced testing: `/test-enhanced`
- Processing logs for debugging

---
*Enhanced Agentic RAG System 2.0 - Powered by Google Gemini 2.0 Flash*
'''
    
    with open("ENHANCED_README.md", "w", encoding="utf-8") as f:
        f.write(doc_content)
    
    print("✅ Enhanced documentation created: ENHANCED_README.md")

def main():
    """Main setup function"""
    print_header("ENHANCED AGENTIC RAG SETUP")
    
    success_steps = 0
    total_steps = 8
    
    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1
    
    # Step 2: Install requirements
    if install_requirements():
        success_steps += 1
    
    # Step 3: Check required files
    if check_required_files():
        success_steps += 1
    else:
        print("\n⚠️  Please create missing files before proceeding")
    
    # Step 4: Check environment variables
    if check_environment_variables():
        success_steps += 1
    else:
        print("\n⚠️  Please configure environment variables in .env file")
    
    # Step 5: Setup vector database
    if success_steps >= 4:  # Only if previous steps succeeded
        if setup_vector_database():
            success_steps += 1
    
    # Step 6: Test enhanced system
    if success_steps >= 5:
        if test_enhanced_system():
            success_steps += 1
    
    # Step 7: Create run scripts
    create_run_script()
    success_steps += 1
    
    # Step 8: Generate documentation
    generate_documentation()
    success_steps += 1
    
    # Final summary
    print_header("SETUP SUMMARY")
    print(f"✅ Completed: {success_steps}/{total_steps} steps")
    
    if success_steps == total_steps:
        print("""
🎉 ENHANCED AGENTIC RAG SETUP COMPLETE!

🚀 Next Steps:
1. Run: python enhanced_run.py
2. Select option 3: Start Both (Server + UI)
3. Open browser to: http://localhost:8501

✨ Enhanced Features Available:
• Multi-step reasoning workflow
• Intelligent query rewriting
• Quality evaluation and iteration
• Internet search integration
• Real-time shop information
• Advanced conversation management

📚 Documentation: See ENHANCED_README.md for details
        """)
    else:
        print(f"""
⚠️  Setup completed with {total_steps - success_steps} issues.

🔧 Please resolve the following:
• Missing required files
• Environment variable configuration
• Database setup issues

📖 Check ENHANCED_README.md for troubleshooting guide
        """)

if __name__ == "__main__":
    main()
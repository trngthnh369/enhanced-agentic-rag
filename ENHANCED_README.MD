# 🚀 Enhanced Agentic RAG System 2.0

## Tổng Quan

Hệ thống Enhanced Agentic RAG 2.0 là một nâng cấp toàn diện của hệ thống RAG truyền thống, tích hợp khả năng **suy luận đa bước**, **đánh giá chất lượng**, và **truy xuất thông tin thông minh** cho cửa hàng điện thoại di động Việt Nam.

## ✨ Tính Năng Nâng Cao

### 🧠 Workflow Suy Luận Đa Bước
```
Query Input → Query Rewriting → Need Assessment → Source Selection → 
Information Retrieval → Response Generation → Quality Evaluation → 
Final Response (hoặc lặp lại nếu chưa đạt chất lượng)
```

### 🎯 Hệ Thống Agent Thông Minh
- **Enhanced Product Agent**: Tư vấn sản phẩm với khả năng suy luận cao
- **Enhanced Shop Info Agent**: Thông tin cửa hàng real-time từ Google Sheets
- **Intelligent Router**: Định tuyến thông minh với điểm tin cậy

### 🔍 Nguồn Thông Tin Đa Dạng
- **Vector Database (ChromaDB)**: Thông tin sản phẩm, giá cả, thông số
- **Google Sheets**: Thông tin cửa hàng real-time
- **Internet Search (SerpAPI)**: Tìm kiếm thông tin mới nhất

### 📊 Đánh Giá Chất Lượng Tự Động
- Đánh giá độ liên quan của câu trả lời
- Cải thiện lặp đi lặp lại cho đến khi đạt chất lượng
- Theo dõi metrics hiệu suất

## 🛠️ Cài Đặt & Thiết Lập

### Bước 1: Chuẩn Bị Môi Trường
```bash
# Clone repository và di chuyển vào thư mục
cd enhanced-agentic-rag

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### Bước 2: Chạy Setup Tự Động
```bash
python enhanced_setup.py
```

Script sẽ tự động:
- ✅ Kiểm tra phiên bản Python
- 📦 Cài đặt dependencies
- 🔍 Kiểm tra files cần thiết
- 🔑 Xác minh API keys
- 🗄️ Thiết lập vector database
- 🧪 Test hệ thống
- 📚 Tạo documentation

### Bước 3: Cấu Hình API Keys

Tạo file `.env`:
```bash
# Required: Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: SerpAPI for internet search
SERPAPI_KEY=your_serpapi_key_here

# System Configuration
MAX_ITERATIONS=3
ENABLE_INTERNET_SEARCH=True
ENABLE_QUALITY_EVALUATION=True
```

### Bước 4: Khởi Chạy Hệ Thống
```bash
python enhanced_run.py
```

Chọn tùy chọn:
1. **Server only** - Chỉ chạy API server
2. **UI only** - Chỉ chạy giao diện Streamlit  
3. **Both** - Chạy cả server và UI (khuyến nghị)
4. **Interactive Test** - Test các tính năng enhanced
5. **System Status** - Kiểm tra trạng thái hệ thống

## 📁 Cấu Trúc Dự Án

```
enhanced-agentic-rag/
├── enhanced_rag.py              # Core Enhanced RAG system
├── enhanced_gemini_serve.py     # Enhanced API server
├── enhanced_client.py           # Enhanced Streamlit UI
├── enhanced_setup.py            # Automated setup script
├── enhanced_run.py              # System launcher
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── hoanghamobile.csv           # Product database
├── mles-class-12c1216b7303.json # Google Sheets credentials
├── db/                         # Vector database
└── ENHANCED_README.md          # Detailed documentation
```

## 🔧 Chi Tiết Kỹ Thuật

### Enhanced RAG Workflow

#### 1. Query Rewriting (Viết Lại Câu Hỏi)
```python
def rewrite_query(self, original_query: str, context: str = "") -> str:
    # Tối ưu hóa câu hỏi để tìm kiếm hiệu quả hơn
    # Ví dụ: "Nokia giá bao nhiêu" → "Nokia 3210 4G giá cả thông số kỹ thuật"
```

#### 2. Information Need Assessment (Đánh Giá Nhu Cầu Thông Tin)
```python
def need_additional_info(self, query: str) -> bool:
    # Quyết định có cần truy xuất thông tin bổ sung không
    # YES: Cần database/internet search
    # NO: Có thể trả lời bằng kiến thức chung
```

#### 3. Source Selection (Chọn Nguồn Thông Tin)
```python
def determine_information_source(self, query: str) -> List[InformationSource]:
    # Chọn nguồn thông tin phù hợp:
    # - VECTOR_DATABASE: Sản phẩm, giá cả
    # - SHOP_INFO: Địa chỉ, giờ mở cửa
    # - INTERNET: Thông tin mới nhất
```

#### 4. Quality Evaluation (Đánh Giá Chất Lượng)
```python
def evaluate_response_quality(self, query: str, response: str) -> Tuple[bool, str]:
    # Đánh giá câu trả lời có:
    # - Trả lời đúng trọng tâm
    # - Thông tin chính xác và đầy đủ
    # - Liên quan trực tiếp đến câu hỏi
```

### API Endpoints

#### Enhanced Chat
```http
POST /chat
Content-Type: application/json

{
  "message": "Nokia 3210 4G có giá bao nhiêu?",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "role": "assistant",
  "content": "Nokia 3210 4G có giá 1,590,000 ₫...",
  "enhanced_processing": true,
  "routing_info": {
    "agent_used": "ENHANCED_PRODUCT",
    "confidence": "HIGH",
    "reasoning": "Product price inquiry detected",
    "complexity": "MEDIUM"
  },
  "processing_info": {
    "processing_time": 2.34,
    "agent_name": "Enhanced Product Assistant",
    "enhanced_features": [
      "Multi-step reasoning",
      "Quality evaluation",
      "Information need assessment"
    ]
  }
}
```

#### System Health
```http
GET /health
```

#### Enhanced Statistics
```http
GET /stats
```

## 🧪 Testing & Debugging

### Interactive Testing
```bash
python enhanced_run.py
# Chọn option 4: Run Interactive Test
```

### API Testing
```bash
# Test basic connection
curl -X GET http://localhost:5001/health

# Test enhanced features
curl -X GET http://localhost:5001/test-enhanced

# Test chat
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Nokia 3210 4G có giá bao nhiêu?"}'
```

### Debug Logs
Hệ thống cung cấp logs chi tiết:
```
🚀 Starting Enhanced RAG Processing
📝 Original query: Nokia 3210 4G có giá bao nhiêu?
🔄 Query rewritten: Nokia 3210 4G giá cả thông số kỹ thuật khuyến mãi
📊 Need additional info: True
🎯 Selected sources: ['vector_database']
🗄️ Vector DB result length: 1250 chars
💬 Generated response length: 420 chars
⭐ Response quality: True
✅ Response approved after 1 iteration(s)
```

## 📊 Metrics & Monitoring

### Performance Metrics
- **Processing Time**: Thời gian xử lý từng request
- **Quality Scores**: Điểm chất lượng câu trả lời
- **Confidence Levels**: Mức độ tin cậy routing
- **Agent Usage**: Thống kê sử dụng agents
- **Iteration Counts**: Số lần lặp để đạt chất lượng

### Streamlit Dashboard
- Real-time conversation analytics
- Processing time trends
- Agent usage distribution
- Quality score tracking
- System health monitoring

## 🔒 Bảo Mật & Hiệu Suất

### Security Features
- API key management qua environment variables
- No persistent user data storage
- Secure Google Sheets integration
- Request timeout và rate limiting

### Performance Optimization
- Intelligent caching cho repeated queries
- Batch processing cho embeddings
- Context-aware conversation management
- Resource-efficient multi-iteration processing

## 🚨 Troubleshooting

### Common Issues

#### 1. Gemini API Connection Error
```bash
❌ Gemini API connection failed: 401 Unauthorized
```
**Solution:** Kiểm tra `GEMINI_API_KEY` trong file `.env`

#### 2. Vector Database Missing
```bash
⚠️ Vector database not found
```
**Solution:** Chạy lại `python enhanced_setup.py`

#### 3. Internet Search Disabled
```bash
⚠️ No SerpAPI key - Internet search disabled
```
**Solution:** Thêm `SERPAPI_KEY` vào `.env` file

#### 4. Google Sheets Integration Error
```bash
❌ Error in retrieve_shop_info
```
**Solution:** Kiểm tra file `mles-class-12c1216b7303.json`

### Performance Issues

#### Slow Processing
- Giảm `MAX_ITERATIONS` trong `.env`
- Disable internet search nếu không cần thiết
- Kiểm tra network connectivity

#### Memory Usage
- Restart hệ thống sau mỗi 1000+ queries
- Clear conversation history thường xuyên
- Monitor system resources

## 🔄 Updates & Maintenance

### Regular Maintenance
```bash
# Update product database
python enhanced_setup.py  # Step 5 only

# Clear conversation history
rm -rf conversation_cache/

# Update vector embeddings
python update_embeddings.py
```

### System Updates
1. Backup existing configuration
2. Update code files
3. Run enhanced_setup.py
4. Test with enhanced_run.py option 4

## 🤝 Contributing

### Development Setup
1. Fork repository
2. Create feature branch
3. Implement enhancements
4. Add comprehensive tests
5. Submit PR with performance benchmarks

### Code Standards
- Type hints for all functions
- Comprehensive error handling
- Performance monitoring
- Documentation for new features

## 📞 Support

### Getting Help
1. **System Health**: Check `/health` endpoint
2. **Enhanced Testing**: Run `/test-enhanced`
3. **Debug Logs**: Enable detailed logging
4. **Documentation**: See `ENHANCED_README.md`

### Reporting Issues
Include in bug reports:
- System configuration (OS, Python version)
- Error logs and stack traces
- API response examples
- Performance metrics

---

## 🎯 Use Cases

### 1. Product Consultation
```
User: "Tôi cần điện thoại dưới 2 triệu có camera tốt"
System: 
- Rewrites: "Điện thoại giá dưới 2 triệu camera chất lượng cao khuyến mãi"
- Sources: Vector database + Internet search
- Response: Detailed product comparison với recommendations
```

### 2. Shop Information
```
User: "Cửa hàng mở cửa lúc nào?"
System:
- Sources: Google Sheets real-time data
- Response: Complete schedule với special hours
```

### 3. Complex Comparisons
```
User: "So sánh Nokia 3210 và Samsung A05s về pin và camera"
System:
- Multi-step reasoning để gather product specs
- Quality evaluation để ensure comprehensive comparison
- Iterative improvement nếu cần thêm thông tin
```

---

**Enhanced Agentic RAG System 2.0** - Powered by Google Gemini 2.0 Flash with advanced reasoning capabilities.

*Created with ❤️ for intelligent conversational AI*
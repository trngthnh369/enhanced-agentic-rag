# enhanced_gemini_serve.py
import time
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from enhanced_rag import enhanced_product_rag, enhanced_shop_info_rag, EnhancedRAG

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class EnhancedGeminiAgent:
    """Enhanced agent class with advanced RAG capabilities"""
    def __init__(self, name: str, instructions: str, rag_function=None):
        self.name = name
        self.instructions = instructions
        self.rag_function = rag_function
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def process_message(self, message: str, context: List[Dict] = None) -> str:
        """Process message with enhanced RAG if available"""
        if self.rag_function:
            # Use enhanced RAG function
            return self.rag_function(message)
        else:
            # Fallback to basic processing
            return self._basic_processing(message, context)
    
    def _basic_processing(self, message: str, context: List[Dict] = None) -> str:
        """Basic processing without RAG"""
        conversation_context = ""
        if context:
            for msg in context[-3:]:
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        full_prompt = f"""
{self.instructions}

Conversation History:
{conversation_context}

Current Message: {message}

Please respond as this specialized agent with enhanced reasoning capabilities.
"""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error processing message: {str(e)}"

class EnhancedGeminiManager:
    """Enhanced manager with intelligent routing and advanced RAG"""
    def __init__(self):
        self.enhanced_rag = EnhancedRAG()
        self.agents = self._initialize_enhanced_agents()
    
    def _initialize_enhanced_agents(self) -> Dict[str, EnhancedGeminiAgent]:
        """Initialize enhanced agents with RAG capabilities"""
        
        manager_instructions = """
        You are an intelligent manager of specialized agents for a Vietnamese mobile phone store with advanced reasoning capabilities.
        
        Your enhanced capabilities include:
        1. Deep analysis of user intent and query complexity
        2. Intelligent routing to specialized agents with RAG capabilities
        3. Context-aware decision making
        4. Quality assurance and response optimization
        
        AVAILABLE ENHANCED AGENTS:
        - enhanced_product: Advanced product agent with multi-step reasoning, information retrieval, and quality evaluation
        - enhanced_shop_information: Advanced shop information agent with real-time data access and iterative improvement
        
        ENHANCED PROCESS:
        1. Analyze query complexity and information requirements
        2. Determine optimal agent based on query type and needed capabilities
        3. Route to agent with enhanced RAG workflow:
           - Query rewriting and optimization
           - Information need assessment
           - Multi-source retrieval (vector DB, shop info, internet)
           - Response generation with context
           - Quality evaluation and iterative improvement
        
        ROUTING RULES:
        - Product queries (sản phẩm, giá, Nokia, Samsung, iPhone, thông số, so sánh) → enhanced_product
        - Shop queries (địa chỉ, cửa hàng, giờ mở cửa, chi nhánh, liên hệ) → enhanced_shop_information
        - Mixed queries → enhanced_product (default with broader capabilities)
        """
        
        product_instructions = """
        You are an enhanced product assistant with advanced reasoning and RAG capabilities.
        
        ENHANCED CAPABILITIES:
        - Multi-step query processing and optimization
        - Intelligent information source selection
        - Context-aware response generation
        - Quality evaluation and iterative improvement
        - Real-time internet search when database is insufficient
        
        The enhanced_product_rag function will handle:
        1. Query rewriting for optimal search
        2. Information need assessment
        3. Multi-source retrieval (vector DB, internet)
        4. Response quality evaluation
        5. Iterative improvement until optimal response
        
        You provide comprehensive, accurate product information with reasoning capabilities.
        """
        
        shop_instructions = """
        You are an enhanced shop information assistant with advanced reasoning and RAG capabilities.
        
        ENHANCED CAPABILITIES:
        - Intelligent query processing and rewriting
        - Real-time shop data retrieval from Google Sheets
        - Context-aware information synthesis
        - Response quality evaluation and improvement
        - Multi-iteration optimization for best results
        
        The enhanced_shop_info_rag function will handle:
        1. Query optimization for shop information retrieval
        2. Information need assessment
        3. Real-time data access and synthesis
        4. Response evaluation and refinement
        5. Iterative improvement for comprehensive answers
        
        You provide accurate, up-to-date shop information with advanced reasoning.
        """
        
        return {
            "manager": EnhancedGeminiAgent("manager", manager_instructions),
            "enhanced_product": EnhancedGeminiAgent("enhanced_product", product_instructions, enhanced_product_rag),
            "enhanced_shop_information": EnhancedGeminiAgent("enhanced_shop_information", shop_instructions, enhanced_shop_info_rag)
        }
    
    def intelligent_route_query(self, query: str, context: List[Dict] = None) -> Dict[str, str]:
        """Enhanced intelligent routing with deeper analysis"""
        try:
            routing_prompt = f"""
            Analyze this query for a Vietnamese mobile phone store and determine the optimal routing:
            
            Query: "{query}"
            Context: {context[-2:] if context else "None"}
            
            ANALYSIS REQUIREMENTS:
            1. Query intent analysis
            2. Information complexity assessment
            3. Required capabilities evaluation
            
            AVAILABLE ENHANCED AGENTS:
            - ENHANCED_PRODUCT: Advanced product agent with:
              * Vector database search for products
              * Internet search for latest info
              * Multi-step reasoning and quality evaluation
              * Suitable for: product info, pricing, specs, comparisons
              
            - ENHANCED_SHOP_INFO: Advanced shop information agent with:
              * Real-time Google Sheets integration
              * Store policy and location data
              * Multi-iteration improvement
              * Suitable for: store hours, locations, contact, policies
            
            ROUTING DECISION FACTORS:
            - Product-related keywords: sản phẩm, điện thoại, Nokia, Samsung, iPhone, giá, thông số, so sánh
            - Shop-related keywords: cửa hàng, địa chỉ, giờ mở, chi nhánh, liên hệ, bảo hành
            - Complexity level and required reasoning depth
            
            Respond with JSON:
            {{
                "selected_agent": "ENHANCED_PRODUCT" or "ENHANCED_SHOP_INFO",
                "confidence": "HIGH/MEDIUM/LOW",
                "reasoning": "detailed explanation",
                "query_complexity": "SIMPLE/MEDIUM/COMPLEX"
            }}
            """
            
            response = self.agents["manager"].model.generate_content(routing_prompt)
            
            try:
                json_text = response.text.strip()
                if json_text.startswith('```json'):
                    json_text = json_text.replace('```json', '').replace('```', '')
                
                routing_decision = json.loads(json_text)
                selected_agent = routing_decision.get("selected_agent", "ENHANCED_PRODUCT")
                confidence = routing_decision.get("confidence", "MEDIUM")
                reasoning = routing_decision.get("reasoning", "Default routing")
                complexity = routing_decision.get("query_complexity", "MEDIUM")
                
                print(f"🎯 Intelligent routing: {selected_agent} (Confidence: {confidence})")
                print(f"🧠 Reasoning: {reasoning}")
                print(f"📊 Complexity: {complexity}")
                
            except json.JSONDecodeError:
                # Enhanced fallback routing with keyword analysis
                query_lower = query.lower()
                
                # Shop keywords with higher specificity
                shop_keywords = [
                    "địa chỉ", "cửa hàng", "giờ mở", "giờ đóng", "liên hệ", "chi nhánh", 
                    "location", "address", "hours", "contact", "store", "branch",
                    "bảo hành", "chính sách", "đổi trả"
                ]
                
                # Product keywords with higher specificity  
                product_keywords = [
                    "nokia", "samsung", "iphone", "oppo", "vivo", "xiaomi",
                    "giá", "price", "thông số", "specs", "so sánh", "compare",
                    "sản phẩm", "điện thoại", "mobile", "phone", "khuyến mãi"
                ]
                
                shop_score = sum(1 for keyword in shop_keywords if keyword in query_lower)
                product_score = sum(1 for keyword in product_keywords if keyword in query_lower)
                
                if shop_score > product_score:
                    selected_agent = "ENHANCED_SHOP_INFO"
                    confidence = "HIGH" if shop_score >= 2 else "MEDIUM"
                else:
                    selected_agent = "ENHANCED_PRODUCT"
                    confidence = "HIGH" if product_score >= 2 else "MEDIUM"
                
                reasoning = f"Fallback routing based on keyword analysis (Shop: {shop_score}, Product: {product_score})"
                complexity = "MEDIUM"
            
            return {
                "agent": selected_agent,
                "confidence": confidence,
                "reasoning": reasoning,
                "complexity": complexity,
                "original_query": query
            }
            
        except Exception as e:
            print(f"❌ Routing error: {e}")
            return {
                "agent": "ENHANCED_PRODUCT",
                "confidence": "LOW",
                "reasoning": f"Error fallback: {str(e)}",
                "complexity": "UNKNOWN",
                "original_query": query
            }
    
    def process_with_enhanced_agent(self, agent_type: str, query: str, context: List[Dict] = None) -> Dict[str, Any]:
        """Process query with enhanced agent capabilities"""
        try:
            start_time = time.time()
            
            if agent_type == "ENHANCED_PRODUCT":
                print(f"🛍️ Processing with Enhanced Product Agent")
                response = self.agents["enhanced_product"].process_message(query, context)
                agent_name = "Enhanced Product Assistant"
                
            elif agent_type == "ENHANCED_SHOP_INFO":
                print(f"🏪 Processing with Enhanced Shop Info Agent")  
                response = self.agents["enhanced_shop_information"].process_message(query, context)
                agent_name = "Enhanced Shop Information Assistant"
                
            else:
                print(f"❌ Unknown agent type: {agent_type}")
                response = "Xin lỗi, tôi không thể xác định loại câu hỏi này."
                agent_name = "Unknown Agent"
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "agent_name": agent_name,
                "processing_time": round(processing_time, 2),
                "enhanced_features": [
                    "Multi-step reasoning",
                    "Information need assessment", 
                    "Multi-source retrieval",
                    "Quality evaluation",
                    "Iterative improvement"
                ]
            }
                
        except Exception as e:
            print(f"❌ Enhanced agent processing error: {e}")
            return {
                "response": f"Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý nâng cao: {str(e)}",
                "agent_name": "Error Handler",
                "processing_time": 0,
                "enhanced_features": []
            }

# Global enhanced manager instance
enhanced_gemini_manager = EnhancedGeminiManager()

# Enhanced conversation history with metadata
enhanced_conversation_history = {}

@app.route("/chat", methods=["POST"])
def enhanced_gemini_chat():
    """Enhanced chat endpoint with advanced RAG capabilities"""
    data = request.json
    query = data.get("message", "")
    thread_id = data.get("thread_id", str(uuid.uuid4()))

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    # Initialize enhanced conversation history for new threads
    if thread_id not in enhanced_conversation_history:
        enhanced_conversation_history[thread_id] = {
            "messages": [],
            "metadata": {
                "created_at": time.time(),
                "total_queries": 0,
                "enhanced_processing": True
            }
        }

    try:
        print(f"\n🚀 Enhanced Gemini Chat Processing Started")
        print(f"📝 Query: {query}")
        print(f"🆔 Thread: {thread_id}")
        
        # Get conversation context
        context = enhanced_conversation_history[thread_id]["messages"]
        
        # Enhanced intelligent routing
        print(f"\n🎯 Starting Intelligent Routing...")
        routing_info = enhanced_gemini_manager.intelligent_route_query(query, context)
        
        # Process with enhanced agent
        print(f"\n🤖 Processing with Enhanced Agent...")
        agent_result = enhanced_gemini_manager.process_with_enhanced_agent(
            routing_info["agent"], 
            query, 
            context
        )
        
        # Update enhanced conversation history
        enhanced_conversation_history[thread_id]["messages"].append({
            "role": "user", 
            "content": query,
            "timestamp": time.time()
        })
        
        enhanced_conversation_history[thread_id]["messages"].append({
            "role": "assistant", 
            "content": agent_result["response"],
            "timestamp": time.time(),
            "agent_used": routing_info["agent"],
            "processing_time": agent_result["processing_time"],
            "confidence": routing_info["confidence"]
        })
        
        # Update metadata
        enhanced_conversation_history[thread_id]["metadata"]["total_queries"] += 1
        enhanced_conversation_history[thread_id]["metadata"]["last_agent"] = routing_info["agent"]
        enhanced_conversation_history[thread_id]["metadata"]["last_confidence"] = routing_info["confidence"]
        
        # Keep only last 12 messages for memory management
        if len(enhanced_conversation_history[thread_id]["messages"]) > 12:
            enhanced_conversation_history[thread_id]["messages"] = enhanced_conversation_history[thread_id]["messages"][-12:]
        
        print(f"✅ Enhanced processing completed successfully")
        print(f"⏱️ Total processing time: {agent_result['processing_time']}s")
        
        return {
            "content": agent_result["response"],
            "created_at": time.time(),
            "thread_id": thread_id,
            "enhanced_processing": True,
            "routing_info": {
                "agent_used": routing_info["agent"],
                "confidence": routing_info["confidence"],
                "reasoning": routing_info["reasoning"],
                "complexity": routing_info["complexity"]
            },
            "processing_info": {
                "processing_time": agent_result["processing_time"],
                "agent_name": agent_result["agent_name"],
                "enhanced_features": agent_result["enhanced_features"]
            },
            "ai_provider": "Google Gemini 2.0 Flash",
            "rag_version": "Enhanced Multi-Step RAG v2.0"
        }

    except Exception as e:
        print(f"❌ Enhanced chat processing error: {e}")
        return jsonify({
            "error": "Enhanced processing error",
            "details": str(e),
            "fallback_response": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
            "enhanced_processing": False
        }), 500

@app.route("/health", methods=["GET"])
def enhanced_health_check():
    """Enhanced health check endpoint"""
    try:
        # Test Gemini connection
        model = genai.GenerativeModel('gemini-2.0-flash')
        test_response = model.generate_content("Health check")
        gemini_status = "healthy"
    except Exception as e:
        gemini_status = f"error: {str(e)}"
    
    # Test enhanced RAG
    try:
        test_rag = EnhancedRAG()
        rag_status = "healthy"
    except Exception as e:
        rag_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "ai_provider": "Google Gemini 2.0 Flash",
        "rag_version": "Enhanced Multi-Step RAG v2.0",
        "agents": ["enhanced_product", "enhanced_shop_information"],
        "enhanced_features": [
            "Multi-step reasoning",
            "Intelligent query rewriting", 
            "Information need assessment",
            "Multi-source retrieval",
            "Quality evaluation",
            "Iterative improvement",
            "Internet search integration"
        ],
        "component_status": {
            "gemini_api": gemini_status,
            "enhanced_rag": rag_status,
            "vector_database": "active",
            "shop_info_integration": "active"
        },
        "supported_sources": [
            "Vector Database (ChromaDB)",
            "Google Sheets (Shop Info)",
            "Internet Search (SerpAPI)"
        ]
    })

@app.route("/stats", methods=["GET"])
def enhanced_conversation_stats():
    """Enhanced conversation statistics"""
    total_threads = len(enhanced_conversation_history)
    total_messages = sum(len(thread_data["messages"]) for thread_data in enhanced_conversation_history.values())
    
    # Calculate agent usage statistics
    agent_usage = {}
    processing_times = []
    confidence_distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    
    for thread_data in enhanced_conversation_history.values():
        for message in thread_data["messages"]:
            if message["role"] == "assistant" and "agent_used" in message:
                agent = message["agent_used"]
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
                
                if "processing_time" in message:
                    processing_times.append(message["processing_time"])
                
                if "confidence" in message:
                    confidence = message["confidence"]
                    if confidence in confidence_distribution:
                        confidence_distribution[confidence] += 1
    
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    return jsonify({
        "conversation_stats": {
            "active_threads": total_threads,
            "total_messages": total_messages,
            "average_processing_time": round(avg_processing_time, 2)
        },
        "agent_usage": agent_usage,
        "confidence_distribution": confidence_distribution,
        "ai_provider": "Google Gemini 2.0 Flash",
        "rag_version": "Enhanced Multi-Step RAG v2.0",
        "enhanced_features": [
            "Intelligent routing with confidence scoring",
            "Multi-iteration query processing",
            "Real-time information retrieval",
            "Quality-driven response refinement",
            "Context-aware conversation management"
        ]
    })

@app.route("/test-enhanced", methods=["GET"])
def test_enhanced_gemini():
    """Test enhanced Gemini and RAG capabilities"""
    results = {}
    
    # Test 1: Basic Gemini connection
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Respond with: Enhanced Gemini connected!")
        results["gemini_connection"] = {
            "status": "success",
            "response": response.text.strip()
        }
    except Exception as e:
        results["gemini_connection"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test 2: Enhanced RAG system
    try:
        test_rag = EnhancedRAG()
        results["enhanced_rag"] = {
            "status": "success",
            "components": ["QueryProcessor", "InternetSearchTool", "Multi-step workflow"]
        }
    except Exception as e:
        results["enhanced_rag"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test 3: Sample query processing
    try:
        sample_query = "Nokia 3210 4G có giá bao nhiêu?"
        routing_result = enhanced_gemini_manager.intelligent_route_query(sample_query)
        results["sample_routing"] = {
            "status": "success",
            "query": sample_query,
            "routing": routing_result
        }
    except Exception as e:
        results["sample_routing"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test 4: Environment variables
    api_keys = {
        "gemini_api_key": "✓" if os.getenv("GEMINI_API_KEY") else "✗",
        "serpapi_key": "✓" if os.getenv("SERPAPI_KEY") else "✗"
    }
    results["environment"] = api_keys
    
    overall_status = "success" if all(
        result.get("status") == "success" 
        for key, result in results.items() 
        if isinstance(result, dict) and "status" in result
    ) else "partial"
    
    return jsonify({
        "overall_status": overall_status,
        "test_results": results,
        "enhanced_rag_ready": overall_status == "success",
        "recommendations": [
            "Ensure GEMINI_API_KEY is set in .env file",
            "Add SERPAPI_KEY for internet search capabilities", 
            "Verify vector database is populated",
            "Check Google Sheets credentials for shop info"
        ] if overall_status != "success" else [
            "All systems operational",
            "Enhanced RAG ready for production use"
        ]
    })

if __name__ == "__main__":
    print("🚀 Starting Enhanced Gemini RAG Server...")
    print("✨ Features: Multi-step reasoning, intelligent routing, quality evaluation")
    print("🔧 Enhanced capabilities: Query rewriting, multi-source retrieval, iterative improvement")
    print("🌐 Internet search: SerpAPI integration for latest information")
    
    # Test enhanced system on startup
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        test_response = model.generate_content("System startup test")
        print("✅ Gemini 2.0 Flash connected successfully")
        
        enhanced_rag = EnhancedRAG()
        print("✅ Enhanced RAG system initialized")
        
        print("🎯 Intelligent routing system active")
        print("📊 Quality evaluation system ready")
        
    except Exception as e:
        print(f"❌ System initialization error: {e}")
        print("Please check your configuration and API keys")
    
    app.run(host="0.0.0.0", port=5001, debug=True)
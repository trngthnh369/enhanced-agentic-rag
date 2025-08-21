# enhanced_rag.py
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import os
import numpy as np
import chromadb
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import time

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient("db")
collection_name = 'products'

class InformationSource(Enum):
    VECTOR_DATABASE = "vector_database"
    SHOP_INFO = "shop_info"
    INTERNET = "internet"
    NONE = "none"

class QueryProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def rewrite_query(self, original_query: str, context: str = "") -> str:
        """Step 1: Rewrite the original query for better understanding"""
        context_prompt = f"\nNgữ cảnh bổ sung: {context}" if context else ""
        
        prompt = f"""
        Bạn là một chuyên gia xử lý câu hỏi về điện thoại di động và cửa hàng. 
        Hãy viết lại câu hỏi sau để làm cho nó rõ ràng, cụ thể và dễ tìm kiếm hơn:

        Câu hỏi gốc: "{original_query}"{context_prompt}

        Yêu cầu viết lại:
        1. Giữ nguyên ý nghĩa chính của câu hỏi
        2. Làm rõ các từ khóa quan trọng về sản phẩm, giá cả, thông số
        3. Bổ sung ngữ cảnh cần thiết cho việc tìm kiếm
        4. Đảm bảo câu hỏi phù hợp với việc tìm kiếm trong database sản phẩm hoặc thông tin cửa hàng
        5. Chỉ trả về câu hỏi đã được viết lại, không giải thích thêm

        Câu hỏi đã viết lại:
        """
        
        try:
            response = self.model.generate_content(prompt)
            rewritten = response.text.strip().strip('"')
            print(f"🔄 Query rewritten: {original_query} -> {rewritten}")
            return rewritten
        except Exception as e:
            print(f"❌ Error in query rewriting: {e}")
            return original_query

    def need_additional_info(self, query: str) -> bool:
        """Step 2: Determine if additional information is needed"""
        prompt = f"""
        Phân tích câu hỏi sau về điện thoại di động/cửa hàng và quyết định có cần truy xuất thông tin bổ sung hay không:

        Câu hỏi: "{query}"

        Các trường hợp CẦN truy xuất thông tin:
        - Hỏi về sản phẩm cụ thể (Nokia, Samsung, iPhone...)
        - Hỏi về giá cả, khuyến mãi
        - Hỏi về thông số kỹ thuật
        - Hỏi về địa chỉ cửa hàng, giờ mở cửa
        - Hỏi so sánh sản phẩm
        - Hỏi về chính sách, bảo hành

        Các trường hợp KHÔNG cần truy xuất:
        - Chào hỏi đơn giản
        - Câu hỏi chung chung về công nghệ
        - Yêu cầu giải thích khái niệm cơ bản

        Hãy trả lời CHÍNH XÁC một từ:
        - "YES" nếu cần truy xuất thông tin từ database/internet
        - "NO" nếu có thể trả lời bằng kiến thức chung

        Trả lời:
        """
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip().upper()
            need_info = answer == "YES"
            print(f"📊 Need additional info: {need_info} for query: {query}")
            return need_info
        except Exception as e:
            print(f"❌ Error in need_additional_info: {e}")
            return True

    def determine_information_source(self, query: str) -> List[InformationSource]:
        """Step 3: Determine which information sources to use"""
        prompt = f"""
        Phân tích câu hỏi về điện thoại/cửa hàng và xác định nguồn thông tin phù hợp:

        Câu hỏi: "{query}"

        Nguồn thông tin có sẵn:
        1. "vector_database" - Thông tin sản phẩm chi tiết (Nokia, Samsung...), giá cả, thông số kỹ thuật, khuyến mãi
        2. "shop_info" - Thông tin cửa hàng: địa chỉ, giờ mở cửa, chi nhánh, liên hệ
        3. "internet" - Tìm kiếm thông tin mới nhất từ web khi database không đủ
        4. "none" - Không cần nguồn bổ sung

        Quy tắc chọn nguồn:
        - Hỏi về sản phẩm/giá/thông số → vector_database
        - Hỏi về địa chỉ/giờ mở cửa/cửa hàng → shop_info  
        - Cần thông tin mới nhất/sản phẩm không có trong DB → internet
        - Có thể dùng nhiều nguồn kết hợp

        Trả lời bằng JSON format:
        {{"sources": ["vector_database"], "reasoning": "lý do chọn nguồn này"}}

        Trả lời:
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean JSON response
            json_text = response.text.strip()
            if json_text.startswith('```json'):
                json_text = json_text.replace('```json', '').replace('```', '')
            
            result = json.loads(json_text)
            sources = []
            for source in result.get("sources", []):
                try:
                    sources.append(InformationSource(source))
                except ValueError:
                    print(f"⚠️ Invalid source: {source}")
                    continue
            
            if not sources:
                sources = [InformationSource.VECTOR_DATABASE]
                
            print(f"🎯 Selected sources: {[s.value for s in sources]}")
            print(f"💭 Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            return sources
            
        except Exception as e:
            print(f"❌ Error in determine_information_source: {e}")
            return [InformationSource.VECTOR_DATABASE]

    def evaluate_response_quality(self, original_query: str, response: str) -> Tuple[bool, str]:
        """Step 4: Evaluate if the response adequately answers the query"""
        prompt = f"""
        Đánh giá chất lượng câu trả lời cho câu hỏi về điện thoại/cửa hàng:

        Câu hỏi gốc: "{original_query}"
        Câu trả lời: "{response}"

        Tiêu chí đánh giá:
        1. Có trả lời đúng trọng tâm câu hỏi không?
        2. Thông tin có chính xác và đầy đủ không?
        3. Có cung cấp chi tiết cụ thể (giá, thông số, địa chỉ...) khi cần?
        4. Có liên quan trực tiếp đến câu hỏi không?
        5. Có đề xuất thêm thông tin hữu ích không?

        Trả lời bằng JSON format:
        {{
            "quality": "YES" hoặc "NO",
            "feedback": "nhận xét chi tiết về chất lượng câu trả lời",
            "missing_info": "thông tin còn thiếu (nếu có)"
        }}

        Trả lời:
        """
        
        try:
            response_eval = self.model.generate_content(prompt)
            json_text = response_eval.text.strip()
            if json_text.startswith('```json'):
                json_text = json_text.replace('```json', '').replace('```', '')
                
            result = json.loads(json_text)
            quality_good = result.get("quality", "NO") == "YES"
            feedback = result.get("feedback", "No feedback provided")
            
            print(f"⭐ Response quality: {quality_good}")
            print(f"📝 Feedback: {feedback}")
            
            return quality_good, feedback
            
        except Exception as e:
            print(f"❌ Error in evaluate_response_quality: {e}")
            return True, "Evaluation error"

class InternetSearchTool:
    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def search_web(self, query: str) -> str:
        """Search the internet using SerpAPI"""
        if not self.serpapi_key:
            return "Không có API key cho tìm kiếm internet."

        try:
            url = "https://serpapi.com/search"
            params = {
                "q": f"{query} điện thoại mobile phone Vietnam",
                "api_key": self.serpapi_key,
                "engine": "google",
                "google_domain": "google.com.vn",
                "gl": "vn",
                "hl": "vi",
                "num": 5
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return f"Lỗi tìm kiếm: HTTP {response.status_code}"
                
            data = response.json()
            
            # Extract useful information
            results = []
            if "organic_results" in data:
                for result in data["organic_results"][:3]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    results.append(f"- {title}: {snippet}")
            
            if results:
                return f"Thông tin từ internet:\n" + "\n".join(results)
            else:
                return "Không tìm thấy thông tin liên quan trên internet."
                
        except Exception as e:
            print(f"❌ Internet search error: {e}")
            return f"Lỗi khi tìm kiếm trên internet: {str(e)}"

class EnhancedRAG:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.internet_search = InternetSearchTool()
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.max_iterations = 3
        
    def get_embedding(self, text: str) -> list[float]:
        """Generate embeddings using Gemini"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return []

    def retrieve_from_vector_db(self, query: str) -> str:
        """Retrieve information from vector database"""
        try:
            collection = chroma_client.get_collection(name=collection_name)
            query_embedding = self.get_embedding(query)
            
            if not query_embedding:
                return "Không thể tạo embedding cho câu truy vấn."
            
            query_embedding = np.array(query_embedding)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            search_results = collection.query(
                query_embeddings=query_embedding.tolist(), 
                n_results=int(os.getenv("MAX_RESULTS_VECTOR_DB", 5))  # Increased for better coverage
            )

            metadatas = search_results.get('metadatas', [])
            search_result = ""
            
            for i, metadata_list in enumerate(metadatas):
                if isinstance(metadata_list, list):
                    for j, metadata in enumerate(metadata_list):
                        if isinstance(metadata, dict):
                            combined_text = metadata.get('information', 'No text available').strip()
                            search_result += f"{i+1}. {combined_text}\n\n"
                            
            result = search_result if search_result else "Không tìm thấy thông tin liên quan trong database."
            print(f"🗄️ Vector DB result length: {len(result)} chars")
            return result
            
        except Exception as e:
            print(f"❌ Error in retrieve_from_vector_db: {e}")
            return "Lỗi khi truy xuất dữ liệu từ cơ sở dữ liệu."

    def retrieve_shop_info(self) -> str:
        """Retrieve shop information from Google Sheets"""
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']

            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                os.getenv("GOOGLE_SHEETS_CREDENTIAL_FILE"), scope
            )
            client = gspread.authorize(credentials)
            sheet = client.open_by_url(
                'https://docs.google.com/spreadsheets/d/1mOkgLyo1oedOG1nlvoSHpqK9-fTFzE9ysLuKob9TXlg'
            ).sheet1

            data = sheet.get_all_records()
            result = json.dumps(data, ensure_ascii=False, indent=2)
            print(f"🏪 Shop info retrieved: {len(result)} chars")
            return result
            
        except Exception as e:
            print(f"❌ Error in retrieve_shop_info: {e}")
            return "Lỗi khi truy xuất thông tin cửa hàng."

    def retrieve_information(self, query: str, sources: List[InformationSource]) -> str:
        """Retrieve information from specified sources"""
        retrieved_info = []
        
        for source in sources:
            print(f"🔍 Retrieving from: {source.value}")
            
            if source == InformationSource.VECTOR_DATABASE:
                info = self.retrieve_from_vector_db(query)
                if info and "Không tìm thấy thông tin" not in info:
                    retrieved_info.append(f"📱 Thông tin sản phẩm:\n{info}")
                
            elif source == InformationSource.SHOP_INFO:
                info = self.retrieve_shop_info()
                if info and "Lỗi" not in info:
                    retrieved_info.append(f"🏪 Thông tin cửa hàng:\n{info}")
                
            elif source == InformationSource.INTERNET:
                info = self.internet_search.search_web(query)
                if info and "Lỗi" not in info:
                    retrieved_info.append(f"🌐 Từ internet:\n{info}")
        
        result = "\n\n---\n\n".join(retrieved_info)
        print(f"📚 Total retrieved info length: {len(result)} chars")
        return result

    def generate_response(self, original_query: str, updated_query: str, context: str = "") -> str:
        """Generate final response using Gemini"""
        if context:
            prompt = f"""
            Bạn là một chuyên viên tư vấn điện thoại di động chuyên nghiệp tại cửa hàng.
            
            Thông tin tham khảo đã được truy xuất:
            {context}

            Câu hỏi gốc của khách hàng: {original_query}
            Câu hỏi đã được xử lý: {updated_query}

            Yêu cầu trả lời:
            1. Trả lời chính xác và đầy đủ dựa trên thông tin đã truy xuất
            2. Ưu tiên thông tin cụ thể: giá cả, thông số kỹ thuật, địa chỉ
            3. Nếu có nhiều lựa chọn, hãy so sánh và đưa ra gợi ý
            4. Trả lời bằng tiếng Việt, giọng điệu thân thiện và chuyên nghiệp
            5. Nếu thông tin không đầy đủ, hãy nói rõ và gợi ý cách tìm hiểu thêm

            Câu trả lời:
            """
        else:
            prompt = f"""
            Bạn là một chuyên viên tư vấn điện thoại di động chuyên nghiệp.
            
            Câu hỏi của khách hàng: {original_query}

            Yêu cầu:
            1. Trả lời dựa trên kiến thức chung về điện thoại di động
            2. Giọng điệu thân thiện và chuyên nghiệp
            3. Trả lời bằng tiếng Việt
            4. Nếu cần thông tin cụ thể, hãy gợi ý khách hàng hỏi chi tiết hơn

            Câu trả lời:
            """

        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            print(f"💬 Generated response length: {len(result)} chars")
            return result
        except Exception as e:
            print(f"❌ Error in generate_response: {e}")
            return "Xin lỗi, tôi không thể tạo câu trả lời lúc này. Vui lòng thử lại sau."

    def process_query(self, original_query: str) -> str:
        """Main enhanced processing workflow with reasoning and action"""
        print(f"\n🚀 Starting Enhanced RAG Processing")
        print(f"📝 Original query: {original_query}")
        print("="*60)
        
        current_query = original_query
        iteration = 0
        context_feedback = ""
        
        while iteration < self.max_iterations:
            print(f"\n🔄 ITERATION {iteration + 1}/{self.max_iterations}")
            print("-" * 40)
            
            # Step 1: Query Rewriting
            print("1️⃣ QUERY REWRITING")
            updated_query = self.query_processor.rewrite_query(current_query, context_feedback)
            
            # Step 2: Information Need Assessment
            print("\n2️⃣ INFORMATION NEED ASSESSMENT")
            needs_info = self.query_processor.need_additional_info(updated_query)
            
            if not needs_info:
                print("ℹ️ No additional information needed, generating direct response")
                response = self.generate_response(original_query, updated_query)
            else:
                print("🔍 Additional information needed, proceeding to retrieval")
                
                # Step 3: Information Source Selection
                print("\n3️⃣ INFORMATION SOURCE SELECTION")
                sources = self.query_processor.determine_information_source(updated_query)
                
                # Step 4: Information Retrieval
                print("\n4️⃣ INFORMATION RETRIEVAL")
                context = self.retrieve_information(updated_query, sources)
                
                # Step 5: Response Generation
                print("\n5️⃣ RESPONSE GENERATION")
                response = self.generate_response(original_query, updated_query, context)
            
            # Step 6: Response Quality Evaluation
            print("\n6️⃣ RESPONSE QUALITY EVALUATION")
            is_good_response, feedback = self.query_processor.evaluate_response_quality(
                original_query, response
            )
            
            if is_good_response:
                print(f"✅ Response approved after {iteration + 1} iteration(s)")
                print("="*60)
                return response
            
            # Prepare for next iteration
            print(f"🔄 Response needs improvement, preparing iteration {iteration + 2}")
            context_feedback = feedback
            current_query = f"""
            Câu hỏi gốc: {original_query}
            Câu trả lời trước chưa đạt yêu cầu: {response}
            Phản hồi: {feedback}
            Hãy cải thiện câu hỏi để có thể truy xuất thông tin tốt hơn.
            """
            iteration += 1
        
        print(f"⚠️ Max iterations reached, returning best available response")
        print("="*60)
        return response

# Global enhanced RAG instance
enhanced_rag = EnhancedRAG()

def enhanced_product_rag(query: str) -> str:
    """Enhanced RAG for product information with reasoning and action"""
    print(f'\n🛍️ Enhanced Product RAG initiated')
    return enhanced_rag.process_query(query)

def enhanced_shop_info_rag(query: str) -> str:
    """Enhanced RAG for shop information with reasoning and action"""
    print(f'\n🏪 Enhanced Shop Info RAG initiated')
    return enhanced_rag.process_query(query)
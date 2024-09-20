import csv
from typing import List, Dict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr


class KeywordRetriever(BaseRetriever):
    """A custom retriever that matches user input with predefined keywords
    and retrieves the associated documents."""

    _keyword_file_path: str = PrivateAttr()
    _k: int = PrivateAttr()
    _keyword_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, keyword_file_path: str, k: int = 5):
        """Initialize the retriever, load keywords from the specified file path."""
        super().__init__()
        object.__setattr__(self, '_keyword_file_path', keyword_file_path)
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, '_keyword_map', self._load_keywords())

    def _load_keywords(self) -> Dict[str, str]:
        """Load keywords and associated documents from the specified CSV file."""
        keyword_map = {}
        with open(self._keyword_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            print(f"CSV Columns: {reader.fieldnames}")  # 打印列名
            for row in reader:
                #print(f"Row: {row}")  # 打印每一行
                keywords = row['keywords'].split(' ')
                for keyword in keywords:
                    keyword_map[keyword.lower()] = row['introduction']
        return keyword_map


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementation for retrieving relevant documents."""
        matched_documents = []
        query_lower = query.lower()

        # Check if any keyword is in the user query
        for keyword, content in self._keyword_map.items():
            if keyword in query_lower:
                doc = Document(page_content=content)
                matched_documents.append(doc)

        # Remove duplicates and limit to top k results
        unique_documents = list({doc.page_content: doc for doc in matched_documents}.values())
        return unique_documents[:self._k]

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import PrivateAttr
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class VectorRetriever(BaseRetriever):
    _directory: str = PrivateAttr()
    _chunk_size: int = PrivateAttr()
    _chunk_overlap: int = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()

    def __init__(self, directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__()
        object.__setattr__(self, '_directory', directory)
        object.__setattr__(self, '_chunk_size', chunk_size)
        object.__setattr__(self, '_chunk_overlap', chunk_overlap)
        
        documents = self._load_documents_from_dir_group()
        splits = self._split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese"))
        object.__setattr__(self, '_vectorstore', vectorstore)

    def _load_documents_from_dir_group(self) -> List[Document]:
        docs = []
        for root, _, files in os.walk(self._directory):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_content = f.read()
                    relative_dir = os.path.relpath(root, self._directory)
                    group = os.path.basename(relative_dir) if relative_dir != '.' else ''
                    doc_metadata = {'source': os.path.splitext(file)[0], 'group': group}
                    doc = Document(page_content=doc_content, metadata=doc_metadata)
                    docs.append(doc)
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        return text_splitter.split_documents(docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        retriever = self._vectorstore.as_retriever()
        return retriever.get_relevant_documents(query, run_manager=run_manager)

    def get_vectorstore(self):
        return self._vectorstore

# 用向量化的关键词实现检索（向量化的关键词检索）
import os
import csv
from typing import List, Dict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pydantic import PrivateAttr
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import torch
from tqdm import tqdm

class KeywordVectorRetriever(BaseRetriever):
    """A custom retriever that encodes keywords and retrieves documents based on vector similarity."""

    _keyword_file_path: str = PrivateAttr()
    _k: int = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()
    _keyword_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, keyword_file_path: str, k: int = 5):
        """Initialize the retriever, load keywords from the specified file path, and create vector store."""
        super().__init__()
        object.__setattr__(self, '_keyword_file_path', keyword_file_path)
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, '_keyword_map', self._load_keywords())

        # Encode keywords and create vector store
        documents = self._create_keyword_documents()
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
        object.__setattr__(self, '_vectorstore', vectorstore)

    def _load_keywords(self) -> Dict[str, str]:
        """Load keywords and associated documents from the specified CSV file."""
        keyword_map = {}
        with open(self._keyword_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc="Loading keywords"):
                keywords = row['keywords'].split(' ')
                for keyword in keywords:
                    keyword_map[keyword.lower()] = row['introduction']
        return keyword_map

    def _create_keyword_documents(self) -> List[Document]:
        """Create documents from keywords for vector encoding."""
        documents = []
        for keyword, content in tqdm(self._keyword_map.items(), desc="Creating keyword documents"):
            doc = Document(page_content=keyword, metadata={'content': content})
            documents.append(doc)
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve relevant documents based on vector similarity."""
        retriever = self._vectorstore.as_retriever()
        keyword_docs = retriever.get_relevant_documents(query, run_manager=run_manager)

        # Retrieve the associated content for the matched keywords
        matched_documents = []
        for doc in keyword_docs:
            content = self._keyword_map.get(doc.page_content, "")
            matched_doc = Document(page_content=content)
            matched_documents.append(matched_doc)

        # Remove duplicates and limit to top k results
        unique_documents = list({doc.page_content: doc for doc in matched_documents}.values())
        return unique_documents[:self._k]

    def get_vectorstore(self):
        return self._vectorstore



# v2版本，将关键词拼在一起再embedding
class KeywordVectorRetriever_v2(BaseRetriever):
    """A custom retriever that encodes keywords and retrieves documents based on vector similarity."""

    _keyword_file_path: str = PrivateAttr()
    _k: int = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()
    _keyword_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, keyword_file_path: str, k: int = 5):
        """Initialize the retriever, load keywords from the specified file path, and create vector store."""
        super().__init__()
        object.__setattr__(self, '_keyword_file_path', keyword_file_path)
        object.__setattr__(self, '_k', k)
        object.__setattr__(self, '_keyword_map', self._load_keywords())

        # Encode keywords and create vector store
        documents = self._create_keyword_documents()
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
        object.__setattr__(self, '_vectorstore', vectorstore)

    def _load_keywords(self) -> Dict[str, str]:
        """Load keywords and associated documents from the specified CSV file."""
        keyword_map = {}
        with open(self._keyword_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc="Loading keywords"):
                keywords = row['keywords']
                keyword_map[keywords.lower()] = row['introduction']
        return keyword_map

    def _create_keyword_documents(self) -> List[Document]:
        """Create documents from keywords for vector encoding."""
        documents = []
        for keywords, content in tqdm(self._keyword_map.items(), desc="Creating keyword documents"):
            doc = Document(page_content=keywords, metadata={'content': content})
            documents.append(doc)
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve relevant documents based on vector similarity."""
        retriever = self._vectorstore.as_retriever()
        keyword_docs = retriever.get_relevant_documents(query, run_manager=run_manager)

        # Retrieve the associated content for the matched keywords
        matched_documents = []
        for doc in keyword_docs:
            content = self._keyword_map.get(doc.page_content, "")
            matched_doc = Document(page_content=content)
            matched_documents.append(matched_doc)

        # Remove duplicates and limit to top k results
        unique_documents = list({doc.page_content: doc for doc in matched_documents}.values())
        return unique_documents[:self._k]

    def get_vectorstore(self):
        return self._vectorstore

# 使用示例：
# retriever = CustomRetriever(directory='/data2/fyy2022/langChain/RAG/doc_group', chunk_size=500, chunk_overlap=100)
# relevant_docs = retriever.get_relevant_documents("某个查询")

from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class HybridRetriever(BaseRetriever):
    def __init__(self, keyword_retriever: KeywordRetriever, vector_retriever: VectorRetriever):
        """初始化混合检索器，包含关键词检索器和向量检索器"""
        super().__init__()
        object.__setattr__(self, 'keyword_retriever', keyword_retriever)
        object.__setattr__(self, 'vector_retriever', vector_retriever)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """根据查询字符串检索相关文档，优先使用关键词检索，若无结果则使用向量检索"""
        # 先进行关键词检索
        keyword_docs = self.keyword_retriever.get_relevant_documents(query, run_manager=run_manager)

        if keyword_docs:
            # 如果关键词检索到结果，直接返回
            return keyword_docs
        else:
            # 如果关键词检索不到结果，则调用向量检索
            return self.vector_retriever.get_relevant_documents(query, run_manager=run_manager)
        




class HybridRetriever_v2(BaseRetriever):
    def __init__(self,keyword_vector_retriever_v2: KeywordVectorRetriever_v2,vector_retriever: VectorRetriever):
        """初始化混合检索器，包含关键词检索器和向量检索器"""
        super().__init__()
        object.__setattr__(self, 'keyword_vector_retriever_v2', keyword_vector_retriever_v2)
        object.__setattr__(self, 'vector_retriever', vector_retriever)


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """根据查询字符串检索相关文档，优先使用关键词检索，若无结果则使用向量检索"""
        # 先进行关键词检索
        keyword_docs = self.keyword_vector_retriever_v2.get_relevant_documents(query, run_manager=run_manager)
        
        if keyword_docs:
            # 如果关键词检索到结果，直接返回
            return keyword_docs
        else:
            # 如果关键词检索不到结果，则调用向量检索
            return self.vector_retriever.get_relevant_documents(query, run_manager=run_manager)
        
from typing import List

# ##简洁版
# class HybridRetriever_v3(BaseRetriever):
#     def __init__(self, keyword_retriever: KeywordRetriever, keyword_vector_retriever_v2: KeywordVectorRetriever_v2, vector_retriever: VectorRetriever):
#         """初始化混合检索器，包含关键词检索器和向量检索器"""
#         super().__init__()
#         object.__setattr__(self, 'keyword_retriever', keyword_retriever)
#         object.__setattr__(self, 'keyword_vector_retriever_v2', keyword_vector_retriever_v2)
#         object.__setattr__(self, 'vector_retriever', vector_retriever)

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
#     ) -> List[Document]:
#         """根据查询字符串检索相关文档，优先使用关键词检索，若无结果则使用向量检索"""
#         # 先进行关键词检索
#         keyword_docs = self.keyword_retriever.get_relevant_documents(query, run_manager=run_manager)

#         keyword_docs_vector = self.keyword_vector_retriever_v2.get_relevant_documents(query, run_manager=run_manager)
        
#         # 合并两个检索结果并使用 page_content 去重
#         combined_docs = list({doc.page_content: doc for doc in keyword_docs + keyword_docs_vector}.values())

#         if combined_docs:
#             # 如果有结果，保留前5条返回
#             return combined_docs[:5]
#         else:
#             # 如果没有结果，调用向量检索
#             vector_docs = self.vector_retriever.get_relevant_documents(query, run_manager=run_manager)
#             return vector_docs[:5]

##详细化输出
from typing import List

class HybridRetriever_v3(BaseRetriever):
    def __init__(self, keyword_retriever: KeywordRetriever, keyword_vector_retriever_v2: KeywordVectorRetriever_v2, vector_retriever: VectorRetriever):
        """初始化混合检索器，包含关键词检索器和向量检索器"""
        super().__init__()
        object.__setattr__(self, 'keyword_retriever', keyword_retriever)
        object.__setattr__(self, 'keyword_vector_retriever_v2', keyword_vector_retriever_v2)
        object.__setattr__(self, 'vector_retriever', vector_retriever)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """根据查询字符串检索相关文档，优先使用关键词检索，若无结果则使用向量检索"""
        # 先进行关键词检索
        keyword_docs = self.keyword_retriever.get_relevant_documents(query, run_manager=run_manager)
        print(f"Keyword Retrieval Results ({len(keyword_docs)} docs):")
        for doc in keyword_docs:
            print(f"- {doc.page_content[:100]}...")  # 只打印前100个字符以节省空间
        
        keyword_docs_vector = self.keyword_vector_retriever_v2.get_relevant_documents(query, run_manager=run_manager)
        print(f"Keyword Vector Retrieval Results ({len(keyword_docs_vector)} docs):")
        for doc in keyword_docs_vector:
            print(f"- {doc.page_content[:100]}...")  # 只打印前100个字符以节省空间

        # 合并两个检索结果并使用 page_content 去重
        combined_docs = list({doc.page_content: doc for doc in keyword_docs + keyword_docs_vector}.values())

        if combined_docs:
            print(f"Combined Results ({len(combined_docs)} docs):")
            for doc in combined_docs[:5]:
                print(f"- {doc.page_content[:100]}...")  # 只打印前100个字符以节省空间
            return combined_docs[:5]
        else:
            # 如果没有结果，调用向量检索
            vector_docs = self.vector_retriever.get_relevant_documents(query, run_manager=run_manager)
            print(f"Vector Retrieval Results ({len(vector_docs)} docs):")
            for doc in vector_docs:
                print(f"- {doc.page_content[:100]}...")  # 只打印前100个字符以节省空间
            return vector_docs[:5]

class KeywordVectorRetriever_v2_single(BaseRetriever):
    """A custom retriever that encodes keywords and retrieves a single document based on vector similarity with a high threshold."""

    _keyword_file_path: str = PrivateAttr()
    _threshold: float = PrivateAttr()
    _vectorstore: Chroma = PrivateAttr()
    _keyword_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, keyword_file_path: str, threshold: float = 0.9):
        """Initialize the retriever, load keywords from the specified file path, and create vector store."""
        super().__init__()
        object.__setattr__(self, '_keyword_file_path', keyword_file_path)
        object.__setattr__(self, '_threshold', threshold)
        object.__setattr__(self, '_keyword_map', self._load_keywords())

        # Encode keywords and create vector store
        documents = self._create_keyword_documents()
        embedding_model = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)
        object.__setattr__(self, '_vectorstore', vectorstore)

    def _load_keywords(self) -> Dict[str, str]:
        """Load keywords and associated documents from the specified CSV file."""
        keyword_map = {}
        with open(self._keyword_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in tqdm(reader, desc="Loading keywords"):
                keywords = row['keywords']
                keyword_map[keywords.lower()] = row['introduction']
        return keyword_map

    def _create_keyword_documents(self) -> List[Document]:
        """Create documents from keywords for vector encoding."""
        documents = []
        for keywords, content in tqdm(self._keyword_map.items(), desc="Creating keyword documents"):
            doc = Document(page_content=keywords, metadata={'content': content})
            documents.append(doc)
        return documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
        """Retrieve the single most relevant document based on vector similarity, with a score above the threshold."""
        retriever = self._vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": self._threshold, "k": 1}
        )
        
        results = retriever.get_relevant_documents(query, run_manager=run_manager)

        if results:
            for result in results:
                print(f"Document: {result.page_content}, Score: {result.metadata}")

            best_doc = results[0]  # Retrieve the top result

            content = self._keyword_map.get(best_doc.page_content, "")
            return [Document(page_content=content)]
            
        return []  # No document exceeds the threshold
    
    

    def get_vectorstore(self):
        return self._vectorstore





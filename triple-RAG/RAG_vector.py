import csv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from LLM import DeepSeek_LLM
from Retrievers import *

# 创建FastAPI应用
app = FastAPI()

# 初始化 single 检索器
single_retriever = KeywordVectorRetriever_v2_single(
    keyword_file_path='/data2/fyy2022/langChain/RAG/keyWords/RAGIndex.csv', 
    threshold=0.85
)

# 初始化其他检索器和 LLM
vectorRetriever = VectorRetriever(
    directory='/data2/fyy2022/langChain/RAG/doc_group',
    chunk_size=1000,  # 可选：默认值为1000
    chunk_overlap=200  # 可选：默认值为200
)
keywordRetriever = KeywordRetriever(keyword_file_path='/data2/fyy2022/langChain/RAG/keyWords/final.csv', k=3)
vector_keywords_retriever = KeywordVectorRetriever_v2(keyword_file_path='/data2/fyy2022/langChain/RAG/keyWords/final.csv', k=3)
retriever = HybridRetriever_v3(keyword_retriever=keywordRetriever, keyword_vector_retriever_v2=vector_keywords_retriever, vector_retriever=vectorRetriever)
llm = DeepSeek_LLM(api_key='sk-b0ee1c46db0d490991f0605d59076c58')

# 定义聊天提示模板
prompt_message = """
你是中山大学信息管理学院的机器人小信，你的答案必须基于context（你的知识库），如果没有就是你不知道的内容，大方承认（我的知识库没有这方面的储备），认真对待校友的问题（question）,简要回答。
question:
{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", prompt_message)])

# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 自定义检索记录函数
def log_retrieval_and_format(docs):
    retrieval_time = datetime.now()
    formatted_docs = format_docs(docs)
    print(f"Retrieval Time: {retrieval_time}")
    print(f"Retrieved Content:\n{formatted_docs}")
    return formatted_docs

# 定义 RAG 链
rag_chain = (
    {"context": retriever | log_retrieval_and_format, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 定义请求模型
class LLMQueryRequest(BaseModel):
    question: str

# 插入数据到CSV文件
def append_to_csv(file_path, keywords, introduction):
    with open(file_path, mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([keywords, introduction])

# 定义POST方法用于RAG链查询
@app.post("/rag_query")
async def rag_query(request: LLMQueryRequest):
    start_time = datetime.now()
    try:
        # 先使用single检索器
        single_result = single_retriever._get_relevant_documents(request.question)
        if single_result:
            # 如果single检索器有结果，格式化并返回与原链一致的格式
            formatted_single_result = format_docs(single_result)
            return JSONResponse(content=formatted_single_result)

        # 如果single检索器无结果，使用RAG链
        response = rag_chain.invoke(request.question)

        # 输出RAG的返回内容
        print("RAG output:", response)

        # 将用户问题和模型回答追加到CSV文件中
        append_to_csv('/data2/fyy2022/langChain/RAG/keyWords/RAGIndex.csv', request.question, response)

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        end_time = datetime.now()
        print(f"RAG query response time: {end_time - start_time}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)

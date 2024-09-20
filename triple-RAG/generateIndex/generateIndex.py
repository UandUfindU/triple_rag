import os
import csv
from LLM import GLMFlash_LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 初始化LLM
llm = GLMFlash_LLM(api_key="50d79333873f56e2ac8029560c32d2aa.pxMNM2qMDk5KdvdQ")

# 定义聊天提示模板
prompt_message = """
提取出TEXT中的中文关键词，尤其注意专有名词，如名称、职称、研究方向等。你的返回有且仅有用空格隔开的关键词
TEXT:
{TEXT}
"""

prompt = ChatPromptTemplate.from_messages([("human", prompt_message)])

# 定义 索引生成 链
rag_chain = (
   { "TEXT": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 定义结果文件夹和CSV文件路径
result_folder = 'result'
csv_file_path = os.path.join(result_folder, 'result.csv')

# 创建result文件夹（如果不存在）
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 定义写入CSV的函数
def write_to_csv(doc_name, index, content):
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writerow(['doc_name', 'index', 'content'])
        
        # 写入内容
        writer.writerow([doc_name, index, content])

# 遍历doc文件夹（包括子文件夹），读取所有文件
doc_folder = 'doc'
for root, dirs, files in os.walk(doc_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 生成索引
        index = rag_chain.invoke({"TEXT": content})
        
        # 写入CSV
        write_to_csv(file_name, index, content)

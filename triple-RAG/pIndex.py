import pandas as pd
import requests
import os
from multiprocessing import Process, Pool
# import torch
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import time
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Union, Optional
from torch.nn import Module
import csv
import json
import re
from lxml import etree
from xml.dom.minidom import parseString
from docx.shared import Pt

#content = ""

def get_prompt(keyValue, content):
    # 生成完整的 prompt
    prompt = f"""
    现有如下输入内容：
    ### 关键词信息(keyValue)：
    {keyValue}
    ### 内容信息(content)：
    {content}
    ### 任务描述：
    请你以中山大学信息管理学院为基础信息，结合内容信息(content)，围绕关键词信息(keyValue)生成20个问题，并返回每个问题对应的回答。其中问题回答请限制在250字以内。
    ### 生成的问题、答案严格用json格式返回，其中字段question表示问题，字段answer表示答案。示例如下：
    {{"question":"科学计量方向代表性的学者有谁？", "answer":"科学计量方向代表性学者有张洋教授、侯剑华教授、李晶副教授等。"}}
    {{"question":"图书馆学知名学者是谁","answer":"图书馆学知名学者是程焕文教授"}}
    ### 开始任务：
    不要返回任何与问题答案无关的内容，仅需严格按上述json格式返回问题及答案，且每一行返回一个结果。
    """
    return prompt


def call_llm_api(prompt):
        url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
        header = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 48f4ac2b17c3552bc66286029614b1fb.wlDFaIhhqtExuS8f"
        }
        data = {
            "model": "glm-4-airx",
            "temperature": 0.95,
            "stream": "false",
            "max_tokens": 8000,
            "top_p": 0.8,
            "messages": [
            {
                "role": "system",
                "content": "You are glm-4, a large language model trained by Zhipu.AI. Follow the user's instructions carefully.",
            },
            {
                "role": "user",
                "content": prompt
            }
            ]
        }
        response = requests.post(url, json=data, headers=header)
        data_json = json.loads(response.text)
        result = data_json["choices"][0]["message"]["content"]
        return result

kHash = {}

with open('resultZZ.csv', 'r', encoding='utf-8') as file:
    # 逐行读取
    for line in file:
    # 移除行尾的换行符，并使用空格切分字符串
        words = line.lstrip('\ufeff').lstrip('"').rstrip('"').strip().replace(", "," ").split(',')
        if len(words)==2:
            #print (words[0])
            #建立倒排索引
            arr = words[1].lstrip('\ufeff').strip().split(' ')
            for i in arr:
                if i in kHash:
                    kHash[i] = kHash[i] + words[0] + "^^"
                else:
                    kHash[i] = words[0] + "^^"

for i in kHash:
    print(i + "\t" + kHash[i].strip())
    content = ""
    arr = kHash[i].strip().rstrip("^^").split("^^")
    for i in arr:
        with open("./allFile/" + i, 'r', encoding='utf-8') as sfile:
            for ln in sfile:
                content = content + ln.strip()
    prompt = get_prompt(i, content)
    result = call_llm_api(prompt)
    print(result)
    


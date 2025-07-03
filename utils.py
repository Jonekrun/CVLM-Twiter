from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import requests
import json


# 数据预处理
def preprocess_texts(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed = []
    for text in texts:
        if not isinstance(text, str):
            continue
        # 去除字符串中的网址
        text = ' '.join([part for part in text.split() if not 'http' in part.lower()])

        # 清洗和分词
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.lower().split()

        # 过滤停用词和短词
        words = [w for w in words if w not in stop_words and len(w) > 2]

        # 词形还原
        words = [lemmatizer.lemmatize(w) for w in words]

        processed.append(words)
    return processed

# 处理deepseek的输出
def process_response(response):
    first_end = response.find("<think>")
    second_end = response.find("</think>", first_end + 1)
    result = response[second_end + len("</think>"):]
    while '\n' in result:
        result = result.replace('\n', '')
    return result

# 发送请求到 Ollama 服务器
def query_ollama(prompt, model="deepseek-r1:14b"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # 如果为 True，则以流式方式返回结果
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}, {response.text}"


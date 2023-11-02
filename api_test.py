import requests
import json
from sys import argv
import magic


server_addr = "http://localhost:12931"


def detect_file_type(file_path):
    try:
        # 创建 magic 实例
        magic_instance = magic.Magic()

        # 使用 magic 实例来检测文件类型
        file_type = magic_instance.from_file(file_path)
        if 'Word' in file_type:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif 'text' in file_type:
            return "text/plain"
        elif 'PDF' in file_type:
            return 'application/pdf'
        elif 'JSON' in file_type:
            return 'application/json'
        else:
            return 'application/octet-stream'
    except Exception as e:
        print(f"文件类型检测失败: {str(e)}")
        return None


def ner(file_path="data/ner/test.pdf"):
    """实体抽取"""

    url = f"{server_addr}/doc_ie/ner"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def re(file_path="data/re/test.pdf"):
    """关系抽取"""

    url = f"{server_addr}/doc_ie/re"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def ae(file_path="data/ae/test.pdf"):
    """属性抽取"""

    url = f"{server_addr}/doc_ie/ae"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def summary(file_path="data/summary/test_2.pdf"):
    """摘要"""

    url = f"{server_addr}/doc_ie/summary"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def keywords(file_path="data/keywords/test_2.pdf"):
    """关键词"""

    url = f"{server_addr}/doc_ie/keywords"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def region(file_path="data/address/test.pdf"):
    """地区识别"""

    url = f"{server_addr}/doc_ie/region"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def sentiment(file_path="data/sentiment/positive.pdf"):
    """情感分析"""

    url = f"{server_addr}/doc_ie/sentiment"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def classificationt(file_path="data/classification/politics/test.pdf"):
    """文本分类"""

    url = f"{server_addr}/doc_ie/classification"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


def similarity(file_path_list=["data/cos/test_1.pdf", "data/cos/test_2.pdf"]):
    """文本相似度比较"""
    url = f"{server_addr}/doc_ie/similarity"
    def get_file_name(file_path): return str(file_path).split('/')[-1]
    files = [
        ('files', (get_file_name(file_path), open(
            file_path, 'rb'), detect_file_type(file_path)))
        for file_path in file_path_list
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers,  files=files)
    print(response.text)


def translate(file_path="data/translate/test.pdf"):
    """文本分类"""

    url = f"{server_addr}/doc_ie/translate"
    file_name = str(file_path).split('/')[-1]

    files = [
        ('file', (file_name, open(file_path, 'rb'), detect_file_type(file_path)))
    ]
    headers = {}
    response = requests.request(
        "POST", url, headers=headers, files=files)
    print(response.text)


if __name__ == "__main__":
    test_branch = str(argv[1]) if len(argv) > 1 else ""
    print(f"test_branch: {test_branch}")
    if test_branch == "ner":
        ner()
    elif test_branch == "re":
        re()
    elif test_branch == "ae":
        ae()
    elif test_branch == "summary":
        summary()
    elif test_branch == "keywords":
        keywords()
    elif test_branch == "region":
        region()
    elif test_branch == "sentiment":
        sentiment()
    elif test_branch == "classification":
        classificationt()
    elif test_branch == "similarity":
        similarity()
    elif test_branch == "translate":
        translate()
    else:
        raise NotImplementedError

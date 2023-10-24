import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi_utils.api_model import APIModel
import time
import os

from IE.ner import ner
# from IE.relation import relation_extraction
# from IE.attribute_extraction import attribute_extraction
# from IE.summary import summary
# from IE.keywords import keywords_extraction
# from IE.region_identifier import region_extraction
# from IE.sentiment_analysis import sentiment_analysis
# from IE.text_classification import text_classification
# from IE.passage_cos import file_cos
# from IE.machine_translation import tranlate


app = FastAPI()


async def save_file(file: UploadFile) -> str:
    """Store uploaded file stream into local server

    Args:
        file (UploadFile): file to be extracted, like PDF, Word, TXT, etc.

    Returns:
        str: local path to read this file
    """
    time_stamp = int(time.time())
    file_prefix = '.'.join(str(file.filename).split('.')[:-1])
    file_suffix = str(file.filename).split('.')[-1]
    if not os.path.exists('tmp'):
        os.mkdir('tmp/')
    file_save_path = f"tmp/{file_prefix}_{time_stamp}.{file_suffix}"
    file_type = file.content_type  # unused
    file_content = await file.read()
    with open(file.filename, "wb") as f:
        f.write(file_content)
    return file_save_path
        

async def doc_ner(file: UploadFile = File(...)):
    file_path = save_file(file)
    result = ner(file_path)  # TODO: try catch
    return {'result': result}


async def test2(apiname, request: Request):
    """
    /test/<apiname>：后面的apiname表示任意名字
    """

    args = await request.json()

    return {'result': f'这是一个POST，您请求的是：{apiname}，您的参数是：{args}'}



# print(relation_extraction("data/re/yanbao007.txt"))
# print(attribute_extraction("data/re/yanbao007.txt"))
# print(summary("data/keywords/test.txt"))
# print(keywords_extraction("data/keywords/test.txt"))
# print(region_extraction("data/address/news.txt"))
# print(sentiment_analysis("data/sentiment/negative.txt"))
# print(text_classification("data/classification/politics/test.txt"))
# print(file_cos("data/cos/党委理论学习中心组学习材料汇编2023年第16期.pdf",
#       "data/cos/党委理论学习中心组学习材料汇编2023年第15期.pdf"))
# print(tranlate("data/translate/test.txt"))

app.post("/doc_ie/ner", tags=["IE"], summary="单文档命名实体识别")(doc_ner)
app.post("/doc_ie/re", tags=["IE"], summary="单文档关系抽取")(test2)


if __name__ == '__main__':
    uvicorn.run(
        app = app,
        host = "localhost",
        port = 8000
    )

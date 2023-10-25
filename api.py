import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi_utils.api_model import APIModel
from fastapi.responses import RedirectResponse
import time
import os
from typing import List

from IE.ner import ner
from IE.relation import relation_extraction
from IE.attribute_extraction import attribute_extraction
from IE.summary import summary
from IE.keywords import keywords_extraction
from IE.region_identifier import region_extraction
from IE.sentiment_analysis import sentiment_analysis
from IE.text_classification import text_classification
from IE.passage_cos import file_cos
from IE.machine_translation import tranlate

PORT = 12931
app = FastAPI()


@app.get("/")
async def redirect():
    """
    Show Restful API doc in explorer, by redirect to doc route
    Returns:
        doc website
    """
    response = RedirectResponse(url=f"http://localhost:{PORT}/docs/")
    return response


async def save_file(file: UploadFile) -> str:
    """
        Store uploaded file stream into local server
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
    with open(file_save_path, "wb") as f:
        f.write(file_content)
    return file_save_path

# 实体抽取


async def doc_ner(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = ner(file_path)  # TODO: try catch
    return {'result': result}

# 关系抽取


async def doc_ee(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = relation_extraction(file_path)  # TODO: try catch
    return {'result': result}

# 属性抽取


async def doc_ae(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = attribute_extraction(file_path)  # TODO: try catch
    return {'result': result}

# 摘要


async def doc_summary(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = summary(file_path)  # TODO: try catch
    return {'result': result}

# 关键词抽取


async def doc_keywords(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = keywords_extraction(file_path)  # TODO: try catch
    return {'result': result}

# 地区识别


async def doc_region(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = region_extraction(file_path)  # TODO: try catch
    return {'result': result}

# 情感分析


async def doc_sentiment(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = sentiment_analysis(file_path)  # TODO: try catch
    return {'result': result}

# 文本分类


async def doc_classification(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = text_classification(file_path)  # TODO: try catch
    return {'result': result}

# 文章相似度比较


async def doc_similarity(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_path = await save_file(file)
        file_paths.append(file_path)

    results = file_cos(file_paths[0], file_paths[1])  # TODO: try catch
    return {'results': results}

# 翻译


async def doc_translate(file: UploadFile = File(...)):
    file_path = await save_file(file)
    result = tranlate(file_path)  # TODO: try catch
    return {'result': result}


app.post("/doc_ie/ner", tags=["IE"], summary="单文档命名实体识别")(doc_ner)
app.post("/doc_ie/re", tags=["IE"], summary="单文档关系抽取")(doc_ee)
app.post("/doc_ie/ae", tags=["IE"], summary="单文档属性抽取")(doc_ae)
app.post("/doc_ie/summary", tags=["IE"], summary="单文档摘要")(doc_summary)
app.post("/doc_ie/keywords", tags=["IE"], summary="单文档关键词抽取")(doc_keywords)
app.post("/doc_ie/region", tags=["IE"], summary="单文档地区识别")(doc_region)
app.post("/doc_ie/sentiment", tags=["IE"], summary="单文档情感分析")(doc_sentiment)
app.post("/doc_ie/classification",
         tags=["IE"], summary="单文档文本分类")(doc_classification)
app.post("/doc_ie/similarity", tags=["IE"], summary="文档相似度比较")(doc_similarity)
app.post("/doc_ie/translate", tags=["IE"], summary="单文档翻译")(doc_translate)

if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="localhost",
        port=PORT
    )

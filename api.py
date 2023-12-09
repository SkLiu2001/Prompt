import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Body
from fastapi_utils.api_model import APIModel
from fastapi.responses import RedirectResponse
import os
import json
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
from IE.paper_read import paper_read
from units.calculate_md5 import calculate_md5
from units.load_data import load_data
from units.load_data import lazy_load_data
from units.del_tmp import delete_files_in_directory
from units.abstract_model import AbstractModel
PORT = 12931
app = FastAPI()
process_exception = HTTPException(
    status_code=422, detail="Process failed, please check your input.")


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
    # time_stamp = int(time.time())
    file_content = await file.read()
    md5 = await calculate_md5(file_content)
    file_prefix = '.'.join(str(file.filename).split('.')[:-1])
    file_suffix = str(file.filename).split('.')[-1]
    if not os.path.exists('tmp'):
        os.mkdir('tmp/')
    file_save_path = f"tmp/{file_prefix}_{md5}.{file_suffix}"
    file_type = file.content_type
    with open(file_save_path, "wb") as f:
        f.write(file_content)
    return file_save_path, file_type

# 实体抽取


async def doc_ner(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await ner(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 关系抽取


async def doc_ee(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await relation_extraction(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception
# 属性抽取


async def doc_ae(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await attribute_extraction(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 摘要


async def doc_summary(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await summary(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 关键词抽取


async def doc_keywords(file: UploadFile = File(...), ):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await keywords_extraction(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 地区识别


async def doc_region(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await region_extraction(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 情感分析


async def doc_sentiment(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await sentiment_analysis(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 文本分类


async def doc_classification(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await text_classification(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 文章相似度比较


async def doc_similarity(files: List[UploadFile] = File(...)):
    try:
        file_paths = []
        file_types = []
        for file in files:
            file_path, file_type = await save_file(file)
            file_paths.append(file_path)
            file_types.append(file_type)
        pages1 = await lazy_load_data(file_paths[0], file_types[0])
        pages2 = await lazy_load_data(file_paths[1], file_types[1])
        results = await file_cos(pages1=pages1, pages2=pages2)
        delete_files_in_directory('tmp')
        return {'results': results}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception

# 翻译


async def doc_translate(file: UploadFile = File(...)):
    max_pages = 5
    try:
        file_path, file_type = await save_file(file)
        data = await load_data(file_path, file_type, max_pages=max_pages)
        result = await tranlate(data)
        delete_files_in_directory('tmp')
        return {'result': result}
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail="File type %s is not a valid file type" % file_type)
    except Exception as e:
        raise process_exception


async def doc_paper_read(data: AbstractModel = Body(...)):
    prompt = []
    with open('prompt/read_paper.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            prompt.append(eval(line.strip()))
    result = await paper_read(data.content, prompt)
    return {'result': result}
    # except Exception as e:
    #     raise process_exception

app.post("/doc_ie/ner", tags=["IE"], summary="单文档命名实体识别")(doc_ner)
app.post("/doc_ie/re", tags=["IE"], summary="单文档关系抽取")(doc_ee)
app.post("/doc_ie/ae", tags=["IE"], summary="单文档属性抽取")(doc_ae)
app.post("/doc_ie/summary", tags=["IE"], summary="单文档摘要")(doc_summary)
app.post("/doc_ie/keywords", tags=["IE"], summary="单文档关键词抽取")(doc_keywords)
app.post("/doc_ie/region", tags=["IE"], summary="单文档地区识别")(doc_region)
app.post("/doc_ie/sentiment", tags=["IE"],
         summary="单文档情感分析")(doc_sentiment)
app.post("/doc_ie/classification",
         tags=["IE"], summary="单文档文本分类")(doc_classification)
app.post("/doc_ie/similarity", tags=["IE"], summary="文档相似度比较")(doc_similarity)
app.post("/doc_ie/translate", tags=["IE"], summary="单文档翻译")(doc_translate)
app.post("/doc_ie/paper_read", tags=["IE"], summary="论文速读")(doc_paper_read)

if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="localhost",
        port=PORT
    )

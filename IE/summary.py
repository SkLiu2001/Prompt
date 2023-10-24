# 摘要
import json
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from units.merge_json import merge_json
from tqdm import tqdm


def summary(path):

    model = "Qwen-14B-Chat-Int4"

    examples = [
        {
            "input": '''新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。''',
            "output": '''{"summary": "修改后的立法法全文公布"}'''
        },
        {
            "input": '''一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。''',
            "output": '''{"summary": "深圳机场9死24伤续：司机全责赔偿或超千万"}'''
        },
        {
            "input": '''1月18日，习近平总书记对政法工作作出重要指示：2014年，政法战线各项工作特别是改革工作取得新成效。新形势下，希望全国政法机关主动适应新形势，为公正司法和提高执法司法公信力提供有力制度保障。''',
            "output": '''{'summary': "孟建柱：主动适应形势新变化提高政法机关服务大局的能力"}'''
        }

    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             '''对输入的文本进行**摘要**, 要求尽可能简洁，输出结果在200字以内，并以json格式输出，格式为：{{"summary": "摘要内容"}}'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             '''目前已知分段分本的摘要,请你根据上述所有摘要生成新的摘要，要求尽可能简洁，输出结果在200字以内，并以json格式输出，格式为：{{"summary": "摘要内容"}}'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=16)
    chain = LLMChain(
        prompt=final_prompt,
        # 温度调为0，可以保证输出的结果是确定的
        llm=ChatOpenAI(
            temperature=0,
            model_name=model,
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1")
        # output_parser=output_parser
    )
    map = ""
    tmp = ""
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = chain(
                    {"input": text}, return_only_outputs=True)['text']
                # print(tmp)
                try:
                    map += tmp
                except Exception as e:
                    continue
            pbar.update(1)
        # print(map)
    res = chain({"input": map}, return_only_outputs=True)['text']
    try:
        json_object = json.loads(res, strict=False)
        return json_object
    except Exception as e:
        return {"summary": tmp}

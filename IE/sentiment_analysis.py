# 情感分析
import json
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from units.merge_json import merge_json


def sentiment_analysis(path):

    model = "Qwen-14B-Chat-Int4"

    examples = [
        {"input": "买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质",
         "output": '''{"sentiment_list":[{"seniment": "positive"}]}'''},
        {"input": "手机不好，不喜欢，就是快递有点慢，不满意",
         "output": '''{"sentiment_list":[{"seniment": "negative"}]}'''},

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
            ("system", '''你现在需要完成一个情感分类的任务，情感的类型包含"积极"和"消极"两种''' +
             '''分类的结构请用json的形式展示'''),
            few_shot_prompt,
            ("human", "{input}")
        ]
    )
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256, chunk_overlap=16)
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

    texts = text_splitter.split_text(data)
    merged_json = {"sentiment_list": []}
    for text in texts:
        print("----------------------------------")
        print(text)
        tmp = chain(
            {"input": text}, return_only_outputs=True)['text']
        try:
            json_object = json.loads(tmp)
        except ValueError as e:
            continue
        print(json_object)
        merged_json = merge_json(merged_json, json_object)
        print("----------------------------------")
    return merged_json

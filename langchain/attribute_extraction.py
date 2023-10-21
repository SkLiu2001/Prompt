# 属性抽取
import json
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
model = "Qwen-14B-Chat-Int4"

examples = [
    {"input": "雅生活服务（03319.HK）成立于1992年，是开发商雅居乐集团的附属公司。2017年公司收购绿地物业并引入绿地控股（600606.SH）成为其长期战略股东。2018 年 2月公司从雅居乐集团拆分后在港交所上市。公司业务涉及住宅物业服务、高端商写资产管理、公共物业服务、社区商业。按照公司收入来源划分，主营业务分为物业管理服务。",
     "output": '''[{"name":"成立时间", "content":"1992年"},{"name":"收购时间", "content":"2017年"},{"name":"上市时间", "content":"2018年2月"},{"name":"主营业务", "content":"物业管理服务"}]'''},
    {"input": '''公司主要聚焦中高端物业，管理的资产类别有住宅物业（包括旅游地产）和非住宅物业（包括商用物业，写字楼和综合体）两大类。截止 2019H1，公司在管面积中住宅类业态占比 58.7%，非住宅类业态占比 41.3%，其中非住宅类占比增加是因为公司通过收并购项目的非住宅业态占比较高''',
     "output": '''[{"name":"住宅类业态占比", "content":"58.7%"},{"name":"非住宅类业态占比", "content":"41.3%"}'''}
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
        ("system", '''你现在需要完成一个**属性抽取**任务，尽可能地抽**数据信息**''' +
         '''抽取的属性请用json的形式展示，其中json的第一个元素为**属性名称**，第二个元素为**属性的具体内容**。
输出格式形为：[{{"name": "属性名称1", "content": "属性的具体内容2"}}, {{"name": "属性名称2", "content": "属性的具体内容2"]。除了这个列表以外请不要输出别的多余的话。'''),
        few_shot_prompt,
        ("human", "{input}")
    ]
)

path = "data/re/"
with open(path+"yanbao007.txt", "r", encoding="utf-8") as f:
    data = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=32)
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

# # 单句测试
# print(final_prompt.format(
#     input="江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心"))
# tmp = chain({"input": "江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心"},
#             return_only_outputs=True)['text']
# print(tmp)


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True


texts = text_splitter.split_text(data)
with open("result.json", "w", encoding="utf-8") as f:
    for text in texts:
        print("----------------------------------")
        print(text)
        tmp = chain(
            {"input": text}, return_only_outputs=True)['text']
        print(tmp)
        try:
            json_object = json.loads(tmp)
        except ValueError as e:
            continue
        json_object = json.loads(tmp)
        json.dump(json_object, f, indent=4, ensure_ascii=False)
        print("----------------------------------")

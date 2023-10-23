# 摘要
import json
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
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
        # few_shot_prompt,
        ("human", "{input}")
    ]
)

path = "report.txt"
with open(path, "r", encoding="utf-8") as f:
    data = f.read()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048, chunk_overlap=16)
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
refine = ""
with open("result.json", "w", encoding="utf-8") as f:
    for text in texts:
        print("----------------------------------")
        print(text)
        tmp = chain(
            {"input": refine[12:]+text}, return_only_outputs=True)['text']
        refine += tmp
        print(tmp)
        try:
            json_object = json.loads(tmp)
        except ValueError as e:
            continue
        json_object = json.loads(tmp)
        json.dump(json_object, f, indent=4, ensure_ascii=False)
        print("----------------------------------")

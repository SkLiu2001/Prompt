import openai
import json
from langchain.document_loaders import PyPDFLoader
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
# openai.api_key = "EMPTY"
# openai.api_base = "http://182.254.164.88:8000/v1"
model = "Qwen-14B-Chat-Int4"

# template = '''假设你是一个命名实体识别模型，现在我会给你一个句子，请根据我的要求按顺序识别出每个句子中的实体，禁止对实体进行重复输出
# 实体类型只要三种：党政机关、地点、人物。请用json的形式展示，其中json的第一个元素为实体名称，第二个元素为实体类型。
# 如果该句子中不含有指定的实体类型，你可以输出:[]。
# 输出格式形为：[{{name:"实体名称1",type:"实体类型1"}}, {{name:"实体名称2", type:"实体类型2"}}]。除了这个列表以外请不要输出别的多余的话。
# 这个句子是：
# "叶利钦总统和夫人亲娜稳步走下脑梯，踏上东道主专为贵宾铺设的红地毯，司前来迎接的中国政府陪同团团长、财政部部长刘仲蔡，中国驻俄罗斯大使李凤林，外交部副部长张德广等热情握手。"
# [{{name:"叶利钦总统", type:"人物"}}, {{name:"亲娜",type:"人物"}}, {{name:"中国政府陪同团团长", type:"党政机关"}},
# {{name:"财政部部长", type:"党政机关"}}, {{name:"中国驻俄罗斯大使",type: "党政机关"}}, {{name:"外交部副部长", type:"党政机关"}}]
# 假设你是一个命名实体识别模型，现在我会给你一个句子，请根据我的要求按顺序识别出每个句子中的实体，禁止对实体进行重复输出
# 实体类型只要三种：政府机关、地点、人物。请用json的形式展示，其中json的第一个元素为实体名称，第二个元素为实体类型。
# 如果该句子中不含有指定的实体类型，你可以输出:[]。
# 输出格式形为：[{{name:"实体名称1",type:"实体类型1"}}, {{name:"实体名称2", type:"实体类型2"}}]。除了这个列表以外请不要输出别的多余的话。
# 这个句子是：
# {sentences}'''
examples = [
    {"input": "叶利钦总统和夫人亲娜稳步走下脑梯，踏上东道主专为贵宾铺设的红地毯，司前来迎接的中国政府陪同团团长、财政部部长刘仲蔡，中国驻俄罗斯大使李凤林，外交部副部长张德广等热情握手。",
     "output": '''[{name:"叶利钦总统", type:"人物"}, {name:"亲娜",type:"人物"}, {name:"中国政府陪同团团长", type:"党政机关"},{name: "财政部部长", type: "党政机关"}, {name: "中国驻俄罗斯大使", type: "党政机关"}, {name: "外交部副部长", type: "党政机关"}]'''},
    {"input": '''刚刚过去的一年,大气磅礴,波澜壮阔。在这一年,以江泽民同志为核心的党中央,继承邓小平同志的遗志,高举邓小平理论的伟大旗帜,领导全党和全国各族人民坚定不移地沿着建设有中国特色社会主义道路阔步前进,
写下了改革开放和社会主义现代化建设的辉煌篇章。顺利地恢复对香港行使主权,胜利地召开党的第十五次全国代表大会———两件大事办得圆满成功。
国民经济稳中求进,国家经济实力进一步增强,人民生活继续改善,对外经济技术交流日益扩大。在国际金融危机的风浪波及许多国家的情况下,我国保持了金融形势和整个经济形势的稳定发展。
社会主义精神文明建设和民主法制建设取得新的成绩,各项社会事业全面进步。外交工作取得可喜的突破,我国的国际地位和国际威望进一步提高。
实践使亿万人民对邓小平理论更加信仰,对以江泽民同志为核心的党中央更加信赖,对伟大祖国的光辉前景更加充满信心。''',
     "output": '''[{name:"江泽民", type:"人物"},{name:"党中央",type:"党政机关"},{name:"邓小平",type:"人物"}, {name:"中国",type:"地点"},{name:"香港",type:"地点"}]'''}
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
print(few_shot_prompt.format())
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''假设你是一个命名实体识别模型，现在我会给你一个句子，请根据我的要求按顺序识别出每个句子中的实体，禁止对实体进行重复输出
实体类型只要三种：党政机关、地点、人物。请用json的形式展示，其中json的第一个元素为实体名称，第二个元素为实体类型。如果该句子中不含有指定的实体类型，你可以输出: []。    输出格式形为：[{{name: "实体名称1", type: "实体类型1"}}, {{name: "实体名称2", type: "实体类型2"}}]。除了这个列表以外请不要输出别的多余的话。'''),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# prompt = PromptTemplate(  # 设置prompt模板，用于格式化输入
#     template=template,
#     input_variables=["sentences"]
# )


path = "data/cos/"
loader_first = PyPDFLoader(path+"党委理论学习中心组学习材料汇编2023年第16期.pdf")
pages_first = loader_first.load_and_split()
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

# # 单句测试
# print(final_prompt.format(
#     input="江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心"))
# tmp = chain({"input": "江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心"},
#             return_only_outputs=True)['text']
# print(tmp)

texts = text_splitter.split_text(pages_first[1].page_content)
for text in texts:
    print(text)
    tmp = chain(
        {"input": text}, return_only_outputs=True)['text']
    print(tmp)


# json_data = json.loads(tmp)
# with open("result/ner.json", "w", encoding="utf-8") as f:
#     json.dump(json_data, f, ensure_ascii=False, indent=4)

# create a completion
# print(template.format(
#     sentences="江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心"))
# completion = openai.Completion.create(
#     model=model, prompt=template.format(
#         sentences="江主席的贺辞说，中俄建设面向 2 1世纪的战略协作伙伴关系，无论在中国，还是在俄罗斯，都有着广泛的社会基础，中俄长期友好合作的思想日益深入人心", max_tokens=2048, temperature=0.2))
# # print the completion
# print(completion.choices[0].text)

# create a chat completion
# completion = openai.ChatCompletion.create(
#     model=model,
#     messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# print the completion
# print(completion.choices[0].message.content)

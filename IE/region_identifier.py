# 地区抽取
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
model = "Qwen-14B-Chat-Int4"

examples = [
    {"input": "本报讯　9月10日，铜陵市检察院在正式受理公安机关移送的教师周恒绑架杀害自己学生一案(本报曾报道)。当天，这起恶性案件被提起公诉。",
     "output": '''[{"province": "安徽省", "city": "铜陵市","address":"检察院"}]'''},
    {"input": '''商丘市邮政速递公司一陈姓负责人在接受采访时说：“只要邮件投递到有收发室的单位，收发室人员签收后，必须将邮件转交收件人本人，不得延误，如无法转交，应及时退还邮政局。”该负责人说，夏邑县一高签收李奎鹏的邮件后，应当把邮件转交给李，如没有及时转交，夏邑县一高负有责任。''',
     "output": '''[{"province": "河南省", "city": "商丘市","address":"快递公司"}],[{"province": "河南省", "city": "夏邑县","address":"一高"}]]'''}
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
        ("system", '''你现在需要完成一个**地区识别**任务，要求按**省、市、地区**三种层次进行输出''' +
         '''抽取的属性请用json的形式展示，json格式如下。[{{"province": "省份名称", "city": "城市名称","address":"具体地址"}}, {{"province": "省份名称", "city": "城市名称","address":"具体地址"}}]。除了这个列表以外请不要输出别的多余的话。
         如果地区中缺少某种层次的地区，请用“无”来表示。例如：[{{"province": "山东省", "city": "济南市","address":"无"}}
         如果已知低层次的地区，但是不知道高层次的地区，请根据自身地知识进行填充。例如：济南市需要识别为[{{"province": "山东省", "city": "济南市","address":"无'''),
        few_shot_prompt,
        ("human", "{input}")
    ]
)

path = "data/address/"
with open(path+"news.txt", "r", encoding="utf-8") as f:
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

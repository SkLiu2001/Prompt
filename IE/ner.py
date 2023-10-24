import openai
import json
from langchain.document_loaders import PyPDFLoader
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
from tqdm import tqdm
model = "Qwen-14B-Chat-Int4"


def ner(path):
    examples = [
        {
            "input": "叶利钦总统和夫人亲娜稳步走下脑梯，踏上东道主专为贵宾铺设的红地毯，司前来迎接的中国政府陪同团团长、财政部部长刘仲蔡，中国驻俄罗斯大使李凤林，外交部副部长张德广等热情握手。",
            "output": '''{"named_entities":[{"name":"叶利钦总统", "type":"人物"}, {"name":"亲娜","type":"人物"}, {"name":"中国政府陪同团团长", "type":"党政机关"},{"name": "财政部部长", "type": "党政机关"}, {"name": "中国驻俄罗斯大使", "type": "党政机关"}, {"name": "外交部副部长", "type": "党政机关"}]}'''
        },
        {
            "input": '''刚刚过去的一年,大气磅礴,波澜壮阔。在这一年,以江泽民同志为核心的党中央,继承邓小平同志的遗志,高举邓小平理论的伟大旗帜,领导全党和全国各族人民坚定不移地沿着建设有中国特色社会主义道路阔步前进,
写下了改革开放和社会主义现代化建设的辉煌篇章。顺利地恢复对香港行使主权,胜利地召开党的第十五次全国代表大会———两件大事办得圆满成功。
国民经济稳中求进,国家经济实力进一步增强,人民生活继续改善,对外经济技术交流日益扩大。在国际金融危机的风浪波及许多国家的情况下,我国保持了金融形势和整个经济形势的稳定发展。
社会主义精神文明建设和民主法制建设取得新的成绩,各项社会事业全面进步。外交工作取得可喜的突破,我国的国际地位和国际威望进一步提高。
实践使亿万人民对邓小平理论更加信仰,对以江泽民同志为核心的党中央更加信赖,对伟大祖国的光辉前景更加充满信心。''',
            "output": '''{"named_entities":[{"name":"江泽民", "type":"人物"},{"name":"党中央","type":"党政机关"},{"name":"邓小平","type":"人物"}, {"name":"中国","type":"地点"},{"name":"香港","type":"地点"}]}'''
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
            ("system", '''假设你是一个命名实体识别模型，现在我会给你一个句子，请根据我的要求按顺序识别出每个句子中的实体，禁止对实体进行重复输出
    实体类型只要三种：党政机关、地点、人物。请用json的形式展示，其中json的第一个元素为实体名称，第二个元素为实体类型。如果该句子中不含有指定的实体类型，你可以输出: []'''),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    # path = "data/cos/"
    loader_first = PyPDFLoader(path)
    pages = loader_first.load_and_split()
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
    merged_json = {"named_entities": []}
    with tqdm(total=len(pages)) as pbar:
        pbar.set_description('Processing:')
        for page in pages:
            texts = text_splitter.split_text(page.page_content)
            for text in texts:
                tmp = chain(
                    {"input": text}, return_only_outputs=True)['text']
                # print(tmp)
                try:
                    json_object = json.loads(tmp, strict=False)
                    merged_json = merge_json(merged_json, json_object)
                except Exception as e:
                    continue
            pbar.update(1)
    return merged_json

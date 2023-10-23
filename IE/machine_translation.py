# 机器翻译
import json
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
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
    {
        "input": '''A boy named Li Ming is reading a science fiction novel. He is attracted by the peculiar world within and the breathtaking imagination. Every character, every scene, seems to vividly present itself in his mind. Li Ming enjoys immersing himself in the world of the novel, as if he could travel to another universe filled with mystery.''',
        "output": '''{"result": "一位名叫李明的男孩正在读一本科幻小说。他被其中的奇特世界和令人叹为观止的想象力所吸引。每个角色，每个场景，似乎都在他的头脑中形象地展现出来。李明喜欢这样沉浸在小说的世界里，仿佛他可以穿越到另一个充满神秘的宇宙。"}'''
    },
    {
        "input": '''おはようございます、李さん。今日は天気がいいですね、散歩に出かけるのに適しています。公園で散歩するのに興味はありますか？''',
        "output": '''{"result": "早上好，李先生。今天天气真好，适合出去散步。你对在公园散步有兴趣吗？"}'''
    },
    {
        "input": '''Dans cette belle ville, il y a de grands parcs verts et de vastes plaines. Chaque aube et chaque coucher de soleil sont extraordinaires. Se promener dans le parc, on peut entendre le chant des oiseaux, ressentir les cadeaux de la nature. Les habitants de cette ville sont très amicaux, ils accueillent toujours chaleureusement les visiteurs du monde entier. Cette ville, c'est chez moi, j'aime chaque brin d'herbe, chaque montagne et chaque goutte d'eau ici.''',
        "output": "{'result': '在这座美丽的城市里，有大片绿色的公园和广袤的平原。每一抹日出和日落都分外壮丽。去公园里散步，可以听到鸟儿的歌唱，感受自然的馈赠。这座城市的居民非常友好，他们总是热情欢迎世界的访客。这座城市，就是我的家，我爱这里的一草一木，一山一水'}"
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
        ("system", '''你现在需要完成一个**机器翻译**的任务，你需要将输入的**其他语言**的文本翻译为**中文**文本''' +
         '''翻译的结构请用json的形式展示，
输出格式形为：{{"result": "翻译结果"}}。除了这个json以外请不要输出别的多余的话。'''),
        few_shot_prompt,
        ("human", "{input}")
    ]
)

path = "chatle.txt"
with open(path, "r", encoding="utf-8") as f:
    data = f.read()
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

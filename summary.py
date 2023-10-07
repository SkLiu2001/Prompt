from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'sk-H2nBsaKMbIGSh2uSl7QvKtGttNPGeOacPUeqy2fXJOr58AhP'
os.environ["OPENAI_API_BASE"] = 'https://api.chatanywhere.com.cn/v1'
llm = ChatOpenAI(temperature=0,
                 model_name='gpt-3.5-turbo',
                 openai_api_key='sk-H2nBsaKMbIGSh2uSl7QvKtGttNPGeOacPUeqy2fXJOr58AhP',
                 openai_api_base='https://api.chatanywhere.com.cn/v1')

with open("report.txt", encoding='utf-8') as f:
    report_2023 = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)

texts = text_splitter.split_text(report_2023)

docs = [Document(page_content=t) for t in texts]
prompt_template = """Write a concise summary in Chinese of the following and Give key sentences in the original text:
{text}
CONCISE SUMMARY:
key sentences:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary and And put all the key sentences together\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in chinese"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(llm, chain_type="refine",
                             return_intermediate_steps=True, question_prompt=prompt,
                             refine_prompt=refine_prompt,)
summ = chain({"input_documents": docs}, return_only_outputs=True)
print(summ["output_text"])

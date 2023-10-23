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
# print(ner("data/cos/党委理论学习中心组学习材料汇编2023年第16期.pdf"))
# print(relation_extraction("data/re/yanbao007.txt"))
# print(attribute_extraction("data/re/yanbao007.txt"))
# print(summary("data/keywords/test.txt"))
# print(keywords_extraction("data/keywords/test.txt"))
# print(region_extraction("data/address/news.txt"))
# print(sentiment_analysis("data/sentiment/negative.txt"))
# print(text_classification("data/classification/politics/test.txt"))
# print(file_cos("data/cos/党委理论学习中心组学习材料汇编2023年第16期.pdf",
#       "data/cos/党委理论学习中心组学习材料汇编2023年第15期.pdf"))
print(tranlate("data/translate/test.txt"))

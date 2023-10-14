import json


def dict2BIO(data):
    text = data['text']
    entity_list = data['entity_list']
    label_list = ['O'] * len(text)
    for entity in entity_list:
        start = entity['entity_index']['begin']
        end = entity['entity_index']['end']
        label_list[start] = 'B-' + entity['entity_type']
        for i in range(start + 1, end):
            label_list[i] = 'I-' + entity['entity_type']
    return label_list


label_list = []
text_list = []
with open('data/ner/people_daily.txt', encoding='UTF-8') as f:
    for line in f:
        tmp = json.loads(line)
        labels = dict2BIO(tmp)
        label_list.append(labels)
print(label_list[0])
# with open('data/ner/people_daily_final.txt', 'w', encoding='UTF-8') as f:
#     for data in data_list:
#         f.write(json.dumps(data, ensure_ascii=False) + '\n')

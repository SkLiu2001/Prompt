def merge_json(json_1, json_2):
    key = list(json_1.keys())[0]
    # 获取每个 JSON 中 named_entities 的列表
    entities_1 = json_1[key]
    entities_2 = json_2[key]

    # 把第二个列表变成第一个列表的一部分
    entities_1.extend(entities_2)

    # 现在 entities_1 包含所有的实体
    merged_json = {key: entities_1}
    return merged_json

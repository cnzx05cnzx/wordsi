import csv
import json
import pandas as pd


def data_read():
    with open('./data/medicial/KUAKE-QQR_test.json', encoding='utf-8') as input_data:
        json_content = json.load(input_data)
        # 逐条读取记录
        word_lists = []
        for block in json_content:
            id = block['id']
            query1 = block['query1']
            query2 = block['query2']
            # label = block['label']
            label = -1
            temp = [id, query1, query2, label]
            word_lists.append(temp)

    with open('./data/medicial/test.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        # 写入列头
        w.writerow(['id', 'q1', 'q2', 'label'])
        for word in word_lists:
            w.writerow([word[0], word[1], word[2], word[3]])


def data_write():
    with open('KUAKE-QQR_test.json') as input_data, open('KUAKE-QQR_test_pred.json', 'w') as output_data:
        json_content = json.load(input_data)
        # 逐条读取记录，并将预测好的label赋值
        for block in json_content:
            query1 = block['query1']
            query2 = block['query2']
            # 此处调用自己的模型来预测当前记录的label，仅做示例用：
            # block['label'] = your_model.predict(query1, query2)
        # 写json文件
        json.dump(json_content, output_data, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    data_read()
    df = pd.read_csv('./data/medicial/dev.csv',encoding='gbk')
    df['l1'] = df['q1'].apply(lambda x: len(x))
    df['l2'] = df['q2'].apply(lambda x: len(x))
    print(df['l1'].describe([.2, .8]))
    print(df['l2'].describe([.2, .8]))

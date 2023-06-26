import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import get_scheduler
import numpy as np
import re
from tqdm.auto import tqdm
import evaluate

# 加载数据
def load_conll_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
            else:
                parts = line.split()
                word = parts[0]
                tag = parts[1]
                word = re.sub(pua_pattern, "", word)  # 删除这些私有域字符
                if word:
                    sentence.append(word)
                    label.append(tag)
        if len(sentence) > 0:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


# 加载测试数据
def load_test_file(file_path):
    sentences = []
    labels = []
    pua_pattern = re.compile("[\uE000-\uF8FF]|[\u200b\u200d\u200e]")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ids, words = line.strip().split('\001')
            # 要预测的数据集没有label，伪造个O，
            words = re.sub(pua_pattern, '', words)
            label = ['O' for x in range(0, len(words))]
            sentence = []
            for c in words:
                sentence.append(c)
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


train_sentences, train_labels = load_conll_file('train.conll')
dev_sentences, dev_labels = load_conll_file('dev.conll')

# 建立tag到id的映射表
tags_list = ['O']
for labels in (train_labels + dev_labels):
    for tag in labels:
        if tag not in tags_list:
            tags_list.append(tag)

tag2id = {tag: i for i, tag in enumerate(tags_list)}
id2tag = {i: tag for i, tag in enumerate(tags_list)}

# print(len(train_sentences), len(dev_sentences), len(tag2id))
# print(train_sentences[0], train_labels[0])
# print(dev_sentences[0], dev_labels[0])
# # print(test_sentences[0],test_labels[0])
# # tag2id
# # print(id2tag)

tokenizer = AutoTokenizer.from_pretrained('C:/Users/小言/Desktop/作业/大课设/lert-base')


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2id):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id

        self.encodings = tokenizer(sentences, is_split_into_words=True, padding=True)

        self.encoded_labels = []
        for label, input_id in zip(labels, self.encodings['input_ids']):
            # create an empty array of 0
            t = len(input_id) - len(label) - 1
            label = ['O'] + label + ['O'] * t
            self.encoded_labels.append([tag2id[l] for l in label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        labels = torch.LongTensor(self.encoded_labels[idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def get_item(self, idx):
        pass
        # sentence = self.sentences[idx]
        # label = self.labels[idx]
        # return {'sentence':sentence,'label': label}


train_dataset = MyDataset(train_sentences, train_labels, tokenizer, tag2id)
eval_dataset = MyDataset(dev_sentences, dev_labels, tokenizer, tag2id)
# test_dataset = MyDataset(test_sentences, test_labels, tokenizer, tag2id)

# train_dataset[0],train_dataset.get_item(0),len(train_dataset[0]['input_ids']),len(train_dataset[0]['labels'])


#定义model
model = AutoModelForTokenClassification.from_pretrained('C:/Users/小言/Desktop/作业/大课设/lert-base', num_labels=len(tag2id))

# 定义Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#如果您可以访问GPU，请指定要使用GPU的设备。否则，在CPU上进行训练可能需要几个小时，而不是几分钟
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#训练model

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for step,batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if step % 100 == 0:
            print(f'Step {step} - Training loss: {loss}')

model.save_pretrained('lert-base')

# 评估函数

metric = evaluate.load('seqeval')

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    for input_id,prediction,label in zip(batch['input_ids'], predictions,batch['labels']):
        index = input_id.tolist().index(102)
        prediction2 = [ id2tag[t.item()]  for t in prediction[1:index]]
        label2 = [ id2tag[t.item()]  for t in label[1:index]]
        metric.add(prediction=prediction2,  reference=label2)

results = metric.compute()
print(results)

test_sentences, test_labels = load_test_file('final_test.txt')
test_dataset = MyDataset(test_sentences, test_labels, tokenizer, tag2id)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# 指定文件名
file_name = "output.txt"

# 打开文件，以写入模式写入数据
with open(file_name, "w", encoding="utf-8") as file:
    i = 1
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        for input_id, prediction in zip(batch['input_ids'], predictions):
            index = input_id.tolist().index(102)
            sentence = tokenizer.decode(input_id[1:index]).replace(" ", "")
            prediction2 = [id2tag[t.item()] for t in prediction[1:index]]
            prediction_str = ' '.join(prediction2)

            line = f"{i}\u0001{sentence}\u0001{prediction_str}\n"
            file.write(line)
            i += 1

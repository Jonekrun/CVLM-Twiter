from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import webbrowser

from utils import preprocess_texts

class Text_features():
    def __init__(self, sample_data, num_topics=10, if_visualize=False):
        # 初始化VADER分析器
        self.analyzer = SentimentIntensityAnalyzer()
        self.if_visualize = if_visualize
        self.id_keywords = self.LDA_features(sample_data, num_topics)

    def vader_sentiment_analysis(self, text):
        # 获取情感得分
        sentiment = self.analyzer.polarity_scores(text)

        # 根据compound分数确定情感标签
        compound_score = sentiment['compound']

        if compound_score >= 0.05:
            label = 1
        elif compound_score <= -0.05:
            label = -1
        else:
            label = 0

        return label

    def LDA_topic_analysis(self, text):
        # 转换为词袋格式
        bow = self.dictionary.doc2bow(list(text))

        # 获取主题分布
        topic_distribution = self.lda_model.get_document_topics(bow, minimum_probability=0)

        # 找出最大概率的主题
        best_topic = max(topic_distribution, key=lambda x: x[1])[0]

        return best_topic

    def LDA_features(self, sample_data, num_topics=5):
        processed_data = preprocess_texts(sample_data)
        # 1 模型构建
        # 构建词典和语料库
        self.dictionary = Dictionary(processed_data)
        corpus = [self.dictionary.doc2bow(text) for text in processed_data]

        # 2 训练LDA模型
        self.lda_model = LdaModel(
            alpha=1 / 3,
            eta=1 / 3,
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=15
        )
        print("Finish training LDA model")

        # 3 可视化分析
        if self.if_visualize:
            # pyLDAvis交互图
            vis = pyLDAvis.gensim_models.prepare(self.lda_model, corpus, self.dictionary)
            pyLDAvis.display(vis)
            pyLDAvis.save_html(vis, 'lda_vis.html')
            webbrowser.open('lda_vis.html')  # 自动调用默认浏览器

            # 词云图
            for topic_num in range(self.lda_model.num_topics):
                words = dict(self.lda_model.show_topic(topic_num, 20))
                wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(words)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis('off')
                plt.title(f"Topic #{topic_num + 1}")
                plt.show()

            # 热力图
            # 获取文档-主题概率矩阵
            doc_topic = []
            for doc in corpus:
                topic_probs = self.lda_model.get_document_topics(doc, minimum_probability=0)
                doc_topic.append([prob for _, prob in topic_probs])

            matrix = np.array(doc_topic)

            # 绘制热力图
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                matrix,
                cmap="YlGnBu",
                xticklabels=[f"Topic {i + 1}" for i in range(self.lda_model.num_topics)],
                yticklabels=[f"Doc {i + 1}" for i in range(len(sample_data))],
                linewidths=0.5
            )
            plt.title("Document-Topic Probability Distribution")
            plt.xlabel("Topics")
            plt.ylabel("Documents")
            plt.show()

        # 主题内容
        num_words = 5
        topics = self.lda_model.print_topics(num_words)
        id_keywords = {}
        for topic in topics:
            id_keywords[topic[0]] = [word.split('*')[1].strip() for word in topic[1].split(' + ')]

        return id_keywords


from transformers import BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch

# 文本分支：BERT + BiLSTM
class TextBranch(nn.Module):
    def __init__(self, model_path=r"checkpoint/bert-base-uncased",
                 lstm_hidden_size=128, num_topics=10, sentiment_dim=1, output_size=256):
        super(TextBranch, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.topic_embedding = nn.Embedding(num_embeddings=num_topics, embedding_dim=32)
        self.topic_fc = nn.Linear(32, 64)
        self.sentiment_fc = nn.Linear(sentiment_dim, 64)
        self.fc = nn.Linear(256 + 64 + 64, output_size)

    def forward(self, input_ids, attention_mask, lengths, topics, sentiments):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        packed = pack_padded_sequence(sequence_output, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(lstm_out.size(0), lstm_out.size(2)).unsqueeze(1)
        final_lstm_out = lstm_out.gather(1, idx.to(lstm_out.device)).squeeze(1)

        topic_emb = self.topic_embedding(topics)
        topic_feat = self.topic_fc(topic_emb)

        sentiment_feat = self.sentiment_fc(sentiments.unsqueeze(1))

        combined = torch.cat([final_lstm_out, topic_feat, sentiment_feat], dim=1)
        text_feat = self.fc(combined)
        return text_feat


if __name__ == '__main__':
    # 示例文本
    sample_texts = [
        "I love this product! It's amazing! 😊",
        "This is okay, but could be better.",
        "I hate this. Worst experience ever! 😠",
        "The weather is neither good nor bad today.",
        "The service was excellent and the staff was very friendly."
    ]

    # 对每个文本进行情感分析
    extractor = Text_features(sample_texts,3, if_visualize=True)

    for text in sample_texts:
        vader_result = extractor.vader_sentiment_analysis(text)
        topic = extractor.LDA_topic_analysis(text)
        print(text, vader_result, topic)


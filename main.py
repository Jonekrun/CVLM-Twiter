import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from Text_features import *

# 图片描述分支：只使用BERT + BiLSTM
class ImageDescriptionBranch(nn.Module):
    def __init__(self, model_path=r"checkpoint/bert-base-uncased",
                 lstm_hidden_size=128, output_size=256):
        super(ImageDescriptionBranch, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(256, output_size)  # 256 = lstm_hidden_size * 2 (bidirectional)

    def forward(self, input_ids, attention_mask, lengths):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        packed = pack_padded_sequence(sequence_output, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(lstm_out.size(0), lstm_out.size(2)).unsqueeze(1)
        final_lstm_out = lstm_out.gather(1, idx.to(lstm_out.device)).squeeze(1)

        image_desc_feat = self.fc(final_lstm_out)
        return image_desc_feat

def load_image_descriptions(file_path):
    """加载图片描述数据"""
    id_descriptions = {}
    image_descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            id_descriptions[data['question_id']] = data['text']
    with open(r'tw_dataset/twitter_dataset/devset/CVLM_questions.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            id = data['question_id']
            image = os.path.basename(data['image'])
            image_descriptions[image] = id_descriptions[id]

    return image_descriptions

def Load_data(csv_file_path, desc_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    extractor = Text_features(list(df['post_text']), 5, if_visualize=False)
    
    # 加载图片描述
    image_descriptions = load_image_descriptions(desc_file_path)

    data_list = []
    
    for index, row in df.iterrows():
        post_text = preprocess_texts([row['post_text']])
        if len(post_text) == 0:
            continue
        post_text = " ".join(post_text[0])
        topic = extractor.LDA_topic_analysis(post_text)
        sentiment = extractor.vader_sentiment_analysis(post_text)
        image_id = row['image_id']
        label = int(row['label'])

        # 处理图片描述
        image_id = str(image_id)  # 确保是字符串类型
        if "," in image_id:
            image_ids = image_id.split(",")
        else:
            image_ids = [image_id]
        
        # 收集对应的图片描述
        image_descriptions_text = []
        for img_id in image_ids:
            img_id = img_id + '.jpg'
            if img_id in image_descriptions:
                image_descriptions_text.append(image_descriptions[img_id])
                # image_descriptions_text.append("No description available")
            else:
                image_descriptions_text.append("No description available")
        
        # 将多个图片描述合并
        combined_image_description = " ".join(image_descriptions_text)

        data_list.append(
            {
                'text': post_text,
                'topic': int(topic) if isinstance(topic, (int, float)) else int(topic[0]) if hasattr(topic, '__len__') else 0,
                'sentiment': float(sentiment) if not isinstance(sentiment, float) else sentiment,
                'image_description': combined_image_description,  # 新增图片描述
                'label': int(label)
            }
        )

    return data_list

class MultimodalClassifier(nn.Module):
    def __init__(self, num_topics=10, text_feature_dim=256, image_desc_feature_dim=256, fusion_output=256, num_classes=2):
        super(MultimodalClassifier, self).__init__()
        # 文本分支用于处理原始文本（包含主题和情感特征）
        self.text_branch = TextBranch(num_topics=num_topics)
        # 图片描述分支，只使用BERT + BiLSTM
        self.image_desc_branch = ImageDescriptionBranch(output_size=image_desc_feature_dim)
        
        # 合并文本特征和图片描述特征
        combined_dim = text_feature_dim + image_desc_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fusion_output),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_output, num_classes)
        )

    def forward(self, input_ids, attention_mask, lengths, topics, sentiments, 
                image_desc_input_ids, image_desc_attention_mask, image_desc_lengths):
        # 处理原始文本（包含主题和情感特征）
        text_feat = self.text_branch(input_ids, attention_mask, lengths, topics, sentiments)
        
        # 处理图片描述文本（只使用BERT + BiLSTM）
        image_desc_feat = self.image_desc_branch(image_desc_input_ids, image_desc_attention_mask, image_desc_lengths)
        
        # 拼接特征
        combined = torch.cat([text_feat, image_desc_feat], dim=1)
        logits = self.classifier(combined)
        return logits

class MultimodalDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 处理原始文本
        encoding = self.tokenizer(
            item['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理图片描述文本
        image_desc_encoding = self.tokenizer(
            item['image_description'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'lengths': torch.sum(encoding['attention_mask']).item(),
            'topic': item['topic'],
            'sentiment': item['sentiment'],
            'image_desc_input_ids': image_desc_encoding['input_ids'].flatten(),
            'image_desc_attention_mask': image_desc_encoding['attention_mask'].flatten(),
            'image_desc_token_type_ids': image_desc_encoding.get('token_type_ids', torch.zeros_like(image_desc_encoding['input_ids'])).flatten(),
            'image_desc_lengths': torch.sum(image_desc_encoding['attention_mask']).item(),
            'label': item['label']
        }

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    token_type_ids = torch.stack([b['token_type_ids'] for b in batch])
    lengths = torch.tensor([b['lengths'] for b in batch], dtype=torch.long)
    topics = torch.tensor([b['topic'] for b in batch], dtype=torch.long)
    sentiments = torch.tensor([b['sentiment'] for b in batch], dtype=torch.float32)

    # 图片描述相关
    image_desc_input_ids = torch.stack([b['image_desc_input_ids'] for b in batch])
    image_desc_attention_mask = torch.stack([b['image_desc_attention_mask'] for b in batch])
    image_desc_token_type_ids = torch.stack([b['image_desc_token_type_ids'] for b in batch])
    image_desc_lengths = torch.tensor([b['image_desc_lengths'] for b in batch], dtype=torch.long)

    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'lengths': lengths,
        'topic': topics,
        'sentiment': sentiments,
        'image_desc_input_ids': image_desc_input_ids,
        'image_desc_attention_mask': image_desc_attention_mask,
        'image_desc_token_type_ids': image_desc_token_type_ids,
        'image_desc_lengths': image_desc_lengths,
        'label': labels
    }

def train_epoch(epoch, model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    batch_count = 0
    
    # 用于记录每个batch的loss和acc
    batch_losses = []
    batch_accs = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lengths = batch['lengths'].to(device)
        topics = batch['topic'].to(device)
        sentiments = batch['sentiment'].to(device)
        
        image_desc_input_ids = batch['image_desc_input_ids'].to(device)
        image_desc_attention_mask = batch['image_desc_attention_mask'].to(device)
        image_desc_lengths = batch['image_desc_lengths'].to(device)
        
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, lengths, topics, sentiments,
                       image_desc_input_ids, image_desc_attention_mask, image_desc_lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
        
        # 记录当前batch的loss和acc
        current_acc = accuracy_score(labels.cpu().numpy(), preds)
        batch_losses.append(loss.item())
        batch_accs.append(current_acc)
        
        batch_count += 1
        
        if batch_count % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}, Acc: {current_acc:.4f}")

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, batch_losses, batch_accs

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    # 用于记录每个batch的loss和acc
    batch_losses = []
    batch_accs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lengths = batch['lengths'].to(device)
            topics = batch['topic'].to(device)
            sentiments = batch['sentiment'].to(device)
            
            image_desc_input_ids = batch['image_desc_input_ids'].to(device)
            image_desc_attention_mask = batch['image_desc_attention_mask'].to(device)
            image_desc_lengths = batch['image_desc_lengths'].to(device)
            
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, lengths, topics, sentiments,
                           image_desc_input_ids, image_desc_attention_mask, image_desc_lengths)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            
            # 记录当前batch的loss和acc
            current_acc = accuracy_score(labels.cpu().numpy(), preds)
            batch_losses.append(loss.item())
            batch_accs.append(current_acc)

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, batch_losses, batch_accs

if __name__ == "__main__":
    csv_file_path = 'tw_dataset/twitter_dataset/devset/posts.csv'
    desc_file_path = 'tw_dataset/twitter_dataset/devset/CVLM_answers.jsonl'
    data_list = Load_data(csv_file_path, desc_file_path)
    print("Finish loading data")

    model_path = r"checkpoint/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print(f'Tokenizer loaded: {model_path}')

    dataset = MultimodalDataset(data_list, tokenizer)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    print(f"train size: {len(train_data)}, val_size: {len(val_data)}, test_size: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 用于记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    all_batch_losses = []
    all_batch_accs = []
    all_val_batch_losses = []
    all_val_batch_accs = []

    for epoch in range(2):
        train_loss, train_acc, batch_losses, batch_accs = train_epoch(epoch, model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, val_batch_losses, val_batch_accs = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
        
        # 记录训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 记录训练batch级别的历史（添加epoch偏移）
        epoch_offset = epoch * len(batch_losses)
        for i, (loss, acc) in enumerate(zip(batch_losses, batch_accs)):
            all_batch_losses.append((epoch_offset + i + 1, loss))
            all_batch_accs.append((epoch_offset + i + 1, acc))
        
        # 记录验证batch级别的历史（添加epoch偏移）
        val_epoch_offset = epoch * len(val_batch_losses)
        for i, (loss, acc) in enumerate(zip(val_batch_losses, val_batch_accs)):
            all_val_batch_losses.append((val_epoch_offset + i + 1, loss))
            all_val_batch_accs.append((val_epoch_offset + i + 1, acc))

    test_loss, test_acc, batch_losses, batch_accs = evaluate(model, test_loader, criterion, device)
    print(f"\ntest Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # 提取batch级别的数据
    train_batch_steps = [step for step, _ in all_batch_losses]
    train_batch_loss_values = [loss for _, loss in all_batch_losses]
    train_batch_acc_values = [acc for _, acc in all_batch_accs]
    
    val_batch_steps = [step for step, _ in all_val_batch_losses]
    val_batch_loss_values = [loss for _, loss in all_val_batch_losses]
    val_batch_acc_values = [acc for _, acc in all_val_batch_accs]
    
    # 绘制训练batch loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_batch_steps, train_batch_loss_values, 'b-', linewidth=2)
    plt.title('Train Batch Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制训练batch acc曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_batch_steps, train_batch_acc_values, 'g-', linewidth=2)
    plt.title('Train Batch Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制验证batch loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(val_batch_steps, val_batch_loss_values, 'r-', linewidth=2)
    plt.title('Valid Batch Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 绘制验证batch acc曲线
    plt.figure(figsize=(10, 6))
    plt.plot(val_batch_steps, val_batch_acc_values, 'orange', linewidth=2)
    plt.title('Valid Batch Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("四张训练曲线已显示完成！")

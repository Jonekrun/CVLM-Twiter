# 认知视觉语言映射器：通过增强视觉知识对齐推进多模态理解

<font size=2><div align='center' >  [[📖 论文链接](https://arxiv.org/abs/2402.13561)] [[📊 数据集](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M)] </div></font>

## 数据集

技术论文"GPT-4V在知识密集型视觉问答上的综合评估"的评估数据集可在Huggingface上查看 [知识密集型数据集](https://huggingface.co/datasets/YunxinLi/Knowledge_QA)

在 [Wikipedia-Knowledge-2M](https://huggingface.co/datasets/Ghaser/Wikipedia-Knowledge-2M) 发布了两百万条维基百科知识数据集。该数据集包含一个JSON文件和一个包含所有图像文件的压缩档案。JSON文件中的图像属性对应压缩档案中的图像文件。

在 [LLaVA-KnowledgeQA-504K](https://huggingface.co/datasets/Ghaser/LLaVA-KnowledgeQA-504K) 提供了504K KnowldgeQA数据集的JSON文件。该数据集主要由OK-VQA、A-OKVQA和TextVQA的训练集组成。此数据集中的图像来自 [COCO Caption](https://cocodataset.org/#home) 和 [TextVQA](https://textvqa.org/)，您需要自行下载。

## 环境
- Pytorch `2.0.1`
```shell
conda env create -n CVLM python=3.8
conda activate CVLM
pip install -r requirement.txt
```

## 评估

GitHub上的sam_images不完整；需要从 [Hugging Face](https://huggingface.co/datasets/Ghaser/CVLM-SAM-Images) 重新下载。

在 [CVLM-LLaVA](https://huggingface.co/Ghaser/CVLM-LLaVA) 发布了基于LLaVA的最佳模型，并在 [CVLM-Opt](https://huggingface.co/Ghaser/CVLM-Opt-pretrain) 上发布了预训练的OPT。

下载检查点后，按如下方式组织权重：

```
└── LLaVA
    ├──checkpoints
        ├──CVLM-LLaVA
```

### LLaVA

LLaVA的评估脚本位于 `scripts/knowledge_qa/eval`，

我们主要评估了六个基准数据集：OK-VQA、VQAv2、A-OKVQA、TextVQA、InfoSeek和SEED-Bench。

#### OK-VQA
请注意，保存的结果文件将位于相应目录的answers_upload文件夹中。
```shell
bash scripts/knowledge_qa/eval/okvqa.sh
cd playground/knowledge_qa/eval/okvqa
python okvqa_eval.py --pred_file answers_upload/llava_okvqa_mscoco_val/CVLM-LLaVA-1epoch.json
```

#### VQAv2

```shell
bash scripts/knowledge_qa/eval/vqav2.sh
cd playground/knowledge_qa/eval/vqav2
python vqa_eval.py
```

#### A-OKVQA

对开放式A-OKVQA进行评估。以下脚本也将执行评估。

```shell
bash scripts/knowledge_qa/eval/aokvqa_oe.sh
```

对多选择A-OKVQA进行评估。

```shell
bash scripts/knowledge_qa/eval/aokvqa.sh
```

#### TextVQA

对TextVQA进行评估。
```shell
bash scripts/knowledge_qa/eval/textvqa.sh
```

#### InfoSeek

对InfoSeek进行评估。
```shell
bash scripts/knowledge_qa/eval/infoseek.sh
```

#### SEED-Bench

对SEED-Bench进行评估。
```shell
bash scripts/knowledge_qa/eval/seedbench.sh
```

<br>
<br>


# 认知视觉语言映射器：通过增强视觉知识对齐推进多模态理解
## 模型架构与实现

本项目实现了一个多模态分类器（MultimodalClassifier），主要包含以下组件：

### 1. 图像描述分支（ImageDescriptionBranch）
- 使用BERT和双向LSTM处理图像描述文本
- BERT用于提取文本特征
- 双向LSTM进行序列建模
- 输出维度为256的特征向量

### 2. 文本分支（TextBranch）
- 支持文本预处理，清洗文本、分词、去停用词等
- 能够提取文本的LDA主题
- 能够分析文本的情感偏向

### 3. 多模态分类器（MultimodalClassifier）
- 文本分支：处理原始文本，包含主题和情感特征
- 图像描述分支：处理图像相关的文本描述
- 特征融合：将文本特征和图像描述特征拼接
- 分类层：使用多层感知机进行最终分类

### 4. 训练与评估
- 实现了训练和评估循环
- 支持批次级别的损失和准确率监控
- 提供详细的训练过程可视化
- 包含验证集评估和测试集性能报告

## 结果文件夹
 - ./result


<br>

<br>
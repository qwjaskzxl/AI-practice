## 解题思路
1. 分析数据，统计词表等等
2. 使用效果较好的baseline如CNN、RNN先试探task难度
3. 加模型，如bert，传统ML的tfidf+svm等等
## 框架
- 运行run.py即可跑单模型，train_eval.py和utils.py为训练预测代码和工具代码
- models内为不同的模型
- fusion.py用于融合多模型（voting）

## 结果
TextCNN双向+TextRNN=0.9567
TextCNN_0.9412+TextRNN_0.9446=0.9540（+FastText0.9343一样）
bert_0.9580
TFIDF+SGDC_0.9447
TextCNN双向+TextRNN+BERT+SGDC_result=0.9615，换了BiLSTM后0.9635，不过是test_b了
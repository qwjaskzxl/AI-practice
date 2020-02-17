## 内容说明
- __CCF_sentiment_analysis__----竞赛代码
  - env：python3.6，pycharm
  - motivation：第一次使用OOP，在实现功能的基础上，力求高内聚低耦合，欲提高使用便利性与可维护性；
  - future work：后期发现，进行pipeline时，如此分离不便于feature engineering+ML model的联合调参，还是得将二者合并，这样与初心背道而驰，将此部分须思考如何改进
- __PyTorch_forNLP__----pytorch学习代码
  - env：python3.6+pytorch1.1，jupyter notebook
  - motivation：按官方tutorial学习pytorch，力求实现CS224n中所有任务模型
    - word representation：word2vec/cbow(done)、glove、EMLo、tf、BERT
    - classification(NER、POS etc.)：traditional model(done)、CNN、RNN
    - LM：have tried LSTM for MLG(done)
    - MIT/QA：RNN、tf->test on BLEU/SQuAD
  - future work：
    - CLF by pure WE、CNN、RNN(+attn)、fastText、tf
    - MIT/QA by RNN(LSTM、GRU)、RNN+attn、tf


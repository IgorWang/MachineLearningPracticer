# MachineLearningPracticer
机器学习实践

### basic
基本算法的调用：notebook详细说明

## 深度学习

### Tensorflow

####test1:minist 
- 构建Graph 相当于构建模型 包括模型的训练;预测;评价
- 单独构建训练的步骤
- 需要为特殊的训练集构造数据读取的方法

#### LSTM语言模型

训练
- config中的Opthion类包含所有的模型参数,可以继承调节,文件中包含训练的FLAGS
- 
```Python
from TensorFlow.word_rnn.configure import *
from TensorFlow.word_rnn.model import *

FLAGS.data_path = os.path.join(os.path.dirname(__file__), 'data')
config = Options()

# reader是数据的读取,包含方法read_data和iterator
# 从类方法训练生成和储存模型
model = WordRNN.train(config, reader, verbose=True)
```
预测
```Python
# 单个句子的预测
config.batch_size = 1
model = WordRNN.load(config, 'your_model_data/')
data, word2id = reader.read_data(FLAGS.data_path)
id2word = {v: k for k, v in word2id.items()}
epoch = 0

for i, (x, y) in enumerate(reader.iterator(data, config.batch_size, config.num_steps)):
    #测试30个句子
    print("x", list(map(lambda x: id2word[x], x[0])))
    pred = model.predict(x)
    print("predict", list(map(lambda x: id2word[x], pred)))
    if i > 30:
        break
```


# CNN for Text Classification(cnnc)

1 预处理后的数据格式-CSV
```
id,id,id,...,id,label
15,32,158,...,156,0,1
45,99,36,...,369,1,0
```
id 表示词的id
label 表示分类的结果 用[0,1] 和 [1,0]分别表示

2 数据读取-CSV reader
```Python
filenames = ['data/train/train.txt']
    input_data = distorted_inputs(filenames, 2)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            features, lables = sess.run([input_data[0],
                                         input_data[1]])
            print(features)
            print(lables)
        print(features.shape)
        print(lables.shape)

        coord.request_stop()
        coord.join(threads)
```

3 模型构建
运行cncc_train.python脚本

4 模型预测
import pandas as pd

filenames = os.path.join('data/train/train.txt')

data = pd.read_csv(filenames, sep=',', header=None)
x = data.iloc[0:, 0:cnnc.SEQUENCE_LENGTH].values
y = data.iloc[0:, cnnc.SEQUENCE_LENGTH:].values

result = evaluate(x, y)
print(result[1])

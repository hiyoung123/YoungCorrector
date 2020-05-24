# YoungCorrector
本项目是参考开源框架 [Pycorrector](https://github.com/shibing624/pycorrector)，自己实现了一套基于规则的纠错系统。
总体来说，基于规则的文本纠错，性能取决于纠错词典和分词质量。目前与 [Pycorrector](https://github.com/shibing624/pycorrector) 相比，在准确率差不多的情况下，本模型所用的时间会少很多（归功于前向最大匹配替代了直接索引混淆词典）。
代码还没有完善，还有很多优化的空间，后续会持续更新。。。

## 中文文本纠错
### 介绍

1. 文本纠错的核心步骤：错误检测，候选召回，纠错排序。
   * 错误检测：找到哪些词是错误的。
   * 候选召回：选出纠错候选词。
   * 纠错排序：对候选词进行排序。
2. 主流的三种方法：
   * 基于规则：pycorrector
   * 基于深度模型：百度纠错系统
   * 基于垂直领域：腾讯DCQC纠错框架
3. 中文纠错需要解决的问题：
   * 谐音字词，如 配副眼睛-配副眼镜
   * 混淆音字词，如 流浪织女-牛郎织女
   * 字词顺序颠倒，如 伍迪艾伦-艾伦伍迪
   * 字词补全，如 爱有天意-假如爱有天意
   * 形似字错误，如 高梁-高粱
   * 中文拼音全拼，如 xingfu-幸福
   * 中文拼音缩写，如 sz-深圳
   * 语法错误，如 想象难以-难以想象


### 资源

1. 语言模型：LSTM，N-gram（kenlm 库）
2. 混淆词典（字典）
3. 通用词典（词频）
4. 音近词典（可使用Pypinyin代替或结合）
5. 形近词典


### 基于规则的中文纠错流程

1. 文本处理

   * 是否为空
   * 是否全是英文
   * 统一编码
   * 统一文本格式

2. 错误检索

   * 从混淆词典/字典中检索

     直接检索

     最大匹配算法

   * 分词后，查看是否存在于通用词典

     分词质量

     分词后产生的单字（字本身是ok的，但是在整个句子中是错误的）

     分词后产生的错词（词本身是ok的，但是在整个句子中是错误的）

     分词后的词是否包含其他字母及符号

   * 词粒度的 N-gram

     计算局部得分

     MAD(Median Absolute Deviation)

   * 字粒度的 N-gram

     计算局部得分

     MAD(Median Absolute Deviation)

   * 将候选错误词中的连续单字合并

3. 候选召回

   * 编辑距离

     错误词的长度大于 1

     只使用编辑距离为 1

     只使用变换和修改，无添加和删除

     对编辑距离得到的词使用拼音加以限制

   * 音近词

   * 形近词

4. 纠错排序

   * 语言模型计算得分（困惑度）


### 功能模块

1. 语言模型（kenlm，自定义ngram，DNNLM）
   * score(sentence) ：计算文本得分
   * ppl(sentence)：困惑度
   * Model(path)：加载模型
2. 分词器（jieba，自定义分词器）
   * tokenize(sentence)
3. 资源加载
   * 加载混淆词典
   * 加载通用词典
   * 加载音近词典
   * 加载形近词典
   * 加载分词词典
4.

### 思考及问题点

   * 存在异常字母标点符号等内容
   * 英文纠错处理
   * 错误词影响分词质量
   * 语言模型计算得分的准确性
   * 执行两遍纠错（效果目前来看不行）
   * 加入依存法分析
   * 从检索，召回和排序都有优化的空间
   * 多线程检索
   * 添加 1-gram（貌似没什么效果）
   * 混淆词典检索优化（使用前向最大匹配，可以节约时间，随着混淆词典越大，节约时间越多）
   * pycorrector 无法处理一句话中有两个混淆词典中的词(使用前向最大匹配替代直接索引，可以解决该问题)


### 更新日程

* 2020/05/24 添加 1-gram
* 2020/05/24 使用前向最大匹配替代直接检索
* 2020/05/22 初版完成


### 参考

* [中文文本纠错算法--错别字纠正的二三事](https://zhuanlan.zhihu.com/p/40806718)

* [NLP上层应用的关键一环——中文纠错技术简述](https://zhuanlan.zhihu.com/p/82807092)

* [PyCorrector文本纠错工具实践和代码详解](https://zhuanlan.zhihu.com/p/138981644)

* [pycorrector](https://github.com/shibing624/pycorrector)

* [correction](https://github.com/ccheng16/correction)

* [kenlm](https://github.com/kpu/kenlm)

* [语言模型kenlm的训练及使用](https://www.cnblogs.com/zidiancao/p/6067147.html)

* [Spelling Correction and the Noisy Channel](https://web.stanford.edu/class/cs124/lec/spelling.pdf)

* [How to Write a Spelling Corrector](https://norvig.com/spell-correct.html)

* [查询纠错](https://github.com/liuhuanyong/QueryCorrection)

* [中文文本纠错综述](https://blog.csdn.net/sinat_26917383/article/details/86737361)

* [百度纠错系统api](https://ai.baidu.com/tech/nlp/text_corrector)





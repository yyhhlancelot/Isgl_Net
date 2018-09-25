import re # 正则表达式
from bs4 import BeautifulSoup # html标签处理
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

def review_to_wordlist(review):
    '''comment to word vec'''
    
    # 去掉HTML标签，拿到内容 去掉<br />
    review_text = BeautifulSoup(review, features = "lxml").get_text()
    
    # print(review_text)
    
    # 用正则表达式取出符合规范的部分 # 去掉标点符号和数字
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    
    # 返回words
    return words
# inputStr = "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
# inputStr = review_to_wordlist(inputStr)

# 载入数据

train = pd.read_csv('J:/Code/kaggle/Bags_of_Words_Meets_Bags_of_Popcorn/labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)

test = pd.read_csv('J:/Code/kaggle/Bags_of_Words_Meets_Bags_of_Popcorn/testData.tsv', header = 0, delimiter = '\t', quoting = 3)

# 取出情感标签

y_train = train['sentiment']

train_data = []

for i in range(0, len(train['review'])):
    train_data.append(" ".join(review_to_wordlist(train['review'][i])))

test_data = []

for i in range(0, len(test['review'])):
    test_data.append(" ".join(review_to_wordlist(test['review'][i])))

# 初始化TFIV对象，去停用词，加2元语言模型

tfv = TFIV(min_df = 3, max_features = None, strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}', ngram_range = (1, 2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1, stop_words = 'english')

# 合并训练和测试集以便进行TFIDF向量化操作,获取特征

X_all = train_data + test_data
len_train = len(train_data)

# print(X_all)

tfv.fit(X_all)
print(tfv)
X_all = tfv.transform(X_all)

# 恢复成训练集和测试集部分
X_train = X_all[:len_train]
X_test = X_all[len_train:]


from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
import json

from sqlalchemy import update

from module1.models import CoronaNet

import requests,urllib
from bs4 import BeautifulSoup

# nltk.download('stopwords')


def testA():
    train_set = ["The sky is blue.", "The sun is bright."]  # Documents
    test_set = ["The sun in the sky is bright."]  # Query
    stopWords = stopwords.words('english')

    vectorizer = CountVectorizer(stop_words=stopWords)
    # print vectorizer
    transformer = TfidfTransformer()
    # print transformer

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(test_set).toarray()
    print('Fit Vectorizer to train set', trainVectorizerArray)
    print('Transform Vectorizer to test set', testVectorizerArray)
    cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)

    for vector in trainVectorizerArray:
        print(vector)
        for testV in testVectorizerArray:
            print(testV)
            cosine = cx(vector, testV)
            print(cosine)

    transformer.fit(trainVectorizerArray)

    print(transformer.transform(trainVectorizerArray).toarray())

    transformer.fit(testVectorizerArray)

    tfidf = transformer.transform(testVectorizerArray)
    print(tfidf.todense())


def testB(option, answer):
    options = []
    answers = []

    options.append(option)
    answers.append(answer)

    if len(options) > 0 and len(answers) > 0:
        stopWords = stopwords.words('english')
        vectorizer = CountVectorizer(stop_words=stopWords)

        optionsVectorizerArray = vectorizer.fit_transform(options).toarray()
        answersVectorizerArray = vectorizer.transform(answers).toarray()

        cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)

        for option in optionsVectorizerArray:
            for answer in answersVectorizerArray:
                cosine = cx(option, answer)
                print(cosine)


def testC():
    with open('./static/questions.json', encoding="utf8") as f:
        data = json.load(f)

    # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
    print(data)


def testD():
    db = SQLAlchemy()



    # data = {'pid': '2975738', 'qid': 'q2', 'answer': 'op1'}
    # corona_net = CoronaNet()
    # stmt = (
    #     update(corona_net).\
    #         where(corona_net.policy_id==data['pid']).\
    #         values(update_type=data['answer'])
    # )

    policy = CoronaNet.query.first()
    print(policy)


def testE():
    url = 'https://hr.usembassy.gov/health-alert-u-s-embassy-zagreb-croatia-march-19-2020/'
    r = requests.get(url, timeout=30)

    r.raise_for_status()  # HTTP请求返回状态码，200表示成功

    r.encoding = r.apparent_encoding
    # r.encoding从HTTP header中猜测的响应内容的编码方式
    # r.apparent_encoding从内容中分析响应内容的编码方式(备选编码方式)

    r.text  # HTTP响应的字符串形式，即，url对应的页面内容


    # url = 'https://blog.csdn.net/github_38885296/article/details/90544542'
    #
    # r = requests.get(url, timeout=30)
    # print(r.apparent_encoding)  # 在爬取过程中发现中文乱码，所以要看看网页编码是不是utf-8，如果不是就得改
    # r.encoding = 'utf-8'

    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.find('body')
    # title = soup.find_all('h1', {'class': 'title-article'})[0].get_text()  # 标题
    # content = soup.find_all('div', {'class': 'htmledit_views'})  # 文字内容
    text = ''
    for c in body:
        if c.name == 'span' or c.name == 'p' or c.name == 'dive':
            print(c)


if __name__ == '__main__':
    testA()
    # testB()
    # testC()
    # testD()
    # testE()

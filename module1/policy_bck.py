from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import CountVectorizer
from flask import Blueprint, render_template
from flask_login import login_required
from module1.models import CoronaNet
from flask import request
from module1 import db

import numpy.linalg as LA
import numpy as np
import torch
import json

bp = Blueprint('policy', __name__)

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")


def setValue(policy, columnName, answer):
    if columnName == 'UpdateType':
        policy.update_type = answer

    if columnName == 'Country':
        policy.country = answer

    return policy


@bp.route("/policies/save_summary", methods=['POST'])
@login_required
def save_summary():
    data = request.data.decode("utf-8").split("------")
    policy_id = data[0]
    summary = data[1]

    policy = db.session.query(CoronaNet).filter_by(policy_id=policy_id).first()
    policy.description = summary

    db.session.commit()

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@bp.route("/policies/save", methods=['POST'])
@login_required
def save():
    dataJson = request.data.decode("utf-8")
    data = json.loads(dataJson)

    policy = db.session.query(CoronaNet).filter_by(policy_id=data["pid"]).first()

    policy = setValue(policy, data['column'], data['answer'])
    # policy.update_type = data['answer']
    db.session.commit()

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@bp.route("/policies/get_highlighting_text", methods=['GET', 'POST'])
@login_required
def get_highlighting_text():
    data = request.data.decode("utf-8").split("------")
    policy_id = data[0]
    summary = data[1]

    return summary

def signle_QA(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])).replace('[CLS]', '')

    if '[SEP]' in answer:
        answer = (answer.split('[SEP]')[1]).strip()

    return answer


def multi_QA(question, contexts):
    answers = ''
    for context in contexts:
        answer = signle_QA(question, context)
        if answer != None and answer != "":
            if answers == '':
                answers = answer
            else:
                answers = answers + "|" + answer
    return answers


@bp.route("/policies/<int:policy_id>/annotation", methods=['GET', 'POST'])
@login_required
def annotation(policy_id):
    with open('./module1/static/questions.json', encoding="utf8") as f:
        q_objs = json.load(f)

        policy = CoronaNet.query.filter_by(policy_id=policy_id).first()
        context = policy.description.split('.')

        for q in q_objs:
            answers = multi_QA(q["question"], context)

            # test
            # if answer == '':
            #     answer = 'A dimension other than the policy initiator'

            q["answers"] = answers

            if q["taskType"] == 1:
                m_cos = 0
                for option in q["options"]:
                    option_text = option["text"]
                    cosine_similarity = max_cos(option_text, answers)

                    # cosine_similarity = 0.8

                    option["cosine_similarity"] = cosine_similarity

                    if m_cos < cosine_similarity:
                        m_cos = cosine_similarity

                    # tmp = option["cosine_similarity"]
                    # print(cosine_similarity)
                    # print(tmp)

                for option in q["options"]:
                    if option["cosine_similarity"] == max_cos:
                        option["checked"] = "True"
                        break

    summary_list = policy.description.split('\n')
    return render_template('annotation.html', policy=policy, questions=q_objs, summary_list=summary_list)


def max_cos(option, answers):
    max=0
    for answer in answers:
        c = cos(option, answer)
        if c > max:
            max = c

    return max


def cos(option, answer):
    cosine = 0

    options = []
    answers = []

    options.append(option)
    answers.append(answer)

    if len(options) > 0 and len(answers) > 0:
        stopWords = stopwords.words('english')
        vectorizer = CountVectorizer(stop_words=stopWords)

        option = vectorizer.fit_transform(options).toarray()[0]
        answer = vectorizer.transform(answers).toarray()[0]

        cosine = consine_cal(option, answer)

    return cosine


def consine_cal(v1, v2):
    a = np.inner(v1, v2)
    b = LA.norm(v1) * LA.norm(v2)
    if a == 0 or b == 0:
        return 0

    return round(a / b, 3)


@bp.route("/policies/<int:policy_id>/view", methods=['GET', 'POST'])
@login_required
def view(policy_id):
    policy = CoronaNet.query.filter_by(policy_id=policy_id).first().__dict__
    return render_template('view.html', policy=policy)


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re


@bp.route("/policies/reload_summary", methods=['GET', 'POST'])
@login_required
def reload_summary():
    policy_id = request.data.decode("utf-8")
    summary = ''

    policy = CoronaNet.query.filter_by(policy_id=policy_id).first()

    nltk.download('punkt')
    text = policy.original_text
    stopWords = stopwords.words('english')
    stopWords = set(stopWords)
    words = word_tokenize(text)
    freqTable = dict()

    for word in words:
        word = word.lower()

        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    # sentences = re.split('; |. |\n', text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    average = int(sumValues / len(sentenceValue))

    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary

@bp.route("/policies/<int:policy_id>/summary", methods=['GET', 'POST'])
@login_required
def summary(policy_id):
    policy = CoronaNet.query.filter_by(policy_id=policy_id).first()

    if policy.description == '' or policy.description == None:
        nltk.download('punkt')
        text = policy.original_text
        stopWords = stopwords.words('english')
        stopWords = set(stopWords)
        words = word_tokenize(text)
        freqTable = dict()

        for word in words:
            word = word.lower()

            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

        sentences = sent_tokenize(text)
        # sentences = re.split('; |. |\n', text)
        sentenceValue = dict()

        for sentence in sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

        sumValues = 0
        for sentence in sentenceValue:
            sumValues += sentenceValue[sentence]
        average = int(sumValues / len(sentenceValue))

        summary = ''
        highlighted = []
        for sentence in sentences:
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                summary += " " + sentence
                highlighted.append([sentence, True])
            else:
                highlighted.append([sentence, False])
        policy.description = summary
        policy.highlighted_text = highlighted
        return render_template('summary.html', policy=policy, has_summary=False)
    else:
        policy.highlighted_text = [[policy.original_text, False]]
        return render_template('summary.html', policy=policy, has_summary=True)


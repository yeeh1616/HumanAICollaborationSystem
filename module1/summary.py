from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from nltk.tokenize import word_tokenize, sent_tokenize
from flask import Blueprint, render_template
from flask_login import login_required
from module1.models import CoronaNet
from nltk.corpus import stopwords
from flask import request
from module1 import db

import json
import nltk

bp_summary = Blueprint('summary', __name__)

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

nltk.download('stopwords')

@bp_summary.route("/policies/save_summary", methods=['POST'])
@login_required
def save_summary():
    data = request.data.decode("utf-8").split("------")
    policy_id = data[0]
    summary = data[1]

    policy = db.session.query(CoronaNet).filter_by(policy_id=policy_id).first()
    policy.description = summary

    db.session.commit()

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@bp_summary.route("/policies/get_highlighting_text", methods=['GET', 'POST'])
@login_required
def get_highlighting_text():
    data = request.data.decode("utf-8").split("------")
    policy_id = data[0]
    summary = data[1]

    return summary


@bp_summary.route("/policies/reload_summary", methods=['GET', 'POST'])
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


@bp_summary.route("/policies/<int:policy_id>/get_summary", methods=['GET', 'POST'])
@login_required
def get_summary(policy_id):
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

    summary = ''
    highlighted = []
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
            highlighted.append([sentence, True])
        else:
            highlighted.append([sentence, False])

    summary = summary.replace('\r','').replace('\n', ' ')

    has_summary = False
    if policy.description == '' or policy.description == None:
        policy.description = summary
    else:
        has_summary = True
        policy.description
    policy.highlighted_text = highlighted
    return render_template('summary.html', policy=policy, has_summary=has_summary)


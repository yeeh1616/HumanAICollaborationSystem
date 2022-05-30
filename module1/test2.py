import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from module1.models import CoronaNet
from flask_sqlalchemy import SQLAlchemy
import json


'''
1. 读数据库的一条policy， 取出 original text
2. 读json文件， 取出 question list
3. 把t5模型名字里一个列表
4. 循环里表里面的model 把当前model名和所有问题以及相应的答案写入到txt文件中（每个model一个txt文件）
5. 挑出最好的model

6. 建KG
7. 找论文80多页的那个模板
'''

# 1
def get_policy_text(policy_id):
    # db = SQLAlchemy()
    # res = db.session.query(CoronaNet).filter_by(policy_id=policy_id).first()
    res = '''Afghanistan Flash Update: Daily Brief: COVID-19, No. 15 (18 March 2020)
Key Messages: UPDATED
• People confirmed to have COVID-19: 22
• People tested for COVID-19: 305
• People confirmed negative for COVID-19: 273
• Pending results: 10
• Key concern: Border crossing areas in the country’s west
(Source: Ministry of Public Health of Afghanistan)
Situation Overview: UPDATED
Globally, 194,029 people have been confirmed to have contracted COVID-19 and 7,873 fatalities have been reported across 164 countries. The overall number of confirmed cases and fatalities outside China is now higher than in China. On 11 March, WHO declared the COVID-19 outbreak as a global pandemic. WHO reminds all countries and communities that the spread of this virus can be significantly slowed or even reversed through the implementation of robust containment and control activities. The increasing spread of the virus from and within Italy, Iran, Spain, France, Germany and South Korea remains a concern. Travel restrictions by countries are changing rapidly and should be monitored on daily basis.
The first person to test positive for COVID-19 in Afghanistan was confirmed on 24 February by the Ministry of Public Health (MoPH). A total of 22 people are now confirmed to have the virus in Hirat (13), Badghis (1), Balkh (1), Daykundi (1), Loghar (2), Kapisa (1) and Samangan (3) provinces. Contact tracing for the people confirmed with COVID-19 is ongoing. The clinical condition of the people both confirmed and presumptive for the virus is considered good. One patient in Hirat has reportedly recovered and been discharged from the treatment facility. On 14 March, the Government of Afghanistan announced that all schools would be closed for an initial period of 4 weeks – through to 18 April 2020. It is reported that all public gatherings in Hirat have been banned until further notice and further advice is being given against public celebration of the Nawruz holiday in Mazar-e-Sharif.
A number of people being held in isolation in hospital in Hirat left the facility on 16 March, although some have reportedly since returned to the hospital. A range of factors including hospital conditions, distrust of the authorities, loss of livelihoods issues, stigma and lack of understanding of risk and fear are likely to have contributed to this situation and warrant a scale-up of awareness raising among those being isolated in hospitals. The Protection Cluster will endeavour to negotiate access to those being held in medical isolation in order to ensure they understand what is happening to them and that their well-being is being protected and their specific needs addressed. Improved awareness raising at border crossings will also support this. Addressing rumours and community fears of seeking medical treatment through community engagement will be critical. The Government has also advised its provincial and district level counterparts to initiate awareness raising through community leaders and using mosques.'''
    return res

# 2.1
def get_json_obj(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)
    return None

# 2.2
def get_question_list(json_obj):
    questions = []

    for q in json_obj:
        questions.append(q['question'])
    return questions


def signle_QA(question, context, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # from transformers import DistilBertTokenizer, DistilBertModel
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])).replace('[CLS]', '')

    if '[SEP]' in answer:
        answer = (answer.split('[SEP]')[1]).strip()

    if answer not in context:
        return ""
    return answer


def signle_QA2(question, context, model_name):
    from transformers import pipeline

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)

    return res


def multiple_QA(question, context, model_name):
    answers = []
    context_list = context.replace('\r', '').split('\n')

    for c in context_list:
        # res = signle_QA(question, c, model_name)
        res = signle_QA2(question, c, model_name)
        answers.append(res['answer'])

    return answers

def test():
    policy_id = '2975738'
    json_path = './static/questions.json'

    # 1. read a policy
    policy_text = get_policy_text(policy_id)

    # 2. read json
    json_obj = get_json_obj(json_path)
    questions = get_question_list(json_obj)

    # 3. t5 model list
    t5 = []
    t5.append('bert-large-uncased-whole-word-masking-finetuned-squad')
    # t5.append('deepset/bert-base-cased-squad2')
    # t5.append('ozcangundes/T5-base-for-BioQA')
    # t5.append('deep-learning-analytics/triviaqa-t5-base')
    # t5.append('ozcangundes/mt5-multitask-qa-qg-turkish')
    # t5.append('Pollawat/mt5-small-thai-qa-qg')
    # t5.append('mrm8488/t5-base-finetuned-quartz')
    # t5.append('ozcangundes/mt5-small-turkish-squad')
    # t5.append('MariamD/my-t5-qa-legal')

    # 4.
    for model_name in t5:
        for question in questions:
            if question == '':
                continue

            answers = multiple_QA(question, policy_text, model_name)

            # delete the file content
            filename = './static/text' + model_name.replace('/', '_') + '.txt'
            with open(filename, 'a+', "utf-8") as f:
                f.write('')

            # 5. write model name, questions and answers to a text file
            with open(filename, 'a+', "utf-8") as f:
                f.write(question + '\n')

                if len(answers) == 0:
                    f.write("None.")

                for answer in answers:
                    line = answer + '\n'
                    f.write(line)
                f.write('\n\n\n')


def test2():
    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text = "What is a car?"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    pass


def test3():
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    # model_name = "deepset/roberta-base-squad2"
    model_name = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)

    pass


def test4():
    from transformers import pipeline
    model_name = 'deepset/bert-base-cased-squad2'
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Was the policy made from a National?',
        'context': 'Ministry of Public Health of Afghanistan.'
    }

    print(nlp(QA_input)['answer'])

    pass


def test5():
    # Write some lines to encode (sentences 0 and 2 are both ideltical):
    sen = [
        '''Afghanistan Flash Update: Daily Brief: COVID-19, No. 15 (18 March 2020)
Key Messages: UPDATED
• People confirmed to have COVID-19: 22
• People tested for COVID-19: 305
• People confirmed negative for COVID-19: 273
• Pending results: 10
• Key concern: Border crossing areas in the country’s west
(Source: Ministry of Public Health of Afghanistan)
Situation Overview: UPDATED
Globally, 194,029 people have been confirmed to have contracted COVID-19 and 7,873 fatalities have been reported across 164 countries. The overall number of confirmed cases and fatalities outside China is now higher than in China. On 11 March, WHO declared the COVID-19 outbreak as a global pandemic. WHO reminds all countries and communities that the spread of this virus can be significantly slowed or even reversed through the implementation of robust containment and control activities. The increasing spread of the virus from and within Italy, Iran, Spain, France, Germany and South Korea remains a concern. Travel restrictions by countries are changing rapidly and should be monitored on daily basis.
The first person to test positive for COVID-19 in Afghanistan was confirmed on 24 February by the Ministry of Public Health (MoPH). A total of 22 people are now confirmed to have the virus in Hirat (13), Badghis (1), Balkh (1), Daykundi (1), Loghar (2), Kapisa (1) and Samangan (3) provinces. Contact tracing for the people confirmed with COVID-19 is ongoing. The clinical condition of the people both confirmed and presumptive for the virus is considered good. One patient in Hirat has reportedly recovered and been discharged from the treatment facility. On 14 March, the Government of Afghanistan announced that all schools would be closed for an initial period of 4 weeks – through to 18 April 2020. It is reported that all public gatherings in Hirat have been banned until further notice and further advice is being given against public celebration of the Nawruz holiday in Mazar-e-Sharif.
A number of people being held in isolation in hospital in Hirat left the facility on 16 March, although some have reportedly since returned to the hospital. A range of factors including hospital conditions, distrust of the authorities, loss of livelihoods issues, stigma and lack of understanding of risk and fear are likely to have contributed to this situation and warrant a scale-up of awareness raising among those being isolated in hospitals. The Protection Cluster will endeavour to negotiate access to those being held in medical isolation in order to ensure they understand what is happening to them and that their well-being is being protected and their specific needs addressed. Improved awareness raising at border crossings will also support this. Addressing rumours and community fears of seeking medical treatment through community engagement will be critical. The Government has also advised its provincial and district level counterparts to initiate awareness raising through community leaders and using mosques.'''
,
        "It is at the national level.",
        "It is at the province or state level.",
        "It is a the city or municipal level.",
        "It is at another governmental level.",
    ]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # Encoding:
    sentence_embeddings = model.encode(sen)
    sentence_embeddings.shape

    from sklearn.metrics.pairwise import cosine_similarity
    # let's calculate cosine similarity for sentence 0:
    return cosine_similarity(
        [sentence_embeddings[0]],
        sentence_embeddings[1:]
    )

import json

def test6():
    x = [[("Test text.", False), ("Test text.", False)], [("Test text.", False), ("Test text.", False)]]
    y = json.dumps(x)

    x2 = json.loads(y)

    print(y)


if __name__ == '__main__':
    # test()
    # test2()
    # test3()
    # test4()
    # v = test5()
    # print(v)
    # test6()
    a = [{1:"asdf",2:[{1:"123"}]}]
    print(json.dumps(a))
    print('Done.')

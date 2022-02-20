import json

from flask import Blueprint, render_template, request
from flask_login import login_required

from module1.models import CoronaNet

bp_configuration = Blueprint('configuration', __name__)


@bp_configuration.route('/show', methods=['GET', 'POST'])
@login_required
def show():
    with open('./module1/static/questions.json', encoding="utf8") as f:
        q_objs = json.load(f)

    return render_template('configuration.html', questions=q_objs)


@bp_configuration.route('/configuration/save', methods=['GET', 'POST'])
@login_required
def save():
    data = request.data.decode("utf-8").split('|')

    with open('./module1/static/questions.json', 'r', encoding="utf8") as f:
        q_objs = json.load(f)

        for q in q_objs:
            if q["columnName"] == data[0]:
                if data[1] == 'True':
                    q["selected"] = True
                else:
                    q["selected"] = False
                break

    with open('./module1/static/questions.json', 'w', encoding="utf8") as f:
        json.dump(q_objs, f)

    return "OK"


@bp_configuration.route('/configuration/done', methods=['GET', 'POST'])
@login_required
def done():
    page = request.args.get('page', 1, type=int)
    policy_list = CoronaNet.query.paginate(page=page, per_page=10)

    return render_template('policy_list.html', policy_list=policy_list)
from flask import Blueprint, render_template, request, make_response
from flask_login import login_required
from module1.models import CoronaNet
from flask_paginate import Pagination, get_page_parameter

bp_policies = Blueprint('policies', __name__)
per_page = 10

@bp_policies.route("/policies/all", methods=['GET', 'POST'])
@login_required
def getAllPolicies():
    # search = bool(request.args.get('q'))
    # search = request.form.get('q')
    page = request.args.get(get_page_parameter(), type=int, default=1)
    policies_count = CoronaNet.query.count()

    policy_list = CoronaNet.query.paginate(page=page, per_page=per_page).items
    pagination = Pagination(page=page,
                            total=policies_count,
                            search=False,
                            record_name='policy_list',
                            css_framework='bootstrap3',
                            per_page=per_page,
                            show_single_page=True)

    return render_template('policy_list.html',
                    policy_list=policy_list,
                    pagination=pagination,
                    )


@bp_policies.route("/policies/<string:policy_id>/search", methods=['GET', 'POST'])
@login_required
def search(policy_id):
    policy_list = CoronaNet.query.filter_by(policy_id=policy_id)
    return seachBase(policy_list)


@bp_policies.route("/policies/searchAll", methods=['GET', 'POST'])
@login_required
def searchAll():
    policy_list = CoronaNet.query.paginate(page=1, per_page=per_page).items
    return seachBase(policy_list)

def seachBase(policy_list):
    policies_count = CoronaNet.query.count()
    pagination = Pagination(page=1,
                            total=policies_count,
                            search=search,
                            record_name='policy_list',
                            css_framework='bootstrap3',
                            per_page=per_page)
    return render_template('policy_list.html', policy_list=policy_list, pagination=pagination)
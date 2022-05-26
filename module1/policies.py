from flask import Blueprint, render_template
from flask_login import login_required
from module1.models import CoronaNet

bp_policies = Blueprint('policies', __name__)


@bp_policies.route("/policies/<int:policy_id>/search", methods=['GET', 'POST'])
@login_required
def search(policy_id):
    policy_list = CoronaNet.query.filter_by(policy_id=policy_id).paginate(page=1, per_page=10)
    return render_template('policy_list.html', policy_list=policy_list)


@bp_policies.route("/policies/searchAll", methods=['GET', 'POST'])
@login_required
def searchAll():
    policy_list = CoronaNet.query.paginate(page=1, per_page=10)
    return render_template('policy_list.html', policy_list=policy_list)
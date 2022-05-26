from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Blueprint, render_template
from flask_login import login_required
from module1.models import CoronaNet
from nltk.corpus import stopwords
from flask import request
from module1 import db

import numpy.linalg as LA
import numpy as np
import torch
import json
import re

bp_annotation = Blueprint('view', __name__)


@bp_annotation.route("/policies", methods=['GET', 'POST'])
@login_required
def view():
    policy_list = CoronaNet.query.paginate(page=1, per_page=10)

    return render_template('policy_list.html', policy_list=policy_list)

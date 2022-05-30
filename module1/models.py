from flask_login import UserMixin

from module1 import db


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class CoronaNet(db.Model):
    __tablename__ = 'corona_net'
    # record_id = db.Column(db.Text, primary_key=True)
    policy_id = db.Column(db.Integer, primary_key=True)
    entry_type = db.Column(db.Text, nullable=False)
    correct_type = db.Column(db.Text, nullable=False)
    update_type = db.Column(db.Text, nullable=False)
    update_level = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=False)
    date_announced = db.Column(db.Text, nullable=False)
    date_start = db.Column(db.Text, nullable=False)
    date_end = db.Column(db.Text, nullable=False)
    country = db.Column(db.Text, nullable=False)
    ISO_A3 = db.Column(db.Text, nullable=False)
    ISO_A2 = db.Column(db.Text, nullable=False)
    init_country_level = db.Column(db.Text, nullable=False)
    domestic_policy = db.Column(db.Text, nullable=False)
    province = db.Column(db.Text, nullable=False)
    ISO_L2 = db.Column(db.Text, nullable=False)
    city = db.Column(db.Text, nullable=False)
    type = db.Column(db.Text, nullable=False)
    type_sub_cat = db.Column(db.Text, nullable=False)
    type_text = db.Column(db.Text, nullable=False)
    institution_status = db.Column(db.Text, nullable=False)
    target_country = db.Column(db.Text, nullable=False)
    target_geog_level = db.Column(db.Text, nullable=False)
    target_region = db.Column(db.Text, nullable=False)
    target_province = db.Column(db.Text, nullable=False)
    target_city = db.Column(db.Text, nullable=False)
    target_other = db.Column(db.Text, nullable=False)
    target_who_what = db.Column(db.Text, nullable=False)
    target_direction = db.Column(db.Text, nullable=False)
    travel_mechanism = db.Column(db.Text, nullable=False)
    compliance = db.Column(db.Text, nullable=False)
    enforcer = db.Column(db.Text, nullable=False)
    dist_index_high_est = db.Column(db.Text, nullable=False)
    dist_index_med_est = db.Column(db.Text, nullable=False)
    dist_index_low_est = db.Column(db.Text, nullable=False)
    dist_index_country_rank = db.Column(db.Text, nullable=False)
    link = db.Column(db.Text, nullable=False)
    date_updated = db.Column(db.Text, nullable=False)
    recorded_date = db.Column(db.Text, nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    status = db.Column(db.Integer, nullable=False)
    highlighted_text = []

    def get_status(self):
        if self.status == 1:
            return 'New'
        elif self.status == 2:
            return 'Doing'
        else:
            return 'Done'
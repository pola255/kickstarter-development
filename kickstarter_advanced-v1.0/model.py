import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier


class Preprocessor(BaseEstimator, TransformerMixin):
    X = None

    def fit(self, X, y=None):
        # pylint: disable=unused-argument
        self.X = X
        return self

    def transform(self, X, y=None):
        # pylint: disable=unused-argument

        # choise of the model columns and building a DataFrame
        features_for_model = ['goal', 'country', 'deadline', 'created_at',
                              'launched_at', 'static_usd_rate', 'category',
                              'profile']
        reduced_df = X[features_for_model]
        numeric_df = reduced_df.copy()

        # Fitting the goal money to usd rate
        numeric_df['goal_in_usd'] = numeric_df['goal'] * \
            numeric_df['static_usd_rate']

        # Delete columns that won't be used
        del numeric_df['goal']
        del numeric_df['static_usd_rate']

        # Building the dates columns
        numeric_df['campaign_duration_ms'] = numeric_df['deadline'] - \
            numeric_df['launched_at']
        numeric_df['campaign_prep_ms'] = numeric_df['launched_at'] - \
            numeric_df['created_at']
        numeric_df['campaign_life_ms'] = numeric_df['deadline'] - \
            numeric_df['created_at']

        # Getting category and subcategory fields of a json data type in
        # category column
        numeric_df['parent_category'] = numeric_df['category']\
            .apply(lambda x: json.loads(x)['slug'].split('/')[0])

        numeric_df['category_name'] = numeric_df['category']\
            .apply(lambda x: json.loads(x)['slug'].split('/')[1])

        del numeric_df['category']

        # Getting state and state_changed_at fields of  a json data type in
        # profile column
        numeric_df['profile_state'] = numeric_df['profile']\
            .apply(lambda x: json.loads(x)['state'])
        numeric_df['profile_state_changed_at'] = numeric_df['profile']\
            .apply(lambda x: json.loads(x)['state_changed_at'])
        del numeric_df['profile']
        numeric_df['profile_state'] = (
            numeric_df['profile_state'] == 'active') * 1

        # Building times: update time and created profile time
        numeric_df['tmp1'] = numeric_df['deadline'] - \
            numeric_df['profile_state_changed_at']
        numeric_df['tmp2'] = numeric_df['launched_at'] - \
            numeric_df['profile_state_changed_at']

        del numeric_df['profile_state_changed_at']
        del numeric_df['deadline']
        del numeric_df['created_at']
        del numeric_df['launched_at']
        self.X = numeric_df
        full_pipeline = self.method_pipeline()
        return full_pipeline.fit_transform(numeric_df)

    def method_pipeline(self):

        # Getting unique data
        parent_categories = self.X['parent_category'].unique()
        category_name = self.X['category_name'].unique()
        country_categories = self.X['country'].unique()

        # Building numerical and categorical pipelines
        numerical_pipeline = Pipeline([
            ('payment_scaler', MinMaxScaler())
        ])
        categorical_pipeline = Pipeline([
            ('one_hot_encoder_parent_categories', self.one_hot_encode(
                'parent_category', parent_categories)),
            ('one_hot_encoder_category_name', self.one_hot_encode(
                'category_name', category_name)),
            ('one_hot_encoder_country', self.one_hot_encode(
                'country', country_categories))
        ])

        full_pipeline = FeatureUnion(
            transformer_list=[
                ('categorical_pipeline',
                 categorical_pipeline),
                ('numerical_pipeline',
                 numerical_pipeline)])
        return full_pipeline

    # one_hot_encode method to encode categorical features
    def one_hot_encode(self, feature_name, categories):
        for category in categories:
            new_feature_name = '{}_{}'.format(feature_name, category)
            self.X[new_feature_name] = (self.X[feature_name] == category) * 1
        del self.X[feature_name]


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """
    preprocessor = Preprocessor()
    model = RandomForestClassifier(n_estimators=200)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

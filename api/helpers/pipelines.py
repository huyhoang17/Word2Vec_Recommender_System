import os
import time
import logging

from annoy import AnnoyIndex
from django.conf import settings as st
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)

from api.helpers.selectors import FeatureSelector
from api.helpers.transformers import (
    NumericalTransformer,
    SequenceTransformer,
    CategoryTransformer,
    MultiColumnLabelEncoder,
    ConvertType
)


def make_scaler(scaler="standard"):
    if scaler == "standard":
        return ("standard", StandardScaler())
    elif scaler == "minmax":
        return ("minmax", MinMaxScaler())
    elif scaler == "normalize":
        return ("normalize", Normalizer())


def make_num_pl(feature_names, strategy="most_frequent", scaler="standard"):

    vec_pipe = [
        ("selector", FeatureSelector(feature_names)),
        ("num_transformer", NumericalTransformer()),
        ("imputer", SimpleImputer(strategy=strategy)),
    ]

    vec_pipe.append(make_scaler(scaler))

    return Pipeline(steps=vec_pipe)


def make_seq_pl(feature_names, dims=64, scaler="standard"):
    vec_pipe = [
        # ('simplify', Simplify()),
        ('selector', FeatureSelector(feature_names)),
        ('seq_transformer', SequenceTransformer()),
        ('tfidf', TfidfVectorizer()),
        ('tsvd', TruncatedSVD(n_components=dims)),
    ]

    vec_pipe.append(make_scaler(scaler))

    return Pipeline(vec_pipe)


def make_cate_pl(feature_names, scaler="standard"):
    vec_pipe = [
        ('selector', FeatureSelector(feature_names)),
        ('seq_transformer', CategoryTransformer()),
        ('custom_mlb', MultiColumnLabelEncoder()),
        ('convert_type', ConvertType()),
    ]

    vec_pipe.append(make_scaler(scaler))

    return Pipeline(vec_pipe)


def make_dr_pineline(dims=32, reducer="tsvd"):
    vec_pipe = []
    if reducer == "tsvd":
        vec_pipe.append(('tsvd', TruncatedSVD(n_components=dims)))
    elif reducer == "pca":
        vec_pipe.append(('pca', PCA(n_components=dims)))

    vec_pipe.append(("norm", Normalizer()))

    return Pipeline(vec_pipe)


def full_pipeline(return_model=False):
    numerical_pipeline = make_num_pl(st.NUMERICAL_FEATURES)
    sequence_pineline = make_seq_pl(st.SEQUENCE_FEATURES)
    category_pineline = make_cate_pl(st.CATEGORICAL_FEATURES)

    f_unions = FeatureUnion(
        transformer_list=[
            ("numerical_pipeline", numerical_pipeline),
            ("sequence_pineline", sequence_pineline),
            ("category_pineline", category_pineline),
        ]
    )

    feature_eng_pl = [
        ('f_unions', f_unions),
        ('dim_reduction', make_dr_pineline(dims=32, reducer="tsvd")),
    ]

    if return_model:
        feature_eng_pl.append(
            ('model', NearestNeighbors(
                n_neighbors=30,
                metric="euclidean", algorithm='kd_tree')
             ),
        )

    logging.info("Pipeline complete")
    return Pipeline(feature_eng_pl)


def make_model(rooms, return_model=False):
    full_pl = full_pipeline(return_model=False)
    embs = full_pl.fit_transform(rooms)
    logging.info(embs.shape)

    # build annoy index
    t = AnnoyIndex(embs.shape[1])
    for index, room_id in enumerate(rooms["id"].values):
        t.add_item(int(room_id), embs[index])

    logging.info("Build annoy index for feature-based")
    start_ = time.time()
    t.build(10)
    t.save(
        os.path.join(
            st.BASE_MODEL,
            "{}.model".format(st.ANNOY_INDEX_FB_KEY)
        )
    )
    logging.info(">>> Took %d's", round(time.time() - start_, 2))

    return full_pl

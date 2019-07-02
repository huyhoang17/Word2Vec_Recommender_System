import datetime
import os
import gc
import logging
import time
import sys

from django.conf import settings as st
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.similarities.index import AnnoyIndexer
import pandas as pd
from tqdm import tqdm

from api import col
from api.helpers.utils import (
    mem_use
)


def filter_(sessions, max_=11, min_=3):
    result = []
    for s in sessions:
        if s <= max_ and s >= min_:
            result.append(True)
        else:
            result.append(False)
    return result


def filter_sessions(df, max_=11, min_=3):
    v_counts = df["custom_session_id"].value_counts()
    no_items_per_session = v_counts.values

    fil = filter_(no_items_per_session, max_=max_, min_=min_)
    result = v_counts[fil]
    return result


def make_df():

    columns = ["custom_session_id", "room_id"]

    days_before = datetime.datetime.now() - \
        datetime.timedelta(days=st.DAYS_BEFORE)
    cur = col.find(
        {"$and": [
            {"room_id": {"$ne": None}},
            {"room.status": "Listed"},
            {"created_at": {"$gte": days_before}}
        ]},
        {"custom_session_id": 1, "room_id": 1, "_id": 0}
    )
    df = pd.DataFrame(list(cur))
    df.columns = columns
    mem_use(df)

    return df


def make_samples(df, sessions, total_mem_use=1000):
    """
    :param df: dataframe to fetch data from mongo's collection,
            included: custom_session_id, room_id
    :param sessions: sessions after filter on range, (index, values)
    :param total_mem_use: max memory used to store session data,
            default to 1000MB
    """
    session_data = []
    index = 0

    # make initial data
    # very SLOWLY interation
    for k, v in tqdm(zip(sessions.index, sessions.values),
                     total=len(sessions)):
        # NOTE: must be convert to string
        session_data.append(list(filter_rooms(df, k).values.astype(str)))

        cur_memory_use = sys.getsizeof(session_data) // 1024
        index += 1
        if index == 1000:
            logging.info("Get size of: %d MB", cur_memory_use)
            if cur_memory_use >= total_mem_use:
                return session_data
            index = 0

    del df
    gc.collect()
    time.sleep(5)
    return session_data


def filter_rooms(df, session_id):
    return df[df['custom_session_id'].isin([session_id])]['room_id'].astype(int)  # noqa


def gen_sessions(df, sessions):
    for k, v in zip(sessions.index, sessions.values):
        # NOTE: must be convert to string
        yield list(filter_rooms(df, k).values.astype(str))


class RoomsGenerator:
    """
    https://rare-technologies.com/word2vec-tutorial/
    """

    def __init__(self, df, sessions):
        self.df = df
        self.sessions = sessions

    def __iter__(self):
        for k, v in tqdm(zip(self.sessions.index, self.sessions.values),
                         total=len(self.sessions)):
            # NOTE: must be convert to string
            yield list(filter_rooms(self.df, k).values.astype(str))

    def __len__(self):
        return len(self.sessions)


class Timer(CallbackAny2Vec):

    def __init__(self, start):
        self.start = start

    def on_epoch_end(self, model):
        logging.info("Take %d's'", time.time() - self.start)


def train_item2vec(df=None, sessions=None, samples=None):
    if df is None and samples is None:
        raise NotImplementedError(
            ">>> Must be specific no items. Can not set `df` and `samples` to None"  # noqa
        )

    if samples is None:
        gen_rooms = RoomsGenerator(df, sessions)
    else:
        gen_rooms = samples

    start_ = time.time()
    model_i2v_path = os.path.join(
        st.BASE_MODEL,
        "{}.model".format(st.ITEM2VEC_KEY)
    )
    if os.path.exists(model_i2v_path):
        logging.info("Load pre-train model")
        model = Word2Vec.load(model_i2v_path)
        logging.info("Vocabulary before re-training: %d", len(model.wv.vocab))

        model.build_vocab(gen_rooms, update=True)
        logging.info("Vocabulary after re-training: %d", len(model.wv.vocab))
        model.train(
            gen_rooms,
            total_examples=model.corpus_count,
            epochs=model.iter,
            callbacks=()
        )
        logging.info(
            "Pre-train model took %d's'",
            time.time() - start_
        )
    else:
        model = Word2Vec(
            gen_rooms,
            sg=st.SG,
            size=st.I2V_DIM,
            window=st.WINDOWS,
            min_count=st.MIN_COUNT,
            workers=st.WORKERS,
            iter=st.EPOCHS,
            sample=st.SAMPLE,
            negative=st.NS,
            compute_loss=st.COMPUTE_LOSS,
            callbacks=[Timer(start_)]
        )

    logging.info("Saving item2vec model")
    model.save(model_i2v_path)

    logging.info("Build annoy index for item2vec model")
    annoy_index = AnnoyIndexer(model, 100)
    annoy_index.save(
        os.path.join(
            st.BASE_MODEL,
            "{}.model".format(st.ANNOY_INDEX_KEY)
        )
    )

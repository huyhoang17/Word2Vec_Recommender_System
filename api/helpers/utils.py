import datetime
import time
from functools import wraps
import logging
from urllib.parse import quote_plus

from django.conf import settings as st
from pymongo import MongoClient


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("- Function {} tooks {}'s".format(func.__name__, end - start))
        return result
    return wrapper


def connect_to(host=st.MONGO_HOST,
               port=st.MONGO_PORT,
               user=st.MONGO_USER,
               password=st.MONGO_PASSWORD,
               db=None,
               col=None):

    if db is None or col is None:
        logging.error("Must be included `db` and `col` fields")

    logging.info("Load :: database:%s - collection:%s", db, col)
    if user is not None and password is not None:
        uri = "mongodb://%s:%s@%s:%s/%s" % (
            quote_plus(user),
            quote_plus(password),
            quote_plus(host),
            quote_plus(port),
            quote_plus(db)
        )
        client = MongoClient(uri)
    else:
        client = MongoClient(host=host, port=int(port))

    db = client[db]
    col = db[col]

    return col


def mem_use(df):
    logging.info(
        "Mem use :: %s MB", df.memory_usage(deep=True).sum() / (1024 ** 2)
    )


def dt_fmt(fmt="%d-%m-%Y-%H-%M-%S"):
    # return datetime.datetime.now().strftime(fmt)
    return time.time()


def convert_time(epoch_time, fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.fromtimestamp(epoch_time).strftime(fmt)

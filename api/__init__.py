import logging

from annoy import AnnoyIndex
from django.conf import settings as st
import gensim
from sklearn.externals import joblib

from api.helpers.utils import connect_to


# logging config
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.root.level = logging.INFO

# initial mongo connection
room_col = connect_to(db=st.MONGO_DB, col=st.ROOMS_COL)
col = connect_to(db=st.MONGO_DB, col=st.LOG_COL)

# load pre-trained model
try:
    model = gensim.models.Word2Vec.load(
        st.MODEL_ITEM2VEC
    )
except Exception as e:
    logging.exception(e)
    model = None

try:
    fb_model = joblib.load(
        st.MODEL_FEATURE_BASED
    )
except Exception as e:
    logging.exception(e)
    fb_model = None

try:
    ann = AnnoyIndex(st.DIMS)
    ann.load(st.MODEL_ANNOY_INDEX_FB)
except Exception as e:
    logging.exception(e)
    ann = None

import os
import logging

from django.conf import settings as st
import numpy as np
from geopy.distance import geodesic
from gensim.similarities.index import AnnoyIndexer
import pandas as pd

from api.helpers.response_format import json_format
from api import (
    model, fb_model, ann,
    col, room_col
)


def get_indexer(fpath, model, room_id):
    if os.path.exists(fpath):
        logging.info("Use annoy_index :: room_id:%s", room_id)
        annoy_index = AnnoyIndexer()
        annoy_index.load(fpath)
        annoy_index.model = model

        return annoy_index
    else:
        # indexer: defaut is None
        return None


def cal_lat_long_location(col,
                          main_id,
                          room_ids,
                          sort=True,
                          return_distance=False,
                          return_sim=False,
                          reverse=False,
                          topn=20):

    main_room = col.find({"id": int(main_id)})[0]
    main_lat_long = (
        main_room["room_address"]["latitude"],
        main_room["room_address"]["longitude"]
    )

    lat_long_dist = []
    for room_id, sim in room_ids:
        try:
            room_info = col.find({"id": int(room_id)})[0]
            lat_long = (
                room_info["room_address"]["latitude"],
                room_info["room_address"]["longitude"]
            )
            dist = geodesic(lat_long, main_lat_long).kilometers
            lat_long_dist.append((room_id, sim, dist))
        except Exception:
            continue

    if sort:
        # sorted by distances
        lat_long_dist.sort(key=lambda x: x[2], reverse=reverse)
    if not return_sim:
        lat_long_dist = [sample[:1] for sample in lat_long_dist]
    if not return_distance:
        lat_long_dist = [sample[:2] for sample in lat_long_dist]
    return lat_long_dist[:topn]


def get_room_by_feature_vector(room_id, topn):
    room = pd.DataFrame(list(room_col.find(
        {"id": int(room_id)},
        {fea: 1 for fea in st.TOTAL_FEATURES}.update({"_id": 0})
    )))
    # transform DataFrame to vector embedding
    emb = fb_model.transform(room)[0]

    room_ids = ann.get_nns_by_vector(emb, topn + 1, include_distances=True)
    return list(zip(room_ids[0], room_ids[1]))[1:]


def get_last_room_session(custom_session_id=None,
                          ip_address=None,
                          no_limit=5000,
                          no_items=10):
    """
    Get last N sessions from mongodb server
    Then filter by custom_session_id field
    """
    cur_rooms = col.find(
        {"room_id": {"$ne": None}},
        {"custom_session_id": 1, "room_id": 1, "ip_address": 1, "_id": 0}
    ).limit(no_limit).sort("_id", -1)

    df = pd.DataFrame(list(cur_rooms))
    if None not in (custom_session_id, ip_address):
        room_ids = df[
            df["custom_session_id"].isin([custom_session_id]) |
            df["ip_address"].isin([ip_address])
        ]["room_id"].values.astype(int)[:no_items]
    elif custom_session_id is None and ip_address is not None:
        room_ids = df[
            df["ip_address"].isin([ip_address])
        ]["room_id"].values.astype(int)[:no_items]

    elif custom_session_id is not None and ip_address is None:
        room_ids = df[
            df["custom_session_id"].isin([custom_session_id])
        ]["room_id"].values.astype(int)[:no_items]
    else:
        return None

    if len(room_ids) == 0:
        return None
    return room_ids


def get_custom_recommender(custom_session_id,
                           ip_address,
                           topn=20):
    """
    Use this function to custom recommender for each user

    :custom_session_id str: session id
    :time_before int: query room_ids viewed from `current_time - time_before`
    to `current_time`

    :return room_ids: recommend rooms
    """

    # get top N last views in current session
    room_ids = get_last_room_session(
        custom_session_id,
        ip_address,
        no_limit=st.NO_LIMIT,
        no_items=st.NO_ITEMS
    )
    logging.info(
        "Room in current session :: room_ids:%s",
        str(room_ids)
    )
    if room_ids is None:
        return json_format(
            code=500,
            message="Custom session id does not existed.",
            data=[],
            errors=True
        )

    if len(room_ids) == 1:
        rooms = model.wv.most_similar(str(room_ids[0]), topn=topn)
        return json_format(
            code=200,
            message="Custom recommender successfully.",
            data=rooms,
            errors=False
        )

    # get average embedding vectors
    mean_emb = []
    for room_id in room_ids:
        try:
            mean_emb.append(model.wv[str(int(room_id))])
        except Exception:
            pass

    if len(mean_emb) == 0:
        return None

    mean_emb = np.mean(mean_emb, axis=0)

    # item2vec model
    rooms = model.wv.similar_by_vector(mean_emb, topn=topn)
    return json_format(
        code=200,
        message="Custom recommender successfully.",
        data=rooms,
        errors=False
    )


def get_room_similar(room_id, topn=20):

    mode = 1
    check_in = str(room_id) in model.wv
    if check_in:
        annoy_index = get_indexer(st.MODEL_ANNOY_INDEX, model, room_id)

        room_ids = model.wv.most_similar(
            str(room_id), topn=topn + 1,
            indexer=annoy_index
        )[1:]
        mode = 1
    else:
        try:
            logging.info("Make feature vector :: room_id:%s", str(room_id))
            room_ids = get_room_by_feature_vector(room_id, topn=topn)
            mode = 0
        except Exception:
            # room_id not existed in database
            error_message = "Room not found"
            logging.info(
                "%s :: room_id:%s",
                error_message, str(room_id)
            )
            return json_format(
                code=500,
                message=error_message,
                data=[],
                errors=False
            )

    # sorterd by lat/long distances
    room_ids = cal_lat_long_location(
        room_col, room_id, room_ids, return_sim=True, topn=topn
    )
    mode = "item2vec" if mode == 1 else "feature-based"
    return json_format(
        code=200,
        message="Recommend by {}".format(mode),
        data=room_ids,
        errors=False
    )

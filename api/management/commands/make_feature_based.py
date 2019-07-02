import os
import gc
import logging
import time

from django.conf import settings as st
from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.externals import joblib

from api import room_col
from api.helpers.pipelines import make_model


class Command(BaseCommand):
    help = "Dump feature-based vector"

    def dump(self):

        # connect to MongoDB server
        rooms = pd.DataFrame(list(room_col.find()))

        # filter unused room
        logging.info("No rooms before filter: %d", len(rooms))
        rooms = rooms[rooms["status"].isin(["Listed"])]
        rooms = rooms[~(
            rooms["content"].isin([None]) |
            rooms["name"].isin([None]) |
            rooms["amenities"].isin([None])
        )]
        rooms = rooms[~rooms["submit_status"].isin([None])]
        logging.info("No rooms after filter: %d", len(rooms))

        # merge room with its address
        room_adds = pd.DataFrame(list(rooms["room_address"]))
        rooms = pd.merge(rooms, room_adds, left_on='id', right_on='room_id')
        del room_adds

        full_pl = make_model(rooms)
        del rooms

        joblib.dump(
            full_pl,
            os.path.join(
                st.BASE_MODEL,
                "{}.pk".format(st.FEATURE_BASED_KEY)
            )
        )

        # clean up
        gc.collect()
        time.sleep(5)

    def handle(self, *args, **kwargs):
        self.dump()
        self.stdout.write(self.style.SUCCESS("Dump feature-based completed"))

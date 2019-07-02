import gc
import time

from django.conf import settings as st
from django.core.management.base import BaseCommand

from api.helpers.item2vec import (
    make_df,
    make_samples,
    filter_sessions,
    train_item2vec
)


class Command(BaseCommand):
    help = "Dump item2vec vector"

    def dump(self):

        df = make_df()
        sessions = filter_sessions(df, st.MAX_CLICK, st.MIN_CLICK)
        gen_rooms = make_samples(df, sessions, 1000)
        train_item2vec(df, sessions, samples=gen_rooms)
        del df, sessions, gen_rooms

        # clean up
        gc.collect()
        time.sleep(5)

    def handle(self, *args, **kwargs):
        self.dump()
        self.stdout.write(self.style.SUCCESS("Dump completed"))

import logging

from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView

from api.forms.room_form import LuxstayRoomForm
from api.helpers.recommenders import (
    get_room_similar,
    get_custom_recommender
)


class RecommenderView(APIView):
    parser_classes = (MultiPartParser, )
    success = "Success"
    failure = "Not existed"

    def post(self, request):
        form = LuxstayRoomForm(request.POST, )
        if not form.is_valid():
            return JsonResponse(form.errors, status=422)

        custom_session_id = form.cleaned_data.get("custom_session_id")
        ip_address = form.cleaned_data.get("ip_address")
        room_id = form.cleaned_data.get("room_id")

        if not custom_session_id:
            custom_session_id = None

        if not ip_address:
            ip_address = None

        # custom recommend for each user
        if custom_session_id is not None or ip_address is not None:
            logging.info(
                "Custom recommender :: custom_session_id:%s - ip_address:%s",
                custom_session_id,
                ip_address or None
            )
            # filter by `custom_session_id` or both
            result = get_custom_recommender(custom_session_id, ip_address, 20)
        else:
            result = get_room_similar(room_id, 20)

        # return list of tuples
        # (room_id, similarity) with item2vec
        # (room_id, distance) with feature-based
        return result

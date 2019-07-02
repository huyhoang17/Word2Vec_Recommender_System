from django import forms

from api.forms.abstract_form import AbstractForm


class LuxstayRoomForm(AbstractForm):
    custom_session_id = forms.CharField(required=False, initial=None)
    ip_address = forms.CharField(required=False, initial=None)

    room_id = forms.IntegerField(required=False, initial=None)

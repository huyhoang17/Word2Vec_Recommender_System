from django import forms


class AbstractForm(forms.Form):
    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data

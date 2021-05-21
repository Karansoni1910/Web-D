from django import forms

class ImageUploadform(forms.Form):
    image = forms.ImageField()
    
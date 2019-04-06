import os, pickle
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
#from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
#auth.authenticate_user()

class DataLoader:

    def __init__(self, german, english):
        self.german_name = german
        self.english_name = english
        # gauth = GoogleAuth()
        # gauth.credentials = GoogleCredentials.get_application_default()
        # self.drive = GoogleDrive(gauth)

    def get_german(self):
        exists = os.path.isfile(self.german_name)
        if exists:
            german_data = {}
            with open(self.german_name, "rb") as f:
                german_data = pickle.load(f)
                print(len(german_data['train'][0]))
                return german_data
        # else:
        #     german = self.drive.CreateFile({'id': '1UqJmu1fvDyNWx2D7joWY9aAmeKRY9tw6'})
        #     german_name = german['title']
        #     german.GetContentFile(german_name)
        #     with open(self.german_name, "rb") as f:
        #         german_data = pickle.load(f)
        #         return german_data



    def get_english(self):
        exists = os.path.isfile(self.english_name)
        if exists:
            english_data = {}
            with open(self.english_name, "rb") as f:
                english_data = pickle.load(f)
                pad = 202 - 202
                pad_arr = [0]*pad
                idx = 0
                for i in english_data['train']:
                    i = i + pad_arr
                    english_data['train'][idx] = i
                    idx += 1
                print(len(english_data['train'][0]))
                return english_data
        # else:
        #     english = self.drive.CreateFile({'id': '1UqJmu1fvDyNWx2D7joWY9aAmeKRY9tw6'})
        #     english_name = english['title']
        #     english.GetContentFile(english_name)
        #     with open(self.english_name, "rb") as f:
        #         english_data = pickle.load(f)
        #         return english_data

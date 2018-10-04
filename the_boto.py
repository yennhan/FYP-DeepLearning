import requests
import cv2
def get_firebase(config):

    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    users = db.child("audio_file").get()
    if users.val() != None:
        for user in users.each():
            voice_url=user.val()
            r= requests.get(voice_url,stream=True)
            #A MP3 file is only binary data, you cannot retrieve its textual part.
            #When you deal with plain textbut for any other binary format, you have to access bytes with doc.content.
            file_name = "the_sound.wav"
            with open(file_name, 'wb') as f:
                f.write(r.content)
        play_sound()
        db.child("audio_file").remove()
        # if the 'q' key is pressed, stop the loop
    else:
        print("Empty object")

def play_sound():
    import subprocess
    subprocess.call(["afplay","the_sound.wav"])
while True:
    import pyrebase
    config = {
        
        "authDomain": "minimumviableproduct-f493e.firebaseapp.com",
        "databaseURL": "https://minimumviableproduct-f493e.firebaseio.com",
        "storageBucket": "gs://minimumviableproduct-f493e.appspot.com",
        "serviceAccount": "/Users/leowyennhan/Desktop/FYP-DeepLearning/minimumviableproduct-f493e-firebase-adminsdk-v8z4z-9bb9769ad7.json"
    }
    get_firebase(config)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from sklearn.impute import SimpleImputer
import parselmouth
from parselmouth.praat import call
import joblib

#home page functions
path = "trainedModel.sav"
clf = joblib.load(path)

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.001, f0min, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    hnr05 = call(harmonicity05, "Get mean", 0, 0)
    harmonicity15 =call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1,1.0)
    hnr15 = call(harmonicity15, "Get mean", 0, 0)
    harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    hnr25 = call(harmonicity25, "Get mean", 0, 0)
    harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
    hnr35 = call(harmonicity35, "Get mean", 0, 0)
    harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
    hnr38 = call(harmonicity38, "Get mean", 0, 0)
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38

    
# Function to predict Parkinson's disease
def predict(file_path):
    file_list = []
    localJitter_list = []
    localabsoluteJitter_list = []
    rapJitter_list = []
    ppq5Jitter_list = []
    localShimmer_list = []
    localdbShimmer_list = []
    apq3Shimmer_list = []
    aqpq5Shimmer_list = []
    apq11Shimmer_list = []
    hnr05_list = []
    hnr15_list = []
    hnr25_list = []
    hnr35_list = []
    hnr38_list = []

    sound = parselmouth.Sound(file_path)
    (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer,
     apq11Shimmer, hnr05, hnr15, hnr25, hnr35, hnr38) = measurePitch(sound, 75, 1000, "Hertz")
    localJitter_list.append(localJitter)  # make a mean F0 list
    localabsoluteJitter_list.append(localabsoluteJitter)  # make a sd F0 list
    rapJitter_list.append(rapJitter)
    ppq5Jitter_list.append(ppq5Jitter)
    localShimmer_list.append(localShimmer)
    localdbShimmer_list.append(localdbShimmer)
    apq3Shimmer_list.append(apq3Shimmer)
    aqpq5Shimmer_list.append(aqpq5Shimmer)
    apq11Shimmer_list.append(apq11Shimmer)
    hnr05_list.append(hnr05)
    hnr15_list.append(hnr15)
    hnr25_list.append(hnr25)
    hnr35_list.append(hnr35)
    hnr38_list.append(hnr38)

    toPred = pd.DataFrame(np.column_stack(
        [localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, localShimmer_list,
         localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, apq11Shimmer_list, hnr05_list, hnr15_list,
         hnr25_list]),
                         columns=["Jitter_rel", "Jitter_abs", "Jitter_RAP", "Jitter_PPQ", "Shim_loc", "Shim_dB",
                                  "Shim_APQ3", "Shim_APQ5", "Shi_APQ11", "hnr05", "hnr15",
                                  "hnr25"])  # add these lists to pandas in the right order

    resp = clf.predict(toPred)
    # Return prediction (1 for Parkinson's disease, 0 for no Parkinson's disease)
    return resp


try:
#alternate method functions
    uploaded_file=st.file_uploader("upload the audio file",type=".wav")
    b=st.button("Detect")
    if b:
        #if file is not uploaded
        if uploaded_file is None:
            st.warning("please upload file")
        else:
        # Assuming 'uploaded_file' is the UploadedFile object
            file_path = "usersupload.wav"  # Provide the desired path to save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            #predictioncall    
            a=predict(file_path=file_path)

            txt=" % analysed "
            my_bar=st.progress(0,text=txt)
            for i in range(100):
                time.sleep(0.1)
                my_bar.progress(i+1,text=txt)
    
    pre=st.button("RESULT")

    if pre:
        if a==0:
            st.success("No Parkinson's disease detected")
            st.balloons()

        else:
            st.warning("Parkinson's disease detected")
except:
    st.error("The recorded audio file is not good!")       
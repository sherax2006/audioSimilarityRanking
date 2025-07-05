

import streamlit as st
import zipfile
import io
import os
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import soundfile as sf
 
from st_audiorec import st_audiorec

st.set_page_config(page_title="Voice MFCC Analyzer", layout="wide")
st.title("Voice MFCC Feature Extraction and Ranking")
 
if 'train_mfccs' not in st.session_state:
    st.session_state.train_mfccs = None
    st.session_state.train_files = []
    st.session_state.csv_path = "train_mfccs.csv"
 
st.header("1. Upload ZIP of Training Audios ")
zip_file = st.file_uploader("Upload a ZIP containing recordings", type=["zip"])
if zip_file and st.button("Extract MFCCs for Training Audios"):
    try:
        st.session_state.train_mfccs = []
        st.session_state.train_files = []
        z = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        for fname in z.namelist():
            if fname.lower().endswith('.ogg'):
                audio_bytes = z.read(fname)
                data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
                st.session_state.train_mfccs.append(np.mean(mfcc, axis=1))
                st.session_state.train_files.append(fname)
        df = pd.DataFrame(
            st.session_state.train_mfccs,
            columns=[f"MFCC_{i+1}" for i in range(13)]
        )
        df.insert(0, "filename", st.session_state.train_files)
        df.to_csv(st.session_state.csv_path, index=False)
        st.success(f"Saved MFCCs for {len(df)} files → {st.session_state.csv_path}")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error processing ZIP: {e}")
 
st.header("2. Test Audio: Upload or Record and Compare")
if st.session_state.train_mfccs is None:
    st.info("Please extract training MFCCs first.")
else:
    method = st.radio("Choose input method:", ["Upload Test Audio", "Record Test Audio"])

    audio_bytes = None
    if method == "Upload Test Audio":
        uploaded = st.file_uploader("Upload a test audio file (.wav, .ogg, .mp3, .flac)",
                                    type=["wav", "ogg", "mp3", "flac"], key="test_upload")
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)
    else:
        st.write("Click the button below to record your test audio.")
        recorded = st_audiorec()
        if recorded is not None:
            audio_bytes = recorded
            st.audio(audio_bytes, format="audio/wav")

    if audio_bytes:
        if st.button("Analyze Test Audio"):
            try:
                buf = io.BytesIO(audio_bytes)
                try:
                    y, sr = sf.read(buf)
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                except Exception:
                    buf.seek(0)
                    y, sr = librosa.load(buf, sr=16000)

                mfcc_test = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_test_mean = np.mean(mfcc_test, axis=1).reshape(1, -1)

                feats = np.vstack(st.session_state.train_mfccs)
                sims = cosine_similarity(mfcc_test_mean, feats)[0]
                ranking_idx = np.argsort(sims)[::-1]

                st.subheader("Similarity Ranking")
                results = [
                    {"Rank": i+1,
                     "filename": st.session_state.train_files[idx],
                     "similarity": f"{sims[idx]:.4f}"}
                    for i, idx in enumerate(ranking_idx)
                ]
                st.table(pd.DataFrame(results))

                st.subheader("Play Top 5 Matches")
                with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z2:
                    for idx in ranking_idx[:5]:
                        fname = st.session_state.train_files[idx]
                        st.markdown(f"**{fname}** (score: {sims[idx]:.4f})")
                        st.audio(z2.read(fname))
            except Exception as e:
                st.error(f"Error analyzing test audio: {e}")

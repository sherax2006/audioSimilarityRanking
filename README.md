 Audio Similarity Ranking

**Audio Similarity Ranking** is a Streamlit-based application that performs MFCC (Mel-Frequency Cepstral Coefficients) feature extraction from audio files and ranks the similarity of a test audio sample against a set of uploaded training samples. It supports both uploading and live recording of test audio.

This project is ideal for **speaker recognition**, **voice matching**, and audio similarity research. Designed with a user-friendly interface and real-time feedback.



User Interface URLS:
(https://github.com/user-attachments/assets/daa0b5d7-e3b8-49f5-b632-e83ff724963e)
(https://github.com/user-attachments/assets/168fc4dc-0676-4dc3-83b2-f0a8916f9ea1)
(https://github.com/user-attachments/assets/2f3c9172-a18c-446a-a9c7-fdc37c7e6e63)

---

 📌 Features

- Upload a ZIP of training audio samples (supports `.ogg` format)
- Extract and store 13-dimensional MFCC features
- Upload or record a test audio sample
- Compare the test sample with training audio using cosine similarity
- Display ranked results and similarity scores
- Playback top 5 matching audio samples directly in the browser

---

Use Case Scenarios

-  **Speaker Identification**: Compare unknown voices to known samples
- **Voice Cloning Detection**: Identify similarity between real and synthetic voices
-  **Research Projects**: Explore MFCC and audio processing methods
-  **Interactive Demos**: Present audio ML applications via browser

---

 ⚙️ How It Works

1. **Upload a ZIP** file of training audio files (only `.ogg` are processed).
2. The app extracts **MFCC features** from each file using `librosa`.
3. User provides a test audio file via upload or recording.
4. MFCC features are extracted from the test audio.
5. **Cosine similarity** is computed between test and training MFCC vectors.
6. Results are ranked and displayed with similarity scores.
7. Top 5 matches are played for auditory comparison.

---

  Installation & Running the App

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/audio-similarity-ranking.git
cd audio-similarity-ranking


Step 2: Create and Activate Virtual Environment:

    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate


Step 3: Install Requirements:
  pip install -r requirements.txt


Step 4: Run the Streamlit App:
  streamlit run app.py

Technologies & Libraries Used:
      | Tool/Library  | Purpose                                    |
| ------------- | ------------------------------------------ |
| Streamlit     | Interactive UI in the browser              |
| Librosa       | Audio processing & MFCC feature extraction |
| NumPy, Pandas | Data manipulation and storage              |
| scikit-learn  | Cosine similarity calculation              |
| SoundFile     | Audio decoding support                     |
| st\_audiorec  | Live in-browser audio recording            |


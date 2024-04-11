import streamlit as st
import pyttsx3
import speech_recognition as sr
from textblob import TextBlob
import openai
import cv2
import numpy as np
from keras.models import load_model
import asyncio
from difflib import SequenceMatcher  # Import SequenceMatcher for string similarity comparison
import comtypes.client

def initialize_engine():
    try:
        # Initialize COM library
        comtypes.client.CoInitialize()
        # Initialize pyttsx3 engine
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        return engine
    except Exception as e:
        print("Error initializing pyttsx3 engine:", e)
        return None

def uninitialize_engine(engine):
    try:
        # Uninitialize COM library
        comtypes.client.CoUninitialize()
    except Exception as e:
        print("Error uninitializing COM library:", e)

# Initialize engine
engine = initialize_engine()

api_data = "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"
openai.api_key = api_data

completion = openai.Completion()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    global engine
    if not engine._inLoop:
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.say(text)
        engine.runAndWait()
    else:
        engine.say(text)

def Reply(question, chat_history=None, total_score=0):
    if chat_history is None:
        chat_history = []
    prompt = ''
    for chat in chat_history:
        prompt += f'User: {chat["user"]}\n Bot: {chat["bot"]}\n'
    prompt += f'User: {question}\n Bot: '
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "I am a Virtual Interview Preparation Coach."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    answer = response['choices'][0]['message']['content'].strip()
    
    # Calculate score based on sentiment analysis
    score = calculate_score(answer)
    
    # Update total score
    total_score += score
    
    return answer, score, chat_history + [{"user": question, "bot": answer}], total_score

# def calculate_score(answer):
#     # Perform sentiment analysis on the answer
#     blob = TextBlob(answer)
#     sentiment_score = blob.sentiment.polarity

#     # Assign score based on sentiment
#     if sentiment_score > 0.5:
#         score = 2  # Excellent
#     elif sentiment_score > 0:
#         score = 1  # Good
#     elif sentiment_score == 0:
#         score = 0  # Neutral
#     elif sentiment_score < -0.5:
#         score = -2  # Very Poor
#     else:
#         score = -1  # Poor
    
#     return score

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #st.sidebar.write("Listening....")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        #st.sidebar.write("Recognizing.....")
        query = r.recognize_google(audio, language='en-in')
        st.write('<div style="padding: 5px; background-color: #f0f0f0; text-align: right; margin-left: auto; margin-right: 0;">User: {}</div>'.format(query), unsafe_allow_html=True)

    except Exception as e:
        #st.sidebar.write("Say That Again....")
        return "None"
    return query

def detect_emotion(face):
    # Convert the input image to grayscale
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Load pre-trained emotion detection model
    emotion_model = load_model('E:/Major project/HOME/pages/model_file_30epochs.h5')

    # Resize and normalize the grayscale image for emotion detection
    face_resized = cv2.resize(gray_face, (48, 48))
    face_normalized = face_resized / 255.0
    face_normalized = face_normalized.reshape((1, 48, 48, 1))  # Ensure the input shape is (1, 48, 48, 1)

    # Perform emotion detection
    predictions = emotion_model.predict(face_normalized)
    
    # Get the dominant emotion
    dominant_emotion_idx = np.argmax(predictions)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    dominant_emotion = emotions[dominant_emotion_idx]

    return dominant_emotion

def load_expected_answers():
    dataset_path = "E:/Major project/HOME/pages/expected_answers.txt"
    try:
        with open(dataset_path, 'r') as file:
            expected_answers = file.readlines()
        return [answer.strip() for answer in expected_answers]
    except FileNotFoundError:
        st.write(f"Failed to open file: {dataset_path}. File not found.")
        return []
    except Exception as e:
        st.write(f"Error occurred while opening file: {dataset_path}. {e}")
        return []
    
def review_answer(user_answer, expected_answers, attended_questions):
    max_similarity = 0
    best_match = "No review available"
    
    # Find the index of the current question in the list of expected answers
    index = attended_questions - 1
    
    if index >= 0 and index < len(expected_answers):
        expected_answer = expected_answers[index]
        similarity = calculate_similarity(user_answer, expected_answer)
        best_match = expected_answer
        max_similarity = similarity

    return best_match, max_similarity

def calculate_similarity(answer1, answer2):
    # Convert answers to lowercase for case-insensitive comparison
    answer1 = answer1.lower()
    answer2 = answer2.lower()

    # Calculate Levenshtein distance
    m = len(answer1)
    n = len(answer2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif answer1[i - 1] == answer2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],      # Insertion
                                   dp[i - 1][j],      # Deletion
                                   dp[i - 1][j - 1])  # Substitution
    
    # Calculate similarity score (inverse of Levenshtein distance normalized by length)
    similarity_score = 1 - (dp[m][n] / max(m, n))
    return similarity_score

async def start_interview():
    st.write("Starting the interview...")
    speak("Starting the interview...")
    questions = get_technical_questions()
    chat_history = []
    total_score = 0
    total_questions = len(questions)
    attended_questions = 0
    
    # Load the dataset containing expected answers
    expected_answers = load_expected_answers()
    
    # Define threshold similarity score
    threshold_similarity = 0.7
    
    for question in questions:
        st.markdown(f'<div style="padding: 5px; background-color: #e0e0e0; text-align: left; margin-left: 0; margin-right: auto;"><strong>Bot:</strong> {question}</div>', unsafe_allow_html=True)
        speak(question)
        
        # Get user's answer
        query = takeCommand().lower()
        user_answer = query
        
        # Review user's answer against dataset
        review, similarity = review_answer(user_answer, expected_answers, attended_questions)
        st.write(f"Review: {review}, Similarity: {similarity}")
        
        # Update total score based on the review
        if similarity >= threshold_similarity:
            total_score += 1  # Assign a positive score if similarity is above the threshold
        else:
            total_score -= 1  # Assign a negative score if similarity is below the threshold
        
        # Display the updated score
        st.write(f"Total Score: {total_score}/{attended_questions}")
        
        attended_questions += 1
        
        # Check if user says "thank you" to end the chat
        if "thank you" in query:
            break

    # Display total score in big font at the end
    st.title(f"Total Score: {total_score}/{attended_questions}")
    st.write(f"Out of {total_questions} questions attended.")

def get_technical_questions():
    # Define the path to the text file containing technical interview questions
    file_path = r'E:/Major project/HOME/interview_questions.txt'
    try:
        with open(file_path, 'r') as file:
            questions = file.readlines()
        return [question.strip() for question in questions]
    except FileNotFoundError:
        st.write(f"Failed to open file: {file_path}. File not found.")
        return []
    except Exception as e:
        st.write(f"Error occurred while opening file: {file_path}. {e}")
        return []

async def camera_feed():
    # Display the camera feed in the sidebar
    st.sidebar.title("Camera Feed")
    video_feed = st.sidebar.empty()

    # Initialize session state
    if "stop_camera" not in st.session_state:
        st.session_state.stop_camera = False

    # Capture video from the camera
    video_capture = cv2.VideoCapture(0)

    # Continuously update the camera feed
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame_rgb, 1.1, 4)
        
        # Draw rectangles around the faces and display emotion
        for (x, y, w, h) in faces:
            face = frame_rgb[y:y+h, x:x+w]
            emotion = detect_emotion(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the frame in the sidebar
        video_feed.image(frame, channels="BGR")

        # Check for a stop event
        if st.session_state.stop_camera:
            break

    # Release the camera and close OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()

async def main():
    st.title("Interview Preparation Coach")
    if st.button("Start Interview"):
        # Run the interview and camera feed concurrently using asyncio.gather()
        await asyncio.gather(start_interview(), camera_feed())
uninitialize_engine(engine)
if __name__ == '__main__':
    asyncio.run(main())

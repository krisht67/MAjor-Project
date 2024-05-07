import streamlit as st
import pyttsx3
import speech_recognition as sr
import openai
import cv2
from bs4 import BeautifulSoup
import requests
import numpy as np
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from keras.models import load_model
import asyncio
from difflib import SequenceMatcher  # Import SequenceMatcher for string similarity comparison
import comtypes.client

TOPIC_URLS = {
    "Data Structures": "https://www.geeksforgeeks.org/commonly-asked-data-structure-interview-questions-set-1/",
    "Logic System Design": "https://www.geeksforgeeks.org/top-10-system-design-interview-questions-and-answers/",
    "Object-Oriented Programming": "https://www.geeksforgeeks.org/oops-interview-questions/"
}

emotion_scores = {
    "Angry": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happy": 9,
    "Sad": 5,
    "Surprise": 8,
    "Neutral": 7
}

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
    
def generate_pdf(feedback):
    # Get the directory path where the script is located
    script_dir = os.path.dirname(__file__)
    # Define the path to save the PDF
    pdf_path = os.path.join(script_dir, r"C:\Users\RIZWAN AT\OneDrive\Desktop\krishna\MAjor-Project\pages\interview_feedback.pdf")
    
    # Create a PDF document
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Add content to the PDF
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Interview Feedback")
    c.drawString(100, 730, "-" * 60)
    c.drawString(100, 710, feedback)
    
    # Save the PDF
    c.save()
    return pdf_path

def uninitialize_engine(engine):
    try:
        # Uninitialize COM library
        comtypes.client.CoUninitialize()
    except Exception as e:
        print("Error uninitializing COM library:", e)

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
    emotion_model = load_model(r'C:\Users\RIZWAN AT\OneDrive\Desktop\krishna\MAjor-Project\model_file_30epochs.h5')

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

    # Assign a score based on the detected emotion
    
    emotion_score = emotion_scores.get(dominant_emotion, 0)  # Default score is 0 for unknown emotions
    
    return dominant_emotion, emotion_score
    return dominant_emotion

def load_expected_answers(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    answers = []
    paragraphs = soup.find_all('p')
    answers = [paragraph.text for paragraph in paragraphs[2:]]  # Skip the first two paragraphs
    return answers
    
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
    # Interview preparation
    selected_topic = st.selectbox("Select Interview Topic", list(TOPIC_URLS.keys()))
    topic_url = TOPIC_URLS[selected_topic]
    questions = get_technical_questions(topic_url)
    expected_answers = load_expected_answers(topic_url)
    
    # Interview loop
    chat_history = []
    total_score = 0
    for question in questions:
        st.markdown(f'<div style="padding: 5px; background-color: #e0e0e0; text-align: left; margin-left: 0; margin-right: auto;"><strong>Bot:</strong> {question}</div>', unsafe_allow_html=True)
        speak(question)
        
        user_answer = takeCommand().lower()
        
        # Review user's answer against dataset
        review, similarity = review_answer(user_answer, expected_answers, len(chat_history) + 1)
        
        # Update total score based on the review
        if similarity >= 0.6:
            total_score += 1
        elif similarity < 0.4:
            total_score -= 1
        
        # Print the score, similarity index, and review of the answer
        st.write(f"Question: {question}")
        st.write(f"Score: {total_score}")
        st.write(f"Similarity: {total_score}")
        st.write(f"Review: {review}")
        
        # Add user's question-answer pair to chat history
        chat_history.append({"user": question, "bot": review})
        
        # Check if user says "thank you" to end the chat
        if "thank you" in user_answer:
            break
    
    # Generate feedback
    total_score = max(0, total_score)  # Ensure total_score is non-negative
    feedback = generate_feedback(total_score, len(questions))
    st.write("Feedback from AI:", feedback)
    
    # Generate and download the PDF
    pdf_path = generate_pdf(feedback)
    st.markdown(f'<a href="{pdf_path}" download="interview_feedback.pdf">Click here to download PDF</a>', unsafe_allow_html=True)

def generate_feedback(total_score, total_questions):
    if total_questions == 0:
        return "No questions were asked. Unable to provide feedback."
    
    percentage_score = (total_score / total_questions) * 100
    
    if percentage_score >= 80:
        return "You performed exceptionally well! Congratulations!"
    elif percentage_score >= 60:
        return "You performed above average. Keep up the good work!"
    elif percentage_score >= 40:
        return "Your performance was average. Try to improve in areas of weakness."
    else:
        return "Your performance was below expectations. Focus on improving your skills."

def get_technical_questions(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    questions = []
    for heading in soup.find_all('h3'):
        questions.append(heading.text)
    return questions

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
            emotion, _ = detect_emotion(face)  # Get emotion as string
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, str(emotion), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Ensure emotion is converted to string

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

if __name__ == '__main__':
    asyncio.run(main())

import streamlit as st
import pyttsx3
import speech_recognition as sr
from textblob import TextBlob
import openai
import cv2
import numpy as np
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from keras.models import load_model
import asyncio
from difflib import SequenceMatcher  # Import SequenceMatcher for string similarity comparison
import comtypes.client

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
    pdf_path = os.path.join(script_dir, "interview_feedback.pdf")
    
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
    emotion_model = load_model(r'C:\Users\ASUS\Desktop\krishna\MAjor-Project\emoton.h5')

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

def load_expected_answers():
    dataset_path = r'C:\Users\ASUS\Desktop\krishna\MAjor-Project\expected_answers.txt'
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
    total_score = 1
    total_questions = len(questions)
    attended_questions = 0
    
    # Load the dataset containing expected answers
    expected_answers = load_expected_answers()
    
    # Define threshold similarity score
    t1 = 0.4
    t2 =0.6
    
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
        if t1<=similarity < t2:
            total_score += 1
        elif similarity > t2:
            total_score+=2
            # Assign a positive score if similarity is above the threshold
        else:
            total_score -= 1  # Assign a negative score if similarity is below the threshold
        
        # Display the updated score
        st.write(f"Total Score: {total_score}")
        st.write(f"Emotion score :{emotion_scores}")
        attended_questions += 1
        
        # Check if user says "thank you" to end the chat
        if "thank you" in query:
            break
    total_score-2
    # Display total score in big font at the end
    st.title(f"Total Score: {total_score}/{attended_questions}")
    st.write(f"Out of {total_questions} questions attended.")

    feedback = generate_feedback(total_score, total_questions)
    st.write("Feedback from AI:", feedback)
    
    # Generate and download the PDF
    pdf_path = generate_pdf(feedback)
    st.markdown(f"### [Download Feedback as PDF]({pdf_path})", unsafe_allow_html=True)
    
    # Generate and download the PDF
    
    
    # Ask for feedback
    st.write("Please provide your feedback on the interview:")
    user_feedback = st.text_area("Feedback")
    st.write("Thank you for your feedback!")
    
    
def generate_feedback(total_score, total_questions):
    if total_questions == 0:
        return "No questions were asked. Unable to provide feedback."
    
    percentage_score = (total_score / (total_questions * 10)) * 100
    
    if percentage_score >= 80:
        return "You performed exceptionally well! Congratulations!"
    elif percentage_score >= 60:
        return "You performed above average. Keep up the good work!"
    elif percentage_score >= 40:
        return "Your performance was average. Try to improve in areas of weakness."
    else:
        return "Your performance was below expectations. Focus on improving your skills."
def get_technical_questions():
    # Define the path to the text file containing technical interview questions
    file_path = r'C:\Users\ASUS\Desktop\krishna\MAjor-Project\interview_questions.txt'
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
uninitialize_engine(engine)
if __name__ == '__main__':
    asyncio.run(main())

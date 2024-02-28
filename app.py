import cv2
import streamlit as st
import tempfile
import os
import shutil
import main
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

st.title('Video Recorder')
target_dir='vedio'
st.subheader("how much fames you want to record")   
fr=st.slider('Select the frame rate', 20, 200, 30)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
# st.write(uploaded_file)

if uploaded_file is not None:
    # Save the uploaded file to a known location
    file_path = os.path.join("vedio", uploaded_file.name) # is the name of the file to save the recorded vedio
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write(file_path)        
# Move or copy the recorded video to a permanent location
def move_video(temp_file_path, target_dir):
    shutil.move(temp_file_path, target_dir)  # or shutil.copy

# Usage:

# Function to record video
def record_video(fr):
    # import cv2

# Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('vedio//output.mp4', fourcc, 20.0, (640, 480)) # 20.0 is the frame rate and (640, 480) is the frame size and output.mp4 is the name of the file

    # Open the default camera
    cap = cv2.VideoCapture(0) # 0 is for inbuilt camera if not work then chnage the value to 1 or 2
    cap.set(3,640)
    cap.set(4,480)
    FRAME_WINDOW = st.image([])
    k=0
    while(cap.isOpened() and stop_recording==False and k<fr):
        k+=1
        ret, frame = cap.read()
        if ret==True:
            # Write the frame into the file 'output.mp4'
            out.write(frame)
            # Display the resulting frame
            # cv2.imshow('frame',frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            # Press 'q' to exit the video recording loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything when the job is done
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Flag to control the recording process
stop_recording = True

# Record video when the start button is clicked
if st.button('Start Recording'):
    stop_recording = False
    video_path = record_video(fr)
    # move_video(video_path, target_dir= target_dir)
    
    st.success(f"Video recorded successfully at {video_path}")

    # Display the recorded video
    st.video(video_path)

    # Optionally, delete the temporary file after displaying the video
    # os.unlink(video_path)

# Stop recording when the stop button is clicked

   
def vedio_processing(path):
    # ved_path='output.mp4'
    os.system(f'python main.py {path} {fr}')
    
def calculation():
     
    with open('data_js.json', 'r') as f: # Open the file in read mode
        new_data = json.load(f)

    if len(new_data) == 2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img = mpimg.imread(new_data['0']['image_path'])
        ax[0].imshow(img)
        ax[0].axis('off')
        
        ax[1].pie(new_data['average'], labels=['Looking away', 'bored', 'Confused', 'drowsy', 'Engaged', 'Frustrated'], startangle=90, autopct='%1.1f%%')
        
        st.pyplot(fig)  
    else:
        fig, ax = plt.subplots(len(new_data)-1, 2, figsize=(10, 5))
        avg = new_data['average']
        del new_data['average']
        
        for i, item in enumerate(new_data):
            img = mpimg.imread(new_data[item]['image_path'])
            ax[i, 0].imshow(img)
            ax[i, 0].axis('off')
            
            ax[i, 1].title.set_text('Student Engagement percentage')
            y = new_data[item]['percentage']
            x = ['Looking away', 'bored', 'Confused', 'drowsy', 'Engaged', 'Frustrated']
            ax[i, 1].pie(y, labels=x, autopct='%1.1f%%') 
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.title.set_text('Average Student Engagement percentage')
        ax1.pie(avg, labels=['Looking away', 'bored', 'Confused', 'drowsy', 'Engaged', 'Frustrated'], startangle=90, autopct='%1.1f%%')
        plt.tight_layout()
        st.pyplot(fig)
        st.pyplot(fig1)

    st.write("The video has been processed successfully!")

if st.button('process recording'):
    stop_recording = True
    if stop_recording:
        st.warning("Recording stopped")  
        with st.spinner('Running...'):
            vedio_processing('output.mp4')
            calculation()
    else:
        st.error("Please stop the recording first")        
          
if st.button("process vedio"):
    # st.load
    if uploaded_file is not None:
        with st.spinner('Running...'):
            
            vedio_processing(file_path)
            
            calculation()
        st.success('Done!')    
    else:
        st.error("Please upload a video file first")    
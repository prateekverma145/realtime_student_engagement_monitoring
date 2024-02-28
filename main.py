import cv2
import numpy as np
import ast
from tensorflow.keras.models import load_model  
import os
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
import face_recognition
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import json
import logging

# time.time()%10000
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Suppress other warnings if necessary
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class module:
    en=None
    # status=None
    image_path=None
    prob=None
    frames=0
    def __init__(self):
        self.en=None
        self.image_path=None
        prob=None
        frames=0
        # self.status={
        #     "Looking away":0,
        #     "bored":0,
        #     "Confused":0,
        #     "drowsy":0,
        #     "Engaged":0,
        #     "Frustrated":0
        # }
        
def cal_dot(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def pre_process_to_dict(data):
    dict={}
    md={
    "image_path": None,
    "percentage": None, 
    }
    for j,i in enumerate( data):
        d=md.copy()
        d['image_path']=str(i.image_path)
        print(i.frames)
        d['percentage']=list(np.apply_along_axis(lambda x:(x*100)/(i.frames),0,i.prob))
        dict[j]=d
    return dict    
    
def sum_prob(p1,p2):
    if p1 is None:
        return p2
    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1 + p2

SIZE=128
def pre_process_image(img, display=False):
    #img = cv2.imread(img)
    img = cv2.resize(img, (SIZE, SIZE))
    img = np.array(img, dtype=np.float32)
    img= img/255.0
    img = np.reshape(img, (-1, SIZE, SIZE, 3))
    return img

data=[]


def database(data,face,fl,k,cropface):
    
    if face.any():
        if data==[]:
            t=module()
            enc=face_recognition.face_encodings(face,[fl])
            t.en=[enc[0]].copy()
          
            timestamp =time.time()%10000
            t.prob=sum_prob(t.prob,k)
            pp=f'saving\\image_{timestamp}.jpg' # path to save the image

            cv2.imwrite(pp,cropface)
            t.image_path=pp
           
            data.append(t)
            return
        
        enc=face_recognition.face_encodings(face,[fl])
        for i in data:
     
            if cal_dot(i.en,enc[0])>0.9:
                
                i.frames+=1
                i.prob=sum_prob(i.prob,k)
                   
                return 
        t=module()
        t.en=[enc[0]].copy()
         
        t.frames+=1
        timestamp = time.time()%10000
        t.prob=sum_prob(t.prob,k)

        pp=f'saving\\image_{timestamp}.jpg' # path to save the image
        cv2.imwrite(pp,cropface)
        t.image_path=pp
        data.append(t)
     
    else:
        print('no face')
  
    return

import json

def render(ved_path,model,fr):
    cap=cv2.VideoCapture(ved_path)
    cap.set(3,640)
    cap.set(4,480)
    prob=[0,0,0,0,0,0,0]
    fk=0
    while cap.isOpened() and fk<fr:
        fk+=1
        ret,frame=cap.read()
        if not ret:
            # print('no frame'    )
            break
        faces=face_recognition.face_locations(frame)
        if(len(faces)==0):
            # print("hbgureb")
            continue
        for i in faces:
            x,y,w,h=i
            l=None
            roi1=frame[x:w,h:y,:]
            roi1=cv2.resize(roi1,(128,128))
            roi=pre_process_image(roi1)
            pred=model.predict(roi)
            
            prob[0]+=pred[0][0]
            prob[1]+=pred[0][1]
            prob[2]+=pred[0][2]
            prob[3]+=pred[0][3]
            prob[4]+=pred[0][4]
            prob[5]+=pred[0][5]
            prob[6]+=1
            # print(pred)    
            database(data,frame,i,pred[0],roi1)
        
        # cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF==ord('q'):
            break
    
    # Calculate average probabilities
    if(data!=[]):
        ff = list(map(lambda x: x/prob[6], prob))
        dict = pre_process_to_dict(data)
        dict['average'] = ff[:-1]
        
        
    
        with open('data1.txt', 'w') as f:
            f.write(str(dict))
        
        cap.release()
        cv2.destroyAllWindows()
        
    return ff[:-1]
    
    

    
def main():
    model=load_model('Student Engagement Model.h5')   # path to the model
    ved_path=sys.argv[1]
    fr=int(sys.argv[2])
    if os.path.exists(ved_path)==False:
        ved_path=0
        
    try:
        ff=render(ved_path,model,fr)
        # # print(ff)
        with open('data1.txt', 'r') as f:
            data = f.read()
            # print(data)
        new_data = ast.literal_eval(data)
        with open('data_js.json','w') as f:
            json.dump(new_data,f)
           
                          
    except:
        print("some error occured")     
          
    
    
if __name__=='__main__':
    main()
    
    

  
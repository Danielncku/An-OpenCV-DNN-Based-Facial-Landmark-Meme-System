import cv2
import numpy as np
import os, urllib.request, time
from PIL import Image, ImageSequence
from collections import deque


DETECT_TIME = 0.8     
HOLD_TIME = 2.0       
MIN_SCORE = 0.6       


# download models

def setup_models():
    files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "lbfmodel.yaml": "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    }
    for f, url in files.items():
        if not os.path.exists(f):
            print(f"正在下載 {f}...")
            urllib.request.urlretrieve(url, f)

setup_models()

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
landmark = cv2.face.createFacemarkLBF()
landmark.loadModel("lbfmodel.yaml")


# gif handling
def load_gif(path):
    if not os.path.exists(path):
        print(f"找不到檔案: {path}")
        return None
    frames = []
    with Image.open(path) as img:
        for f in ImageSequence.Iterator(img):
            rgba = np.array(f.convert("RGBA"))
            frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return frames

def show_gif(win, frames, tick):
    if frames:
        frame = frames[tick % len(frames)]
        cv2.imshow(win, frame)


# feature extraction

def mouth_ratio(p):
    return np.linalg.norm(p[62]-p[66]) / (np.linalg.norm(p[48]-p[54])+1e-6)

def eye_ratio(p):
    l = np.linalg.norm(p[37]-p[41]) / (np.linalg.norm(p[36]-p[39])+1e-6)
    r = np.linalg.norm(p[43]-p[47]) / (np.linalg.norm(p[42]-p[45])+1e-6)
    return (l+r)/2

def yaw_ratio(p):
    l_dist = np.linalg.norm(p[1]-p[30])
    r_dist = np.linalg.norm(p[15]-p[30])
    return abs(l_dist - r_dist) / (l_dist + r_dist + 1e-6)


#  ROI 

def hand_presence(frame, box):
    sx, sy, ex, ey = box
    h, w = frame.shape[:2]
    roi_h = int((ey-sy)*0.8)
    roi_w = int((ex-sx)*0.7)

    left = frame[ey:min(h, ey+roi_h), max(0, sx-roi_w):sx]
    right = frame[ey:min(h, ey+roi_h), ex:min(w, ex+roi_w)]

    if left.size == 0 or right.size == 0:
        return False, False

    l_std = np.std(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    r_std = np.std(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    
    
    return l_std > 20, r_std > 20


def main():
    
    gifs = {
        "YELL": load_gif("gif/yell.gif"),
        "SMART": load_gif("gif/smart.gif"),
        "CRAZY": load_gif("gif/crazy.gif"),
        "CRY": load_gif("gif/cry.gif"),
        "SIGMA": load_gif("gif/sigma.gif"),
    }

    buf = {k: deque(maxlen=12) for k in gifs}
    current_state = "NONE"
    locked_until = 0
    meme_open = False

    cap = cv2.VideoCapture(0)
    tick = 0

    print("program started. ")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        face_net.setInput(blob)
        dets = face_net.forward()

        this_frame_action = {k: False for k in gifs}

        for i in range(dets.shape[2]):
            if dets[0, 0, i, 2] > 0.6:
                box = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, lm = landmark.fit(gray, np.array([[box[0], box[1], box[2]-box[0], box[3]-box[1]]]))
                
                if not ok: continue
                p = lm[0][0]

                # feature extraction
                mr = mouth_ratio(p)
                er = eye_ratio(p)
                yr = yaw_ratio(p)
                l_hand, r_hand = hand_presence(frame, box)

              
                
                # 1. SIGMA
                this_frame_action["SIGMA"] = yr > 0.21 and mr < 0.2
                
                # 2. SMART 
                this_frame_action["SMART"] = l_hand and not r_hand and yr < 0.1
                
                # 3. YELL
                this_frame_action["YELL"] = l_hand and r_hand and mr > 0.45
                
                # 4. CRY
                is_sad = (p[48][1] + p[54][1]) / 2 > p[62][1]
                this_frame_action["CRY"] = l_hand and r_hand and is_sad and mr < 0.35
                
                # 5. CRAZY 
                this_frame_action["CRAZY"] = l_hand and r_hand and er > 0.2 and 0.2 <= mr <= 0.52

                # face box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
                break

        
        now = time.time()
        for k in gifs:
            buf[k].append(this_frame_action[k])

        # state machine
        if now > locked_until:
            best_match = "NONE"
            highest_score = 0
            
            for k in gifs:
                score = sum(buf[k]) / len(buf[k])
                if score >= MIN_SCORE and score > highest_score:
                    highest_score = score
                    best_match = k
            
            if best_match != "NONE":
                current_state = best_match
                locked_until = now + HOLD_TIME
            else:
                current_state = "NONE"

        #  gif state
        if current_state != "NONE":
            show_gif("MEME_WINDOW", gifs[current_state], tick)
            meme_open = True
            cv2.putText(frame, f"STATUS: {current_state}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            if meme_open:
                try: cv2.destroyWindow("MEME_WINDOW")
                except: pass
                meme_open = False

        cv2.imshow("Meme Master AI", frame)
        tick += 1
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
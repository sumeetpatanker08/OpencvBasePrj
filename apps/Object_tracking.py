import streamlit as st
import numpy as np
import cv2



def app():

    st.title("Object Tracking section")
    st.write('Please make sure Camera is working and enabled, press Escape button to exit')
    track_options = st.selectbox('Object Tracking Options',('Optical Flow','MeanShift and CamShift Tracking','Object Tracking API'))

    if track_options == 'Optical Flow':
        col1,col2= st.beta_columns([1,1])

        with col1:
            st.subheader('lukas Kanade Optical Flow')
            run_point = st.button('Run Point Optical Flow')
            setting_point = st.beta_expander('Settings')
            with setting_point:
                no_points = st.slider('Number of points to detect',1,100,10,1)
                qualityLevel = st.slider('Quality Level',0.0,10.0,0.3,0.1)
                minDistance = st.slider('MinDistance',0,20,7,1)
                blockSize = st.slider('BlockSize',0,100,7,1)
                winSize = st.slider('WindowSize',50,1000,200,10)
                maxLevel = st.slider('MaxLevel',1,5,2,1)
                cri_count = st.slider('CriteriaCount',1,20,10,1)
                cri_eps = st.slider('CriteriaEPS',0.001,2.0,0.03,0.001)
                corner_track_params = dict(maxCorners=no_points,qualityLevel=qualityLevel,minDistance=minDistance,blockSize=blockSize)
                lk_params = dict(winSize=(winSize,winSize),maxLevel=maxLevel,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,cri_count,cri_eps))
                
                if run_point:
                    cap = cv2.VideoCapture(0)
                    ret,prev_frame = cap.read()
                    prev_gray= cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
                    prevPts = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
                    mask = np.zeros_like(prev_frame)

                    while True:
                        ret,frame = cap.read()
                        frame_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        nextPts,status,err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPts,None,**lk_params)
                        good_new = nextPts[status==1]
                        good_prev = prevPts[status==1]
                        for i,(new,prev) in enumerate(zip(good_new,good_prev)):
                            x_new,y_new = new.ravel()
                            x_prev,y_prev = prev.ravel()
                            mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0,2))
                            frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
                        img = cv2.add(frame,mask)
                        cv2.imshow('Tracking',img)
                        k= cv2.waitKey(1)
                        if k == 27:
                            break
                        prev_gray = frame_gray.copy()
                        prevPts = good_new.reshape(-1,1,2)
                    cv2.destroyAllWindows()
                    cap.release()

        with col2:
            st.subheader('Dense Optical Flow')
            run_dense = st.button('Run Dense Optical Flow')
            setting_dense = st.beta_expander('Settings')
            with setting_dense:
                pyr_scale = st.slider('Pyramid Scale',0.05,2.0,0.5,0.01)
                levels = st.slider('Levels',1,5,3,1)
                winsize = st.slider('Window Size',1,100,15,1)
                iterations = st.slider('Iterations',1,20,3,1)
                poly_n = st.slider('Poly_n',1,10,5,1)
                poly_sigma = st.slider('Poly_sigma',0.1,10.0,1.2,0.1)
                flags = st.slider('Flags',0,10,0,1)
                
                if run_dense:
                    cap = cv2.VideoCapture(0)
                    ret,frame1 = cap.read()
                    prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                    hsv_mask = np.zeros_like(frame1)
                    hsv_mask[:,:,1] = 255
                    while True:
                        ret,frame2 = cap.read()
                        nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg,None,pyr_scale,levels,
                                                            winsize,iterations,poly_n,poly_sigma,flags)
                        mag,ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True)
                        hsv_mask[:,:,0] = ang/2
                        hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                        bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
                        cv2.imshow('frame',bgr)
                        k = cv2.waitKey(1)
                        if k == 27:
                            break
                        prvsImg=nextImg
                    cv2.destroyAllWindows()
                    cap.release()
    elif track_options == 'MeanShift and CamShift Tracking':
        col1,col2= st.beta_columns([1,1])
        with col1:
            st.subheader('MeanShift Tracking')
            run_ms = st.button('Run MeanShift Tracking')
            setting_ms= st.beta_expander('Settings')
            with setting_ms:
                cri_count = st.slider('CriteriaCount',1,20,10,1)
                cri_eps = st.slider('CriteriaEPS',0.001,2.0,1.0,0.001)

                if run_ms:
                    cap = cv2.VideoCapture(0)
                    ret,frame = cap.read()
                    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                    face_rects = face_cascade.detectMultiScale(frame)
                    (face_x,face_y,w,h) = tuple(face_rects[0])
                    track_window = (face_x,face_y,w,h)
                    roi = frame[face_y:face_y+h,face_x:face_x+w]
                    hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
                    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
                    cv2.normalize(roi_hist,0,255,cv2.NORM_MINMAX)
                    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,cri_count,cri_eps)
                    while True:
                        ret,frame = cap.read()
                        if ret == True:
                            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                            ret,track_window = cv2.meanShift(dst,track_window,term_crit)
                            x,y,w,h = track_window
                            img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
                            cv2.imshow('img',img2)
                            k = cv2.waitKey(1)
                            if k==27:
                                break
                        else:
                            st.write('Unable to connect to camera')
                            break
                    cv2.destroyAllWindows()
                    cap.release()

        with col2:
            st.subheader('CamShift Tracking')
            run_cs = st.button('Run CamShift Tracking')
            setting_cs= st.beta_expander('Settings')
            with setting_cs:
                cri_count_cs = st.slider('CriteriaCount CamShift',1,20,10,1)
                cri_eps_cs = st.slider('CriteriaEPS CamShift',0.001,2.0,1.0,0.001)

                if run_cs:
                    cap = cv2.VideoCapture(0)
                    ret,frame = cap.read()
                    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                    face_rects = face_cascade.detectMultiScale(frame)
                    (face_x,face_y,w,h) = tuple(face_rects[0])
                    track_window = (face_x,face_y,w,h)
                    roi = frame[face_y:face_y+h,face_x:face_x+w]
                    hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
                    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
                    cv2.normalize(roi_hist,0,255,cv2.NORM_MINMAX)
                    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,cri_count_cs,cri_eps_cs)
                    while True:
                        ret,frame = cap.read()
                        if ret == True:
                            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                            ret,track_window = cv2.CamShift(dst,track_window,term_crit)
                            pts = cv2.boxPoints(ret)
                            pts = np.int0(pts)
                            img2 = cv2.polylines(frame,[pts],True,(0,0,255),5)
                            cv2.imshow('img',img2)
                            k = cv2.waitKey(1)
                            if k==27:
                                break
                        else:
                            st.write('Unable to connect to camera')
                            break
                    cv2.destroyAllWindows()
                    cap.release()
    elif track_options == 'Object Tracking API':
        options = {'MIL':'cv2.TrackerMIL_create()','KCF':'cv2.TrackerKCF_create()'}
        #choice = st.radio('TrackingAPIs',tuple(options.keys()))
        run_mil = st.button('Run MIL Tracker')
        if run_mil:
            choice = 'MIL'
            cap = cv2.VideoCapture(0)
            ret,frame = cap.read()
            roi = cv2.selectROI(frame,False)
            tracker = eval(options[choice])
            ret = tracker.init(frame,roi)
            while True:
                ret,frame = cap.read()
                success, roi = tracker.update(frame)
                (x,y,w,h) = tuple(map(int,roi))
                if success:
                    p1 = (x,y)
                    p2 = (x+w,y+h)
                    cv2.rectangle(frame,p1,p2,(0,255,0),3)
                else:
                    cv2.putText(frame,'Failure to Detect Tracking',(100,200),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),5)
                cv2.putText(frame,choice,(20,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                cv2.imshow(choice,frame)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            cap.release()
            cv2.destroyAllWindows()


#app()
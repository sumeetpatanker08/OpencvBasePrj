import streamlit as st
import numpy as np
import cv2
from matplotlib import cm


def app():

    st.title("Object Detection section")
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    col1,col2,col3 = st.beta_columns([1,1,1])
    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        fix_img = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)

        with col1:
            st.subheader('Original Image')
            st.image(fix_img)

        with col3:
            option = st.selectbox('Object Detection Options',('Template Matching','Corner Detection','Edge Detection','Grid Detection','Contour Detection','Feature Matching','Watershed','FaceDetection','EyesDetection','Live Face Detection'))
            if option == 'Template Matching':
                temp_match = st.beta_expander('Template Matching Options',expanded=True)
                with temp_match:
                    sub_img = st.file_uploader("Upload Template image to be matched", type=["png", "jpg", "jpeg"])
                    if sub_img is not None:
                        sub_img_bytes = np.asarray(bytearray(sub_img.read()), dtype=np.uint8)
                        opencv_subimage = cv2.imdecode(sub_img_bytes, 1)
                        fix_subimg = cv2.cvtColor(opencv_subimage,cv2.COLOR_BGR2RGB)
                        fix_img_copy = fix_img.copy()
                        with col1:
                            st.header('Image to be Matched')
                            st.image(fix_subimg)
                        method_options = ('TM_CCOEFF','TM_CCOEFF_NORMED','TM_CCORR','TM_CCORR_NORMED','TM_SQDIFF','TM_SQDIFF_NORMED')
                        sel_method_options = st.selectbox('Matching Methods',method_options)
                        res = cv2.matchTemplate(fix_img_copy,fix_subimg,eval('cv2.' + sel_method_options))
                        min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
                        if sel_method_options in ['TM_SQDIFF','TM_SQDIFF_NORMED']:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        h,w,c = fix_subimg.shape
                        bottom_right = (top_left[0]+w,top_left[1]+h)
                        cv2.rectangle(fix_img_copy,top_left,bottom_right,(255,0,0),10)
                        with col2:
                            st.subheader('Template Detected')
                            st.image(fix_img_copy,use_column_width=True)
            elif option =='Corner Detection':
                cor_det = st.beta_expander('Corner Detection Options',expanded=True)   
                with cor_det:
                    fix_img_gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                    sel_cor_det = st.selectbox('Corner Dection Methods',('Harris Detector','Shi-Tomasi Detector'))
                    if sel_cor_det == 'Harris Detector':
                        fix_img_gray = np.float32(fix_img_gray)
                        fix_img_copy = fix_img.copy()
                        blockSize = st.slider('BlockSize',1,5,2,1)
                        cor_ksize = st.slider('Kernel Size',1,31,3,2)
                        cor_k = st.slider('k',0.01,1.0,0.04,0.01)
                        dst = cv2.cornerHarris(fix_img_gray,blockSize,cor_ksize,cor_k)
                        dst = cv2.dilate(dst,None)
                        fix_img_copy[dst>0.01*dst.max()] = [255,0,0]
                        with col2:
                            st.subheader('Corner Detected')
                            st.image(fix_img_copy,use_column_width=True)
                    elif sel_cor_det == 'Shi-Tomasi Detector':
                        fix_img_copy = fix_img.copy()
                        all_cor = st.checkbox('Detect all corners')
                        if all_cor:
                            det_cor_no = -1
                        else:
                            det_cor_no = st.slider('Number of corners to detect',1,1000,5,1)
                        quality_level = st.slider('Quality Level',0.001,0.99,0.01,0.001)
                        min_dist = st.slider('Minimum Distance',1,100,10,1)
                        corners = cv2.goodFeaturesToTrack(fix_img_gray,det_cor_no,quality_level,min_dist)
                        corners = np.int0(corners)
                        for i in corners:
                            x,y = i.ravel()
                            cv2.circle(fix_img_copy,(x,y),3,(255,0,0),-1)
                        with col2:
                            st.subheader('Corner Detected')
                            st.image(fix_img_copy,use_column_width=True)
            elif option == 'Edge Detection':
                edge_det = st.beta_expander('Edge Detection',expanded=True)
                with edge_det:
                    fix_img_copy = fix_img.copy()
                    blur = st.checkbox('Blur Image')
                    if blur:
                        blur_ksize = st.slider('Blur Kernel Size',1,11,5,1)
                        fix_img_copy = cv2.blur(fix_img_copy,ksize=(blur_ksize,blur_ksize))
                    threshold1 = st.slider('Canny Edge Detector Threshold1',1,255,127,1)
                    threshold2 = st.slider('Canny Edge Detector Threshold2',1,255,127,1)
                    edges = cv2.Canny(fix_img_copy,threshold1,threshold2)
                    with col2:
                            st.subheader('Edges Detected')
                            st.image(edges,use_column_width=True)
            elif option == 'Grid Detection':
                grid_det = st.beta_expander('Grid Detection',expanded = True)
                with grid_det:
                    fix_img_copy = fix_img.copy()
                    grid_type = st.selectbox('Grid Type',('ChessBoard Grid','Circles Grid'))
                    patSize = st.slider('Pattern Size',2,100,7,1)
                    if grid_type == 'ChessBoard Grid':
                        found,corners = cv2.findChessboardCorners(fix_img_copy,(patSize,patSize))
                    elif grid_type == 'Circles Grid':
                        try:   
                            found,corners = cv2.findCirclesGrid(fix_img_copy,(patSize,patSize),cv2.CALIB_CB_SYMMETRIC_GRID)
                        except:
                            found=False
                    if found == True:
                        cv2.drawChessboardCorners(fix_img_copy,(patSize,patSize),corners,found)
                        with col2:
                            st.subheader('Grid Detected')
                            st.image(fix_img_copy,use_column_width=True)
                    else:
                        st.write('Unable to find Grid')
            elif option == 'Contour Detection':
                con_det = st.beta_expander('Contour Detection',expanded=True)
                with con_det:
                    fix_img_copy = fix_img.copy()
                    fix_img_gray = cv2.cvtColor(fix_img_copy,cv2.COLOR_RGB2GRAY)
                    contours,hierarchy= cv2.findContours(fix_img_gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    sel_contour = st.selectbox('Contour Type',('External','Internal'))
                    base_contour = np.zeros(fix_img_gray.shape)
                    if sel_contour == 'External':
                        for i in range(len(contours)):
                            if hierarchy[0][i][3] == -1:
                                cv2.drawContours(base_contour,contours,i,255,-1)
                    elif sel_contour == 'Internal':
                        for i in range(len(contours)):
                            if hierarchy[0][i][3] != -1:
                                cv2.drawContours(base_contour,contours,i,255,-1)

                    base_contour[base_contour>1] = 1
                    base_contour[base_contour<0] = 0
                    with col2:
                            st.subheader('Contour Detected')
                            st.image(base_contour,use_column_width=True)         
            elif option == 'Feature Matching':
                fea_mat=st.beta_expander('Feature Detection',expanded=True)
                with fea_mat:
                    sub_img_file = st.file_uploader("Upload Template image to be matched", type=["png", "jpg", "jpeg"])
                    if sub_img_file is not None:
                        sub_img_bytes = np.asarray(bytearray(sub_img_file.read()), dtype=np.uint8)
                        opencv_subimage = cv2.imdecode(sub_img_bytes, 1)
                        sub_img_rgb = cv2.cvtColor(opencv_subimage,cv2.COLOR_BGR2RGB)
                        sub_img = cv2.cvtColor(sub_img_rgb,cv2.COLOR_RGB2GRAY)
                        with col1:
                            st.header('Image to be Matched')
                            st.image(sub_img_rgb)
                        target = fix_img.copy()
                        target = cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
                        sel_fea_mat = st.selectbox('Feature Detection Methods',('ORB','SIFT','FLANN'))
                        if sel_fea_mat == 'ORB':
                            no_match = st.slider('Number of Matches',1,1000,25,1)
                            flags = st.slider('Flags',2,10,2,2)
                            orb = cv2.ORB_create()
                            kp1,ds1 = orb.detectAndCompute(sub_img,None)
                            kp2,ds2 = orb.detectAndCompute(target,None)
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
                            matches = bf.match(ds1,ds2)
                            matches = sorted(matches,key=lambda x:x.distance)
                            sub_img_match = cv2.drawMatches(sub_img,kp1,target,kp2,matches[:no_match],None,flags=flags)
                            with col2:
                                st.subheader('Features Detected')
                                st.image(sub_img_match,use_column_width=True)
                        elif sel_fea_mat == 'SIFT':
                            flags = st.slider('Flags',0,10,2,2)
                            sift = cv2.xfeatures2d.SIFT_create()
                            #pip install opencv-contrib-python
                            kp1,ds1 = sift.detectAndCompute(sub_img,None)
                            kp2,ds2 = sift.detectAndCompute(target,None)
                            bf = cv2.BFMatcher()
                            matches = bf.knnMatch(ds1,ds2,k=2)
                            good = []
                            for match1,match2 in matches:
                                if match1.distance < 0.75*match2.distance:
                                    good.append([match1])
                            sift_matches = cv2.drawMatchesKnn(sub_img,kp1,target,kp2,good,None,flags=flags)
                            with col2:
                                st.subheader('Features Detected')
                                st.image(sift_matches,use_column_width=True)
                        elif sel_fea_mat == 'FLANN':
                            flags = st.slider('Flags',0,10,2,2)
                            sift = cv2.xfeatures2d.SIFT_create()
                            #pip install opencv-contrib-python
                            kp1,ds1 = sift.detectAndCompute(sub_img,None)
                            kp2,ds2 = sift.detectAndCompute(target,None)
                            FLANN_INDEX_KDTREE = st.slider('FLANN_INDEX_KDTREE',0,5,0,1)
                            trees = st.slider('Trees',1,10,5,1)
                            checks = st.slider('Checks',1,100,50,1)
                            index_param = dict(algorithm=FLANN_INDEX_KDTREE,trees=trees)
                            search_param = dict(checks=checks)
                            flann = cv2.FlannBasedMatcher(index_param,search_param)
                            matches = flann.knnMatch(ds1,ds2,k=2)
                            matchesMask = [[0,0] for i in range(len(matches))]
                            good = []
                            for i,(match1,match2) in enumerate(matches):
                                if match1.distance < 0.75*match2.distance:
                                    matchesMask[i]=[1,0]
                            draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),
                                                matchesMask=matchesMask,flags=flags)
                            flann_matches = cv2.drawMatchesKnn(sub_img,kp1,target,kp2,matches,None,**draw_params)
                            with col2:
                                st.subheader('Features Detected')
                                st.image(flann_matches,use_column_width=True)
            elif option == 'Watershed':
                watershed = st.beta_expander('Watershed Algorithm',expanded=True)
                with watershed:
                    manual_seed = st.button('Manual Seeding')
                    if not manual_seed:
                        fix_img_copy = fix_img.copy()
                        img_toshow = fix_img.copy()
                        blur_ksize = st.slider('Median Blur Kernel Size',1,101,35,2)
                        noise_ksize = st.slider('Noise removal kernel size',1,11,3,1)
                        fix_img_copy = cv2.medianBlur(fix_img_copy,blur_ksize)
                        gray = cv2.cvtColor(fix_img_copy,cv2.COLOR_RGB2GRAY)
                        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                        noise_kernel = np.ones((noise_ksize,noise_ksize),np.uint8)
                        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,noise_kernel,iterations=2)
                        sure_bg = cv2.dilate(opening,noise_kernel,iterations=3)
                        maskSize = st.slider('DistanceTransform MaskSize',0,3,3,3)
                        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,maskSize)
                        ret,sure_fg =cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
                        sure_fg = np.uint8(sure_fg)
                        unknown = cv2.subtract(sure_bg,sure_fg)
                        ret,markers = cv2.connectedComponents(sure_fg)
                        markers = markers+1
                        markers[unknown==255] = 0
                        markers = cv2.watershed(fix_img_copy,markers)
                        ws_contours,ws_hierarchy = cv2.findContours(markers,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                        for i in range(len(ws_contours)):
                            if ws_hierarchy[0][i][3] == -1:
                                cv2.drawContours(img_toshow,ws_contours,i,(255,0,0),10)
                        with col2:
                                st.subheader('Template Detected')
                                st.image(img_toshow,use_column_width=True)
                    else:
                        st.write('Manual Sedding is selected')

                        img_touse = opencv_image.copy()
                        img_toshow = opencv_image.copy()
                        marker_image = np.zeros(img_touse.shape[:2],dtype=np.int32)
                        segments = np.zeros(img_touse.shape,dtype=np.uint8)
                        def create_rgb(i):
                            return tuple(np.array(cm.tab10(i)[:3])*255)
                        colors = []
                        for i in range(10):
                            colors.append(create_rgb(i))
                        # global variable color choice and markers updated
                        n_markers = 10
                        current_marker =1
                        marks_updated = False
                        # callback function
                        def mouse_callback(event,x,y,flags,param):
                            nonlocal marks_updated
                            if event ==cv2.EVENT_LBUTTONDOWN:
                                cv2.circle(marker_image,(x,y),10,(current_marker),-1)
                                cv2.circle(img_touse,(x,y),10,colors[current_marker],-1)
                                marks_updated = True
                        cv2.namedWindow('Image')
                        cv2.setMouseCallback('Image',mouse_callback)

                        while True:
                            cv2.imshow('Watershed Segments',segments)
                            cv2.imshow('Image',img_touse)

                            k = cv2.waitKey(1)
                            if k==27:
                                break
                            elif k == ord('c'):
                                img_touse=opencv_image.copy()
                                marker_image = np.zeros(img_touse.shape[:2],dtype=np.int32)
                                segments = np.zeros(img_touse.shape,dtype=np.uint8)
                            elif k>0 and chr(k).isdigit():
                                current_marker = int(chr(k))
                            
                            if marks_updated:
                                marker_image_copy = marker_image.copy()
                                cv2.watershed(img_toshow,marker_image_copy)
                                segments = np.zeros(img_toshow.shape,dtype=np.uint8)
                                for color_ind in range(n_markers):
                                    segments[marker_image_copy==(color_ind)] = colors[color_ind]
                        with col2:
                            st.subheader('Features Detected')
                            st.image(segments,use_column_width=True)

                        cv2.destroyAllWindows()
            elif option == 'FaceDetection':
                gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                scaleFactor = st.slider('ScaleFactor',0.0,10.0,1.2,0.1)
                minNeigh = st.slider('Minimum Neighbors',0,20,5,1)
                face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                def detect_face(img,scaleFactor,minNeigh):
                    face_img = img.copy()
                    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=scaleFactor,minNeighbors=minNeigh)
                    for (x,y,w,h)  in face_rects:
                        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
                    return face_img
                detected_face = detect_face(gray,scaleFactor,minNeigh)
                with col2:
                        st.subheader('Detected Face')
                        st.image(detected_face,use_column_width=True)
            elif option == 'EyesDetection':
                gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                scaleFactor = st.slider('ScaleFactor',0.0,10.0,1.2,0.1)
                minNeigh = st.slider('Minimum Neighbors',0,20,5,1)
                eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
                def detect_eye(img,scaleFactor,minNeigh):
                    eye_img = img.copy()
                    eye_rects = eye_cascade.detectMultiScale(eye_img,scaleFactor=scaleFactor,minNeighbors=minNeigh)
                    for (x,y,w,h)  in eye_rects:
                        cv2.rectangle(eye_img,(x,y),(x+w,y+h),(255,255,255),10)
                    return eye_img
                detected_eye = detect_eye(gray,scaleFactor,minNeigh)
                with col2:
                        st.subheader('Detected Eyes')
                        st.image(detected_eye,use_column_width=True)
            elif option == 'Live Face Detection':
                st.write('Please make sure Camera is working and enabled, press Escape button to exit')
                scaleFactor = st.slider('ScaleFactor',0.0,10.0,1.2,0.1)
                minNeigh = st.slider('Minimum Neighbors',0,20,5,1)
                face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                def detect_face(img,scaleFactor,minNeigh):
                    face_img = img.copy()
                    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=scaleFactor,minNeighbors=minNeigh)
                    for (x,y,w,h)  in face_rects:
                        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
                    return face_img
                face_det = st.button('Run Face Detection')
                if face_det:
                    cap = cv2.VideoCapture(0)
                    while True:
                        ret,frame = cap.read(0)
                        frame = detect_face(frame,scaleFactor,minNeigh)
                        cv2.imshow('Video Face Detect',frame)
                        k = cv2.waitKey(1)
                        if k == 27:
                            break
                    cap.release()
                    cv2.destroyAllWindows()



#app()
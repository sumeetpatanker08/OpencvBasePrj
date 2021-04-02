import streamlit as st
import numpy as np
import cv2


def app():

    st.title("Image edit section")
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    add_img = st.checkbox('Add Image')
    if add_img :
        add_file = st.file_uploader("Add image", type=["png", "jpg", "jpeg"])
    add_mark = st.checkbox('Add Watermark')
    if add_mark:
        wmark_file = st.file_uploader("Upload Watermark image", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        fix_img = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
        col1,col2 = st.beta_columns([2,1])
        with col2:
            gray = st.checkbox('GrayImage')
            flip = st.checkbox('Flip Vertically')
            if flip:
                fix_img = cv2.flip(fix_img,1)

            st.write('Resize Image')
            
            aspect_ratio= st.slider("Image Resize",0.1,1.0,1.0,0.1)
            fix_img = cv2.resize(fix_img,(0,0),fix_img,aspect_ratio,aspect_ratio)
            
            h,w,c = fix_img.shape
            
            if add_img:
                if add_file is not None:
                    st.write('Added Image Settings')
                    addfile_bytes = np.asarray(bytearray(add_file.read()), dtype=np.uint8)
                    addopencv_image = cv2.imdecode(addfile_bytes, 1)
                    addfix_img = cv2.cvtColor(addopencv_image,cv2.COLOR_BGR2RGB)
                    
                    addaspect_ratio = st.slider('Height Width Ratio',0.01,1.0,1.0,0.1)
                    addfix_img = cv2.resize(addfix_img,(0,0),addfix_img,addaspect_ratio,addaspect_ratio)
                    y_add,x_add,c_add = addfix_img.shape
                    xpos= st.slider('X Position',1.0,(float(w)-x_add),1.0,10.0)
                    ypos = st.slider('Y Position',1.0,(float(h)-y_add),1.0,10.0)
                    try:
                        fix_img[int(ypos):int(ypos)+y_add,int(xpos):int(xpos)+x_add] = addfix_img
                    except:
                        st.write('Image size is more than base image Please decrease Size of image using AspectRatio')

                
            
            if add_mark:
                if wmark_file is not None:
                    st.write('WaterMark Image Settings')
                    wmfile_bytes = np.asarray(bytearray(wmark_file.read()), dtype=np.uint8)
                    wmopencv_image = cv2.imdecode(wmfile_bytes, 1)
                    wmfix_img = cv2.cvtColor(wmopencv_image,cv2.COLOR_BGR2RGB)
                    aspect_ratio = st.slider('Height Width Ratio',0.1,1.0,1.0,0.1)
                    wmfix_img = cv2.resize(wmfix_img,(0,0),wmfix_img,aspect_ratio,aspect_ratio)
                    y_img1,x_img1,c_img1 = fix_img.shape
                    y_img2,x_img2,c_img2 = wmfix_img.shape
                    xoff = x_img1 - x_img2
                    yoff = y_img1 - y_img2
                    roi = fix_img[yoff:y_img1,xoff:x_img1]
                    img2gray=cv2.cvtColor(wmfix_img,cv2.COLOR_RGB2GRAY)
                    mask_inv = cv2.bitwise_not(img2gray)
                    wbg = np.full(wmfix_img.shape,255,dtype=np.uint8)
                    bk = cv2.bitwise_or(wbg,wbg,mask=mask_inv)
                    final_roi = cv2.bitwise_or(roi,bk)
                    fix_img[yoff:y_img1,xoff:x_img1] = final_roi 
            
            if gray :
                fix_img = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
            else:
                if len(fix_img.shape)==2:
                    fix_img = cv2.cvtColor(fix_img,cv2.COLOR_GRAY2RGB)

            save = st.button('Save Image')
            if save:
                cv2.imwrite("modified_image.jpg",cv2.cvtColor(fix_img, cv2.COLOR_RGB2BGR))
                st.write('Image saved as modified_image.jpg')

        with col1:
            st.image(fix_img,use_column_width=True)

    else:
        st.write('No file selected')


#app()
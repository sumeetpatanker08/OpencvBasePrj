import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def app():
    
    st.title("Image Process section")
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
            Thresholding = st.beta_expander("Thresholding", expanded=False)
            with Thresholding:
                threshold = st.checkbox('Thresholding')
                
                if threshold:
                    img_gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                    adapt_thres = st.checkbox('Adaptive Thresholding')
                    thres_options = {'BinaryThreshold':'cv2.THRESH_BINARY','BinaryInverseThreshold':'cv2.THRESH_BINARY_INV',
                        'ThresholdTruncated':'cv2.THRESH_TRUNC','ThresholdToZero':'cv2.THRESH_TOZERO',
                        'ThresholdToZeroInverse':'cv2.THRESH_TOZERO_INV'}
                    if adapt_thres:
                        adapt_thres = {'ThresholdMean':'cv2.ADAPTIVE_THRESH_MEAN_C','ThresholdGaussian':'cv2.ADAPTIVE_THRESH_GAUSSIAN_C'}
                        thres_options = {'BinaryThreshold':'cv2.THRESH_BINARY','BinaryInverseThreshold':'cv2.THRESH_BINARY_INV'}
                        adapt_thres_type = st.selectbox('AdaptiveThresholdMethods',tuple(adapt_thres.keys()))
                        thres_type = st.selectbox('Threshold options',tuple(thres_options.keys()))
                        thres = cv2.adaptiveThreshold(img_gray,255,eval(adapt_thres[adapt_thres_type]),eval(thres_options[thres_type]),11,8)
                        nor_thres = st.checkbox('Blend Normal Thresholding')
                        if nor_thres:
                            thres_options_norm = {'BinaryThreshold':'cv2.THRESH_BINARY','BinaryInverseThreshold':'cv2.THRESH_BINARY_INV',
                                                'ThresholdTruncated':'cv2.THRESH_TRUNC','ThresholdToZero':'cv2.THRESH_TOZERO',
                                                'ThresholdToZeroInverse':'cv2.THRESH_TOZERO_INV'}
                            thres_type_norm = st.selectbox('Threshold Options to Blend',tuple(thres_options_norm.keys()))
                            thresh_val_norm = st.slider('Threshold values ',1.0,255.0,127.0,1.0)
                            weight = st.slider('Thresholding Weight',0.0,1.0,0.5,0.1)
                            rel,thres1 = cv2.threshold(img_gray,thresh_val_norm,255.0,eval(thres_options_norm[thres_type_norm]))
                            blended = cv2.addWeighted(src1=thres,alpha=1-weight,src2=thres1,beta=weight,gamma=0)
                            thres=blended
                            
                        fix_img = thres
                    else:
                        thres_type = st.selectbox('Threshold options',tuple(thres_options.keys()))
                        thresh_val = st.slider('Threshold value',1.0,255.0,127.0,1.0)
                        rel,thres = cv2.threshold(img_gray,thresh_val,255.0,eval(thres_options[thres_type])) 
                        fix_img = thres

            Blurring = st.beta_expander('Blurring',expanded=False)
            with Blurring:
                Thresholding.expanded=False
                st.write(Thresholding.expanded)
                gamma = st.checkbox('Gamma Correction')
                if gamma:
                    gamma = st.slider('gamma value',0.1,10.0,1.0,0.1)
                    fix_img = np.power(fix_img,gamma)
                    fix_img[fix_img>1] = 1
                    fix_img[fix_img<0] = 0
                blur = st.checkbox('Blurring')
                if blur:
                    options=st.selectbox('Blurring Options',('Filter2D','BlurDefault','GaussianBlur','MedianBlur','BilateralFilter'))

                    if options=='Filter2D':
                        kersize = st.slider('KernelSize',2,10,5,1)
                        kervalue = st.slider('KernelValue',0.01,5.0,0.5,0.01)
                        kernel = np.ones(shape=(5,5),dtype=np.uint8)
                        fix_img=cv2.filter2D(fix_img,-1,kernel)
                    elif options=='BlurDefault':
                        kersize = st.slider('KernelSize',2,10,5,1)
                        fix_img = cv2.blur(fix_img,ksize=(kersize,kersize))
                    elif options=='Gaussian':
                        kersize = st.slider('KernelSize',2,10,5,1)
                        sigma = st.slider('Sigma',1,20,10,1)
                        fix_img = cv2.GaussianBlur(fix_img,(kersize,kersize),sigma)
                    elif options == 'MedianBlur':
                        kersize = st.slider('KernelSize',1,11,5,2)
                        fix_img = cv2.medianBlur(fix_img,kersize)
                    elif options =='BilateralFilter':
                        d = st.slider('D value',1,20,9,1)
                        sigmaColor = st.slider('SigmaColor',1,100,75,1)
                        sigmaSpace = st.slider('SigmaSpace',1,100,75,1)
                        fix_img = cv2.bilateralFilter(fix_img,d,sigmaColor,sigmaSpace)

            Morphology = st.beta_expander('Morphology',expanded=False)
            with Morphology:
                morph = st.checkbox('Morphology')
                if morph:
                    fix_img2gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                    h,w = fix_img2gray.shape
                    morph_options = st.selectbox('Morphological Operations',('Erosion','Dilation','Opening','Closing','Gradient'))
                    morph_kersize = st.slider('Morph Kernel Size',2,10,5,1)
                    morph_kervalue = st.slider('Morph Kernel Value',0.01,10.0,1.0,0.01)
                    morph_kernel = np.ones((morph_kersize,morph_kersize),dtype=np.float32)*morph_kervalue
                    if morph_options == 'Erosion':
                        iterations = st.slider('Iteration',1,10,4,1)
                        fix_img2gray=cv2.erode(fix_img2gray,morph_kernel,iterations=iterations)
                        fix_img=fix_img2gray
                    elif morph_options == 'Dilation':
                        iterations = st.slider('Iteration',1,10,4,1)
                        fix_img2gray=cv2.dilate(fix_img2gray,morph_kernel,iterations=iterations)
                        fix_img=fix_img2gray
                    elif morph_options == 'Opening':
                        wnoise = st.checkbox('Add Background Noise')
                        if wnoise:
                            wnoise_img = np.random.randint(low=0,high=2,size=(h,w))
                            wnoise_img = wnoise_img*255
                            fix_img2gray=fix_img2gray+wnoise_img
                        fix_img2gray = cv2.morphologyEx(fix_img2gray,cv2.MORPH_OPEN,morph_kernel)
                        fix_img=fix_img2gray
                    elif morph_options == 'Closing':
                        bnoise = st.checkbox('Add Foreground Noise')
                        if bnoise:
                            bnoise_img = np.random.randint(low=0,high=2,size=(h,w))
                            bnoise_img = bnoise_img*-255
                            fix_img2gray=fix_img2gray+bnoise_img
                            fix_img2gray[fix_img2gray==-255] = 0
                        fix_img2gray = cv2.morphologyEx(fix_img2gray,cv2.MORPH_CLOSE,morph_kernel)
                        fix_img=fix_img2gray
                    elif morph_options=='Gradient':
                        fix_img2gray = cv2.morphologyEx(fix_img2gray,cv2.MORPH_GRADIENT,morph_kernel)
                        fix_img=fix_img2gray

            Gradient = st.beta_expander('Gradient',expanded=False)
            with Gradient:
                grad = st.checkbox('Gradient')
                if grad:
                    fix_img2gray = cv2.cvtColor(fix_img,cv2.COLOR_RGB2GRAY)
                    grad_options = st.selectbox('Gradient Options',('SobelX','SobelY','SobelXY','Laplacian'))
                    grad_ker_size = st.slider('Sobel Kernel Size',1,11,5,2)
                    if grad_options=='SobelX':
                        fix_img2gray=cv2.Sobel(fix_img2gray,cv2.CV_64F,1,0,ksize=grad_ker_size)
                        fix_img2gray[fix_img2gray>1] = 1
                        fix_img2gray[fix_img2gray<0] = 0
                        fix_img = fix_img2gray
                    elif grad_options == 'SobelY':
                        fix_img2gray=cv2.Sobel(fix_img2gray,cv2.CV_64F,0,1,ksize=grad_ker_size)
                        fix_img2gray[fix_img2gray>1] = 1
                        fix_img2gray[fix_img2gray<0] = 0
                        fix_img = fix_img2gray
                    elif grad_options == 'SobelXY':
                        sobelx=cv2.Sobel(fix_img2gray,cv2.CV_64F,1,0,ksize=grad_ker_size)
                        sobely=cv2.Sobel(fix_img2gray,cv2.CV_64F,0,1,ksize=grad_ker_size)
                        blended = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=0)
                        blended[blended>1] = 1
                        blended[blended<0] = 0
                        fix_img = blended
                    elif grad_options == 'Laplacian':
                        fix_img2gray=cv2.Laplacian(fix_img2gray,cv2.CV_64F,ksize=grad_ker_size)
                        fix_img2gray[fix_img2gray>1] = 1
                        fix_img2gray[fix_img2gray<0] = 0
                        fix_img = fix_img2gray

            histogram = st.beta_expander('Color Historgram',expanded=False)
            with histogram:
                hist = st.checkbox('Show Color Histogram')

                if hist:
                    masking = st.checkbox('Masking')
                    equalizer = st.checkbox('Equalize Color')
                    color = ('b','g','r')
                    fig, ax = plt.subplots()
                    y_max_val =[]
                    for i,col in enumerate(color):
                        histr = cv2.calcHist([opencv_image],[i],None,[256],[0,256])
                        y_max_val.append(np.max(histr))
                    ymax = np.max(y_max_val) + 2
                    xmin,xmax = st.slider('Original Image xLimit',0.0,256.0,(0.0,256.0),1.0)
                    ymin,ymax = st.slider('Original Image yLimit',0.0,float(ymax),(0.0,float(ymax)),1.0)
                    for i,col in enumerate(color):
                        histr = cv2.calcHist([opencv_image],[i],None,[256],[0,256])
                        y_max_val.append(np.max(histr))
                        plt.plot(histr,color=col)
                        plt.xlim([xmin,xmax])
                        plt.ylim([ymin,ymax])
                    fig_mod=fig

                
                    if masking:
                        h_mask,w_mask,c_mask = opencv_image.shape
                        mask = np.zeros((h_mask,w_mask),np.uint8)
                        h_mask_min,h_mask_max = st.slider('Height_mask',0,h_mask,(0,h_mask),1)
                        w_mask_min,w_mask_max = st.slider('Width_mask',0,w_mask,(0,w_mask),1)
                        mask[h_mask_min:h_mask_max,w_mask_min:w_mask_max] = 255
                        masked_img = cv2.bitwise_and(opencv_image,opencv_image,mask=mask)
                        fix_img = cv2.bitwise_and(fix_img,fix_img,mask=mask)

                        if not equalizer:
                            color = ('b','g','r')
                            fig_mask, ax_mask = plt.subplots()
                            y_max_val =[]
                            for i,col in enumerate(color):
                                histr = cv2.calcHist([opencv_image],[i],mask=mask,histSize=[256],ranges=[0,256])
                                y_max_val.append(np.max(histr))
                            ymax = np.max(y_max_val) + 2
                            xmin_mask,xmax_mask = st.slider('Modified Image xLimit',0.0,256.0,(0.0,256.0),1.0)
                            ymin_mask,ymax_mask = st.slider('Modified Image yLimit',0.0,float(ymax),(0.0,float(ymax)),1.0)
                            for i,col in enumerate(color):
                                histr = cv2.calcHist([opencv_image],[i],mask=mask,histSize=[256],ranges=[0,256])
                                y_max_val.append(np.max(histr))
                                plt.plot(histr,color=col)
                                plt.xlim([xmin_mask,xmax_mask])
                                plt.ylim([ymin_mask,ymax_mask])
                            fig_mod=fig_mask
                        opencv_image = masked_img

                
                    if equalizer:
                        hsv = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2HSV)
                        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
                        opencv_image_eq = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                        fix_img = cv2.cvtColor(opencv_image_eq,cv2.COLOR_BGR2RGB)
                        color = ('b','g','r')
                        fig_eq, ax_eq = plt.subplots()
                        y_max_val =[]
                        for i,col in enumerate(color):
                            histr = cv2.calcHist([opencv_image_eq],[i],None,[256],[0,256])
                            y_max_val.append(np.max(histr))
                        ymax = np.max(y_max_val) + 2
                        xmin_eq,xmax_eq = st.slider('Modified Image xLimit',0.0,256.0,(0.0,256.0),1.0)
                        ymin_eq,ymax_eq = st.slider('Modified Image yLimit',0.0,float(ymax),(0.0,float(ymax)),1.0)
                        for i,col in enumerate(color):
                            histr = cv2.calcHist([opencv_image_eq],[i],None,[256],[0,256])
                            y_max_val.append(np.max(histr))
                            plt.plot(histr,color=col)
                            plt.xlim([xmin_eq,xmax_eq])
                            plt.ylim([ymin_eq,ymax_eq])
                        fig_mod = fig_eq                

        with col1:
            if hist:
                st.subheader('Color Historgram')
                st.pyplot(fig)
                    
        with col2:
            st.subheader('Modified Image')
            st.image(fix_img,use_column_width=True)
            if hist:
                st.subheader('Modified Image Historgram')
                st.pyplot(fig_mod)

#app()
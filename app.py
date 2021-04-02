import streamlit as st
from pandas.core import base
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import plotly.figure_factory as ff
from multiapp import MultiApp
from apps import img_edit,img_process,Object_Detection,Object_tracking,project # import your app modules here

st.set_page_config(layout="wide")
app = MultiApp()

# Add all your application here
app.add_app("Image Editing", img_edit.app)
app.add_app("Image Processing", img_process.app)
app.add_app("Object Detection",Object_Detection.app)
app.add_app('Object Tracking',Object_tracking.app)
app.add_app('Project',project.app)

# The main app
app.run()
import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import Matern

# Back-end

## App formatting
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")

do_stuff_on_page_load()

st.markdown(f'''
<style>
.appview-container .main .block-container{{
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;    }}
</style>
''', unsafe_allow_html=True,
)

## Figure making
def make_figure(V,c,logG):

    VINT = round(49*(V-0.012)/(0.6-0.012))
    cINT = round(49*(c-15)/4)
    logGINT = round(49*(logG-5)/2)

    try:
        st.image(f'Figures/final_posterior_V{VINT}_c{cINT}_logG{logGINT}.png')
    except:
        st.image('Figures/final_posterior_V24_c24_logG24.png')

## Calculate probability
### We need sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
### Load all other data for GP Prediction
prior = np.load('GPData/prior.npy')
X = np.load('GPData/X.npy')
alpha = np.load('GPData/alpha.npy')
X_train_gp = np.load('GPData/X_train_gp.npy')

kernel = Matern(nu=.5,length_scale=[0.18,0.4,0.3],length_scale_bounds='fixed')
pred = (sigmoid((kernel(X,X_train_gp)@alpha) + prior)).reshape(50,50,50)

def probCalc(V,c,logG):
    VINT = round(49*(V-0.012)/(0.6-0.012))
    cINT = round(49*(c-15)/4)
    logGINT = round(49*(logG-5)/2)
    return(round(pred[cINT,VINT,logGINT],5))



# Front-end

st.title('Quantitative PF Modeling of the Fe-Cr Alloy System')
st.markdown('***')
st.markdown("This app predicts the microstucture of your alloy during soludification given solidification velocity (V), thermal gradient (G) and composition (c).")
st.markdown("TRIAL VERSION: Use dynamic figure by moving log(G) slider with V = [0.012,0.306,0.6] and c = 17. Probability calculation works in all space.")

col1, col2 = st.columns(2, gap='large')
with col1:
    st.markdown('### Select the value for your variables.')

    V = st.slider("Select value for V [m/s]:",0.012, 0.6, 0.306,0.001,"%0.3f")
    c = st.slider("Select value for c [wt. %Cr]:",15.0, 19.0, 17.0 ,0.01)
    logG = st.slider(r"Select value for $\log_{10}$(G [K/m]):",5.0, 7.0, 6.0 ,0.01)

    st.markdown(f'The probability that the microstructure of your alloy during soludification is planar is: {probCalc(V,c,logG)}.')

with col2:
    make_figure(V,c,logG)
import streamlit as st
import numpy as np
import time
import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import plotly.express as px
from utils import adamax, objective, derivative, gradient_descent,adam, rmsprop
st.set_page_config(page_title=None, page_icon=None, layout="wide")
# define range for input
bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
xaxis = np.arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = np.arange(bounds[1,0], bounds[1,1], 0.1)
x, y = np.meshgrid(xaxis, yaxis)
    # create a mesh from the axis
results = objective(x, y)
    

# Main layout begins here

st.title("""
Three key findings from the 2022 UN Population Prospects
""")

st.write("""
The UN releases an update of its World Population Prospects every two years. 
Its latest release was delayed due to the COVID-19 pandemic and was released in 2022.
""")
placeholder2 = st.empty()
coll1, coll2, coll3 = st.columns(3,gap="medium")
with coll1:
    st.markdown('#### Objective Function')
    # define range for input
    r_min, r_max = -1.0, 1.0
    # sample input range uniformly at 0.1 increments
    xaxis1 = np.arange(r_min, r_max, 0.1)
    yaxis1 = np.arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = np.meshgrid(xaxis1, yaxis1)

    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = plt.figure(figsize=(6,6))
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    st.pyplot(figure)
st.markdown("---")

        
#############################################################
np.random.seed(1)

# define the total iterations
n_iter = 60

placeholder1 = st.empty()
with placeholder1.container():
    col1, col2, col3, col4 = st.columns(4,gap="medium")
    with col1:
        st.write ('Adam ')
        # steps size
        values = st.slider(
            'Select a range of values',
            0.0, 0.1, 0.02)
        st.write('steps size:', values)
        alpha = values
        # factor for average gradient
        values = st.slider(
            'Select a range of values',
            0.0, 1.0, 0.1)
        st.write('average gradient:', values)
        beta1 = values
        # factor for average squared gradient
        values = st.slider(
            'Select a range of values',
            0.0, 1.0, 0.001)
        st.write('average squared gradient:', values)
        beta2 = values

        # perform the gradient descent search with adam
        solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
        solutions = np.asarray(solutions)
        x_adam=solutions[:, 0]
        y_adam=solutions[:, 1]
    with col2:
        st.write ('SGD momentum ')
        # steps size
        values1 = st.slider('Select a range of values',0.0, 1.5, 0.1)
        st.write('steps size:', values1)
        step_size = values1
        # factor for momentun
        values1 = st.slider('Select a range of values',0.1, 1.2, 0.1)
        st.write('momentum:', values1)
        momentum = values1
        solutions = gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)
        solutions = np.asarray(solutions)
        x_sgd=solutions[:, 0]
        y_sgd=solutions[:, 1]

        # perform the gradient descent search with adam
        
    with col3:
        st.write ('Rmsprop ')
        # steps size
        values = st.slider('Select a range of values',0.0, 0.01, 0.001)
        st.write('steps size:', values)
        step_size = values
        # factor for average gradient
        values = st.slider('Select a range of values',0.0, 1.99, 0.01)
        st.write('rho:', values)
        rho = values
        
        # perform the gradient descent search with adam
        solutions = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
        solutions = np.asarray(solutions)
        x_rmsprop=solutions[:, 0]
        y_rmsprop=solutions[:, 1]
    with col4:
        st.write ('Adamax ')
        # steps size
        values = st.slider('Select a range of values',0.0, 0.8, 0.01)
        st.write('steps size:', values)
        alpha = values
        # factor for average gradient
        values = st.slider('Select a range of values',0.0, 1.2, 0.2)
        st.write('average gradient:', values)
        beta1 = values
        # factor for average squared gradient
        values = st.slider('Select a range of values', 0.01, 1.99, 0.01)
        st.write('average squared gradient:', values)
        beta2 = values

        # perform the gradient descent search with adam
        solutions = adamax(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
        solutions = np.asarray(solutions)
        x_adamax=solutions[:, 0]
        y_adamax=solutions[:, 1]
        
################################################################################################


progress_bar = st.progress(0)
placeholder = st.empty()
#chart = st.line_chart(df['x'].iloc[0])
xaxis = np.arange(bounds[0,0], bounds[0,1], 0.1)



for i in range(60):
    
    with placeholder.container():
        fig_col1, fig_col2  = st.columns(2)
        with fig_col1:
            st.markdown("#### Adam")
            fig, ax = plt.subplots()
            ax.contourf(x, y, results, levels=50, cmap='jet')
            ax.plot(x_adam[i], y_adam[i],'.-', color='w', markersize=12)
            fig1=st.pyplot(fig)
            
            st.markdown("#### SGD")
            fig, ax = plt.subplots()
            ax.contourf(x, y, results, levels=50, cmap='jet')
            ax.plot(x_sgd[i], y_sgd[i],'.-', color='w', markersize=12)
            fig1=st.pyplot(fig)
        with fig_col2:
            st.markdown("#### Rmsprop")
            fig, ax = plt.subplots()
            ax.contourf(x, y, results, levels=50, cmap='jet')
            ax.plot(x_rmsprop[i], y_rmsprop[i],'.-', color='w', markersize=12)
            fig1=st.pyplot(fig)
            
            st.markdown("#### Adamax")
            fig, ax = plt.subplots()
            ax.contourf(x, y, results, levels=50, cmap='jet')
            ax.plot(x_adamax[i], y_adamax[i],'.-', color='w', markersize=12)
            fig1=st.pyplot(fig)
        
            
    
st.balloons()


 

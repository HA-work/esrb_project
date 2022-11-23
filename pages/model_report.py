import streamlit as st
import pandas as pd
import pycaret
from pycaret.classification import *



# tried saving this as a method to try and imporve speed
@st.cache
def run_multiple_models():



		

	df = pd.read_csv('data/cleaned_games.csv')
	df = df.drop(columns=['title',  'esrb_encoded', 'no_descriptors'])


	# dropping encoded as it will give away the rating


	#df.head()

	setup(data = df, target = 'esrb_rating', session_id=69, silent=True) 

	with st.spinner('Running multiple machine learning models. Please wait....'):
		best_model = compare_models()

	return best_model

	



st.set_page_config(
    page_title="Model Report",
   
)


# is there a way to cache this?


st.header('This page shows how a bunch of different models performed on the data.')

best_model = run_multiple_models()

st.header('Completed running all models!')
st.write('See results below.')
st.balloons()
report = pull()

st.dataframe(
	report.style.highlight_max(axis=0, color='yellow'),
	use_container_width=True
	)
		


best_choice = str(best_model)

st.markdown("The best model is:\n\n"+
	best_choice
	)




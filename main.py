import streamlit as st
from utils import utils


st.title("Fitness AI")


def main():
    #Get the question from the user
    user_input=st.text_input("Type your fitness goal",placeholder="Ex. I want to lose belly fat. Suggest me some exercises with the plan of 3 months.")
    submit=st.button("Submit")

    if submit:
        #get the tags from the user_input
        user_tags=utils.extract_tags(user_input)

        #run the Crew 
        output=utils.execute_crew(user_tags)
        st.markdown(output)


# Run the app
if __name__ == "__main__":
    main()
import streamlit as st
from agents import retriever_agent, filter_agent, summarizer_agent, rating_agent, should_use_summarizer, should_skip_to_rating_agent, get_model_tokenizer, retriever, app
from typing import Literal

# Define Streamlit app interface
def run_app():
    st.title("Review Rating Prediction App")

    # User input
    query = st.text_area("Enter a review or query:")

    # Model selector
    model_type = st.selectbox("Select model type:", options=["base", "finetuned"])

    # Summarizer toggle
    use_summarizer = st.checkbox("Use summarizer", value=True)

    # Rating only toggle (bypass retriever, filter, summarizer)
    use_rating_only = st.checkbox("Use only RatingAgent (skip all others)", value=False)

    if st.button("Analyze"):
        if query.strip():
            # Build initial state
            state = {
                "query": query,
                "retriever": retriever,  # required for RetrieverAgent internally
                "retrieved_reviews": [],
                "most_relevant_review": "",
                "review_summary": "",
                "rating_prediction": "",
                "use_summarizer": use_summarizer,
                "model_type": model_type,
                "use_rating_only": use_rating_only,
            }

            # Run through LangGraph workflow
            result_state = app.invoke(state)

            # Show Results
            st.subheader("Most Relevant Review")
            st.write(result_state.get("most_relevant_review", "N/A"))

            st.subheader("Review Summary")
            st.write(result_state.get("review_summary", "N/A"))

            st.subheader("Predicted Rating")
            st.write(result_state.get("rating_prediction", "N/A"))

        else:
            st.warning("Please enter a query before running the analysis.")

# Run the app
if __name__ == "__main__":
    run_app()
()

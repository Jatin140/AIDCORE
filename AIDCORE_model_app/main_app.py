from clearml import PipelineDecorator, PipelineController
import streamlit as st

# Streamlit app will be launched from here
# @PipelineDecorator.component(return_values=["None"],cache=False)
def launch_app(*args, **kwargs):
    """
    TBD
    """
    # logger = PipelineController.get_logger()
    # logger.report_text("Launching streamplit app...")    
    # logger.report_text("Please click here --> http://localhost:8501")    

    exit_app = True
    
    # Main Streamlit app loop
    while exit_app:
        st.title("AIDCORE DEMO...")
        st.write("This app is used for giving a product review and sending email to product owner in case negative review for a product is provided by customer.")
        
        # Add an exit button
        if st.button("Exit",key='exit_button'):
            exit_app = False
        
    st.write("App has been exited.")
    overall_review_sentiment = "POSITIVE" # or NEGATIVE

    return (0,overall_review_sentiment)


if __name__ == "__main__":
    launch_app()
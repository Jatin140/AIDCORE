import smtplib, ssl
from clearml import PipelineDecorator, PipelineController

sender_email_id = ""
receiver_email_id = ""
sender_email_id_password = ""

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = sender_email_id
password = sender_email_id_password

# Create a secure SSL context
context = ssl.create_default_context()

@PipelineDecorator.component(return_values=["None"],cache=False)
def send_email_to_product_owner(text,review_type="POSITIVE"):
    """
    TBD
    """
    logger = PipelineController.get_logger()

    # Try to log in to server and send email
    if review_type == "NEGATIVE":
        logger.report_text("Sending email to product owner if review is negative...")   
        try:
            server = smtplib.SMTP(smtp_server,port)
            server.ehlo() # Can be omitted
            server.starttls(context=context) # Secure the connection
            server.ehlo() # Can be omitted
            server.login(sender_email, password)
            # TODO: Send email here
        except Exception as e:
            # Print any error messages to stdout
            print(e)
        finally:
            server.quit()
    else:
        logger.report_text("Review is positive, enjoy :-) ...") 

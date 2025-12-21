# user/notifications/email_service.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config

def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = Config.EMAIL_FROM
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
        server.starttls()
        server.login(Config.SMTP_USER, Config.SMTP_PASS)
        server.send_message(msg)

import email
import os
import imaplib
import email.header
import time
import re

from constants import EMAIL
# Connect to the email server


# Connect to the Outlook mail server

# mail.login(username, password)

# Select the mailbox to search in (e.g., Inbox)
# mail.select('inbox')

# # Search for unread emails from a specific email address

#mail.search(None, search_query)

# Loop continuously to check for new emails
def check_for_new_events():
    # Check for new emails
    username = EMAIL
    password = os.environ['EMAIL_PASSWORD']
    mail = imaplib.IMAP4_SSL('imap-mail.outlook.com')
    mail.login(username, password)
    mail.select('inbox')
    search_query = 'FROM "slurm@desy.de" UNSEEN'
    status, response = mail.search(None, search_query)
    new_email_ids = response[0].split()
    alerts = []
    # Process new emails
    for email_id in new_email_ids:
        status, response = mail.fetch(email_id, '(RFC822)')
        email_data = response[0][1]
        mail_message = email.message_from_bytes(email_data)
        # Get the email subject by decoding the bytes
        header = str(email.header.make_header(
            email.header.decode_header(mail_message['Subject'])))
        header = header.replace('\n', '')
        # Define the regular expression pattern to extract the Name and alert status
        pattern = r'Job_id=(\d+)\s+Name=([\w_]+).*Run time.*\b(COMPLETED|FAILED|CANCELLED)\b'
        # Extract the Name and alert status from the input string
        match = re.search(pattern, header)
        job_id = match.group(1)
        name = match.group(2)
        status = match.group(3)
        # Print the Name and alert status
        # print(f'Job_id: {job_id}')
        # print(f'Name: {name}')
        # print(f'Status: {status}')
        alerts.append({'job_id': job_id, 'name': name, 'status': status})
        # Mark the email as seen
        mail.store(email_id, '+FLAGS', '\\Seen')
    mail.close()
    mail.logout()
    return alerts

    # # Wait for a few seconds before checking for new emails again
    # time.sleep(20)

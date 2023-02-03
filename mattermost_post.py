import requests
import os
webhook_token = os.environ.get('WEBHOOK_TOKEN')
MATTERMOST_WEBHOOK_URL = f"https://mattermost.web.cern.ch/hooks/{webhook_tokenecho }"

payload = {
    "username": "SyReAL",
    "text": "Hello, this is a direct message from a bot!",
    "channel": "@frederik.bornemann-studium.uni-hamburg.de",
}

response = requests.post(MATTERMOST_WEBHOOK_URL, json=payload)

if response.status_code != 200:
    print("Failed to send direct message. Response code:", response.status_code)
else:
    print("Direct message sent successfully.")
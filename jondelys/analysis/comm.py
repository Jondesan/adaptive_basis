import requests
token = "7094097669:AAGOaEL4UI1ZH0KSG9YbcyX_RVto_sBWARs"
url = f'https://api.telegram.org/bot{token}'
chat_id = 1223997185

def send_notification(msg):
    params = {'chat_id': chat_id, 'text': msg}
    r = requests.get(url + "/sendMessage", params=params)

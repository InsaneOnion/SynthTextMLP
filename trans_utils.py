import hashlib
import random
import requests

# 百度翻译 API 配置信息
APP_ID = '20241204002218674'
API_KEY = 'mCq9FmT6wcqbzMzK4A_U'
URL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

def baidu_translate(query, from_lang='en', to_lang='zh'):
    salt = str(random.randint(32768, 65536))

    sign = APP_ID + query + salt + API_KEY
    sign = hashlib.md5(sign.encode('utf-8')).hexdigest()

    params = {
        'q': query,
        'from': from_lang,
        'to': to_lang,
        'appid': APP_ID,
        'salt': salt,
        'sign': sign
    }

    response = requests.get(URL, params=params)
    result = response.json()

    if 'trans_result' in result:
        return result['trans_result'][0]['dst']
    else:
        return 'Error: ' + result.get('error_msg', 'Unknown error')

if "__main__" == __name__:
    text = r"   I   \n want to go  \n to beijing"
    translated_text = baidu_translate(text)
    print("译文:", translated_text)

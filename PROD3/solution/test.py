

import requests
if __name__ == '__main__':
    r = requests.post('http://localhost:5000/predict', json={'num': 5})


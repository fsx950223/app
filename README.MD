## Install
1.Install RabbitMQ
2.Install dependencies
```pip install -r requirements.txt```
## How to run it?
1. start server
```python chatbot.py```
2. start flask client
```flask --app app run```
3. test with curl
```curl http://127.0.0.1:5000?message=hello```
## Install
1.Install RabbitMQ
```docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management```
2.Install MongoDB
```docker run --name mongodb -p 27017:27017 -d mongodb/mongodb-community-server:latest```
3.Install dependencies
```pip install -r requirements.txt```
## How to run it?
1. start server
```python chatbot.py```
2. start flask client
```flask --app app run```
3. test with curl
```curl http://127.0.0.1:5000?message=hello```  
## Tests
```pytest chatbot_test.py```
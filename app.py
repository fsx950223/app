from flask import Flask, request, jsonify
import pika
import uuid


app = Flask(__name__)

class ChatbotRpcClient:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body.decode('utf-8')

    def call(self, prompt):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=prompt)
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        return self.response

rpc_client = ChatbotRpcClient()

@app.route('/', methods=['GET'])
def generate():
    prompt = request.args.get('message', '')
    app.logger.info(f'Get prompt: {prompt}')
    response = rpc_client.call(prompt)
    app.logger.info(f'Get Response: {response}')
    return jsonify(prompt=prompt, response=response)
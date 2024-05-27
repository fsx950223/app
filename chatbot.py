from transformers import OPTForCausalLM, GPT2Tokenizer
import pika
import torch
from pymongo import MongoClient
from datetime import datetime

model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
uri = "localhost:27017"
client = MongoClient(uri)
database = client.get_database("chatbot")
records = database.get_collection("records")

def do_record(prompt, response, start_time, end_time):
    query = { "prompt": prompt, "response": response, "start time": start_time, "end time": end_time }
    records.insert_one(query)

@torch.inference_mode()
def get_response(prompt):
    start_time = datetime.now().ctime()
    inputs = tokenizer([prompt], return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=None, do_sample=False)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = datetime.now().ctime()
    do_record(prompt, response, start_time, end_time)
    return response


def on_request(ch, method, props, body):
    prompt = body.decode('utf-8')
    response = get_response(prompt)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)

if __name__ == '__main__':
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))

    channel = connection.channel()

    channel.queue_declare(queue='rpc_queue')


    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

    print("Awaiting RPC requests")
    channel.start_consuming()

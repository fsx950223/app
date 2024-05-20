# from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import pika

# model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
# tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

model = LlamaForCausalLM.from_pretrained("/raid/sixifang/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("/raid/sixifang/Llama-2-7b-hf")

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='rpc_queue')


def on_request(ch, method, props, body):
    prompt = body
    inputs = tokenizer([prompt], return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=None, do_sample=False)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print("Awaiting RPC requests")
channel.start_consuming()

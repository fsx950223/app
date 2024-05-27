import pytest
from chatbot import get_response, do_record, records
import torch
from datetime import datetime

def test_model():
    torch.manual_seed(0)
    prompt="hello"
    response = get_response(prompt)
    assert response == "hello, I'm a new player and I'm looking for a good team to play with."

def test_record():
    prompt="hello"
    response = get_response(prompt)
    start_time = datetime.now().ctime()
    end_time = datetime.now().ctime()
    do_record(prompt, response, start_time, end_time)
    record = records.find_one({ "prompt": prompt, "response": response, "start time": start_time, "end time": end_time })
    assert record is not None

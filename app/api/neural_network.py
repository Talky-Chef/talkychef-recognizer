from fastapi import FastAPI
from pydantic import BaseModel

from app import command_recognizer


app = FastAPI()
recognizer = command_recognizer.CommandRecognizer()


class Text(BaseModel):
    value: str


@app.post("/recognize")
def recognize(text: Text) -> int:
    return recognizer.recognize_command(text.value)

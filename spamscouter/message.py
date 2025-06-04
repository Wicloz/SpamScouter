import email
from bs4 import UnicodeDammit


def _part_to_text(part):
    encoding = part.get_content_charset()
    encodings = [encoding] if encoding else []

    text = UnicodeDammit(part.get_payload(decode=True), encodings, is_html=True).unicode_markup

    if not text:
        return ''
    return text


def _message_to_text(message):
    text = ''
    for part in message.walk():
        mime_type = part.get_content_type()
        if mime_type == 'text/plain' or mime_type == 'text/html':
            text += _part_to_text(part)
    return text


def parse_milter_message(message_bytes):
    return _message_to_text(email.message_from_bytes(message_bytes))


def message_builder(message_bytes, read, unique_identifier, folder_name, folder_flags, config):
    message = email.message_from_bytes(message_bytes)
    message.read = read

    label = config.get_spam_status(message, folder_name, folder_flags)
    text = _message_to_text(message)

    return Message(text, unique_identifier, label)


class Message:
    def __init__(self, text, uid, label):
        self.text = text
        self.uid = uid
        self.label = label

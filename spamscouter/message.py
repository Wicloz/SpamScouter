import email
from bs4 import UnicodeDammit, BeautifulSoup
from subprocess import run
from tempfile import TemporaryDirectory
from mimetypes import guess_extension
from os.path import exists


PANDOC_SUPPORTED_EXTENSIONS = {
    '.xml', '.html', '.xhtml', '.htm',
    '.docx', '.rtf', '.txt', '.md', '.odt',
    '.tex', '.epub',
}


def _part_to_markdown(part):
    mime_type = part.get_content_type()

    if mime_type == 'text/plain':
        return _part_to_unicode(part)

    extension = guess_extension(mime_type)
    if extension not in PANDOC_SUPPORTED_EXTENSIONS:
        return ''

    if mime_type.startswith('text/'):
        payload = _part_to_unicode(part).encode('UTF8')
    else:
        payload = part.get_payload(decode=True)

    if not payload:
        return ''

    with TemporaryDirectory() as temp:
        input_file = f'{temp}/input' + extension
        output_file = f'{temp}/output' + '.md'

        with open(input_file, 'wb') as fp:
            fp.write(payload)

        run(['pandoc', input_file, '-o', output_file])

        if not exists(output_file):
            return ''

        with open(output_file, encoding='UTF8') as fp:
            return fp.read()


def _part_to_unicode(part):
    encoding = part.get_content_charset()
    encodings = [encoding] if encoding else []

    text = UnicodeDammit(part.get_payload(decode=True), encodings, is_html=True).unicode_markup

    if not text:
        return ''
    return text


def _part_clean_html(part):
    return BeautifulSoup(_part_to_unicode(part), 'lxml').get_text()


def _process_with_pandoc(message):
    text = ''
    for part in message.walk():
        text += _part_to_markdown(part)
    return text


def _all_text_to_unicode(message):
    text = ''
    for part in message.walk():
        if part.get_content_type().startswith('text/'):
            text += _part_to_unicode(part)
    return text


def _body_but_unicode(message):
    text = ''
    for part in message.walk():
        if part.get_content_type() in {'text/plain', 'text/html'}:
            text += _part_to_unicode(part)
    return text


def _body_but_cleaned(message):
    text = ''
    for part in message.walk():
        if part.get_content_type() in {'text/plain', 'text/html'}:
            text += _part_clean_html(part)
    return text


MESSAGE_PROCESS_METHODS = {
    'pandoc': _process_with_pandoc,
    'unicode': _all_text_to_unicode,
    'body_cleaned': _body_but_cleaned,
    'body_unicode': _body_but_unicode,
}


def parse_milter_message(method, message_bytes):
    return MESSAGE_PROCESS_METHODS[method](email.message_from_bytes(message_bytes))


def message_builder(message_bytes, read, unique_identifier, folder_name, folder_flags, spam_status_fn, method):
    message = email.message_from_bytes(message_bytes)
    message.read = read

    label = spam_status_fn(message, folder_name, folder_flags)
    text = MESSAGE_PROCESS_METHODS[method](message)

    return Message(text, unique_identifier, label)


class Message:
    def __init__(self, text, uid, label):
        self.text = text
        self.uid = uid
        self.label = label

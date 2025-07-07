from abc import ABC, abstractmethod
from ..message import Message
import email
from pathlib import Path


class ConnectorBase(ABC):
    def __init__(self, settings):
        self.settings = settings

    def _message_factory(self, message_bytes, unique_identifier, read, folder_name, folder_flags):
        message = email.message_from_bytes(message_bytes)
        label = self.settings.get_spam_status(message, read, folder_name, folder_flags)
        return Message(message, unique_identifier, label)

    @abstractmethod
    def recipients(self):
        pass

    @abstractmethod
    def iterate_messages_for_user(self, recipient):
        pass

    def iterate_all_messages(self):
        for recipient in self.recipients():
            yield from self.iterate_messages_for_user(recipient)

    @abstractmethod
    def estimate_message_count_for_user(self, recipient):
        pass

    def estimate_total_message_count(self):
        total_count = 0
        for recipient in self.recipients():
            total_count += self.estimate_message_count_for_user(recipient)
        return total_count

    @abstractmethod
    def iterate_all_message_accessors(self):
        pass

    @abstractmethod
    def fetch_messages_for_accessors(self, accessors):
        pass

    def save_as_local_cache(self, path):
        path = Path(path)

        for recipient in self.recipients():
            spam_path = path / recipient / 'spam'
            spam_path.mkdir(parents=True)
            ham_path = path / recipient / 'ham'
            ham_path.mkdir(parents=True)
            none_path = path / recipient / 'indeterminate'
            none_path.mkdir(parents=True)

            for message in self.iterate_messages_for_user(recipient):
                if message.label is None:
                    message_path = none_path
                if message.label is True:
                    message_path = spam_path
                if message.label is False:
                    message_path = ham_path

                message_path = message_path / f'{hash(message.uid)}.eml'

                try:
                    message_bytes = bytes(message.email)
                except Exception as e:
                    print(e)

                with open(message_path, 'wb') as fp:
                    fp.write(message_bytes)

from abc import ABC, abstractmethod
from ..message import Message
import email


class ConnectorBase(ABC):
    def __init__(self, settings, config):
        self.settings = settings
        self.config = config

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

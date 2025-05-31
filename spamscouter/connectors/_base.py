from abc import ABC, abstractmethod


class ConnectorBase(ABC):
    def __init__(self, config):
        self.config = config

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

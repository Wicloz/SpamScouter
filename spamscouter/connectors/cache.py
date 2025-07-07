from ..connectors._base import ConnectorBase
from ..message import Message
from pathlib import Path
import email


class ConnectorCache(ConnectorBase):
    def __init__(self, settings):
        super().__init__(settings)
        self.path = Path(settings.cache_path)

    def recipients(self):
        for folder in self.path.iterdir():
            yield folder.name

    @staticmethod
    def _load_message(path):
        with open(path, 'rb') as fp:
            message = email.message_from_bytes(fp.read())

        if path.parent.name == 'indeterminate':
            label = None
        if path.parent.name == 'spam':
            label = True
        if path.parent.name == 'ham':
            label = False

        return Message(message, path.stem, label)

    def iterate_messages_for_user(self, recipient):
        for file in (self.path / recipient / 'indeterminate').iterdir():
            yield self._load_message(file)
        for file in (self.path / recipient / 'spam').iterdir():
            yield self._load_message(file)
        for file in (self.path / recipient / 'ham').iterdir():
            yield self._load_message(file)

    def estimate_message_count_for_user(self, recipient):
        count = 0

        for _ in (self.path / recipient / 'indeterminate').iterdir():
            count += 1
        for _ in (self.path / recipient / 'spam').iterdir():
            count += 1
        for _ in (self.path / recipient / 'ham').iterdir():
            count += 1

        return count

    def iterate_all_message_accessors(self):
        for recipient in self.recipients():
            for path in (self.path / recipient / 'indeterminate').iterdir():
                yield path.relative_to(self.path)
            for path in (self.path / recipient / 'spam').iterdir():
                yield path.relative_to(self.path)
            for path in (self.path / recipient / 'ham').iterdir():
                yield path.relative_to(self.path)

    def fetch_messages_for_accessors(self, accessors):
        for accessor in accessors:
            yield self._load_message(self.path / accessor)

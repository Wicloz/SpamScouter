from abc import ABC, abstractmethod
from .message import Message
import email
from pathlib import Path
from contextlib import contextmanager
from imap_tools import MailBox
import re


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


class ConnectorIMAP(ConnectorBase):
    def recipients(self):
        yield from self.settings.imap_recipients

    @contextmanager
    def _mailbox(self, recipient):
        with MailBox(
            self.settings.imap_host, self.settings.imap_port,
        ).login(
            self.settings.imap_get_user(recipient), self.settings.imap_get_pass(recipient),
        ) as mailbox:
            yield mailbox

    def _list_all_folders(self, mailbox):
        for folder in mailbox.folder.list():
            yield folder.name, {flag.lstrip('\\').lower() for flag in folder.flags}

    def _iterate_selected_messages_here(self, mailbox, uids, recipient, folder, flags):
        for response in mailbox._fetch_in_bulk(uids, '(RFC822 FLAGS)', False, 100):
            header = response[0][0].decode('ASCII')
            uid = int(re.search(r'UID (\d+)', header).group(1))
            read = re.search(r'FLAGS \([^\(\)]*\\Seen[^\(\)]*\)', header) is not None
            message = response[0][1]
            yield self._message_factory(message, f'{recipient}/{folder}/{uid}', read, folder, flags)

    def iterate_messages_for_user(self, recipient):
        with self._mailbox(recipient) as mailbox:

            for folder, flags in self._list_all_folders(mailbox):
                if 'sent' in flags or 'drafts' in flags or 'trash' in flags:
                    continue
                mailbox.folder.set(folder, readonly=True)

                yield from self._iterate_selected_messages_here(mailbox, mailbox.uids(), recipient, folder, flags)

    def estimate_message_count_for_user(self, recipient):
        estimate = 0

        with self._mailbox(recipient) as mailbox:
            for folder, flags in self._list_all_folders(mailbox):
                if 'sent' in flags or 'drafts' in flags or 'trash' in flags:
                    continue
                estimate += mailbox.folder.status(folder, ['MESSAGES'])['MESSAGES']

        return estimate

    def iterate_all_message_accessors(self):
        for recipient in self.recipients():
            with self._mailbox(recipient) as mailbox:
                for folder, flags in self._list_all_folders(mailbox):
                    if 'sent' in flags or 'drafts' in flags or 'trash' in flags:
                        continue

                    mailbox.folder.set(folder, readonly=True)
                    for uid in mailbox.uids():
                        yield (recipient, folder, uid)

    def fetch_messages_for_accessors(self, accessors):
        organized_accessors = {}

        for recipient, folder, uid in accessors:
            if recipient not in organized_accessors:
                organized_accessors[recipient] = {}
            if folder not in organized_accessors[recipient]:
                organized_accessors[recipient][folder] = []
            organized_accessors[recipient][folder].append(uid)

        for recipient, folder_data in organized_accessors.items():
            with self._mailbox(recipient) as mailbox:
                for folder, uids in folder_data.items():
                    for folder_name, flags in self._list_all_folders(mailbox):
                        if folder_name == folder:
                            break
                    mailbox.folder.set(folder, readonly=True)

                    yield from self._iterate_selected_messages_here(mailbox, uids, recipient, folder, flags)

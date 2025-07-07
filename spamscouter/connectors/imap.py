from ..connectors._base import ConnectorBase
from contextlib import contextmanager
from imap_tools import MailBox
import re


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

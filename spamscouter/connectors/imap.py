from ..connectors._base import ConnectorBase
from ..message import message_builder
from contextlib import contextmanager
from imap_tools import MailBox
import re


class ConnectorIMAP(ConnectorBase):
    def recipients(self):
        yield from self.config.imap_recipients

    @contextmanager
    def _mailbox(self, recipient):
        with MailBox(
            self.config.imap_host, self.config.imap_port,
        ).login(
            self.config.imap_get_user(recipient), self.config.imap_get_pass(recipient),
        ) as mailbox:
            yield mailbox

    def _list_all_folders(self, mailbox):
        for folder in mailbox.folder.list():
            yield folder.name, {flag.lstrip('\\').lower() for flag in folder.flags}

    def iterate_messages_for_user(self, recipient):
        with self._mailbox(recipient) as mailbox:

            for folder, flags in self._list_all_folders(mailbox):
                if 'sent' in flags or 'drafts' in flags or 'trash' in flags:
                    continue
                mailbox.folder.set(folder, readonly=True)

                for response in mailbox._fetch_in_bulk(mailbox.uids(), '(RFC822 FLAGS)', False, 100):
                    header = response[0][0].decode('ASCII')
                    uid = int(re.search(r'UID (\d+)', header).group(1))
                    read = re.search(r'FLAGS \([^\(\)]*\\Seen[^\(\)]*\)', header) is not None
                    message = response[0][1]
                    yield message_builder(message, read, uid, folder, flags, self.config)

    def estimate_message_count_for_user(self, recipient):
        estimate = 0

        with self._mailbox(recipient) as mailbox:
            for folder, flags in self._list_all_folders(mailbox):
                if 'sent' in flags or 'drafts' in flags or 'trash' in flags:
                    continue
                estimate += mailbox.folder.status(folder, ['MESSAGES'])['MESSAGES']

        return estimate

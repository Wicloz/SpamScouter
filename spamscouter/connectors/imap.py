from ..connectors._base import ConnectorBase
from ..message import message_builder


class ConnectorIMAP(ConnectorBase):
    def recipients(self):
        yield from self.config.recipients

    def iterate_messages_for_user(self, recipient):
        with self.config.server(recipient) as server:

            for folder in server.list_folders():
                if b'\\Sent' in folder[0] or b'\\Drafts' in folder[0] or b'\\Trash' in folder[0]:
                    continue
                server.select_folder(folder[2], readonly=True)
                flags = {flag.lstrip(b'\\').decode('ASCII') for flag in folder[0]}

                messages = server.search()
                for idx in range(0, len(messages), 100):
                    for uid, data in server.fetch(messages[idx:idx + 100], [b'BODY[]', b'FLAGS']).items():
                        yield message_builder(data[b'BODY[]'], b'\\Seen' in data[b'FLAGS'], uid, folder[2], flags, self.config)

    def estimate_message_count_for_user(self, recipient):
        estimate = 0
        with self.config.server(recipient) as server:

            for folder in server.list_folders():
                if b'\\Sent' in folder[0] or b'\\Drafts' in folder[0] or b'\\Trash' in folder[0]:
                    continue
                server.select_folder(folder[2], readonly=True)

                estimate += len(server.search())
        return estimate

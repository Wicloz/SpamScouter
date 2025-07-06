from spamscouter.settings import BaseSettings
from spamscouter.trainer import Trainer


class ScouterSettings(BaseSettings):
    CONNECTOR = 'IMAP'

    imap_host = 'example.net'
    imap_port = 993
    imap_recipients = []

    @staticmethod
    def imap_get_user(recipient):
        return recipient
        # return f'{recipient}@example.net'

    @staticmethod
    def imap_get_pass(recipient):
        return 'your_master_password'


if __name__ == '__main__':
    Trainer(ScouterSettings()).build()

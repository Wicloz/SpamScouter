from abc import ABC
from pathlib import Path


class BaseConfig(ABC):
    STORAGE = Path('/var/lib/spamscouter/')
    CONNECTOR = None

    @staticmethod
    def get_spam_status(message, folder_name, folder_flags):
        if not message.read or folder_name == 'INBOX':
            return None

        if 'junk' in folder_flags:
            return True

        return False

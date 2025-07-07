from abc import ABC
from pathlib import Path


class BaseSettings(ABC):
    STORAGE = Path('/var/lib/spamscouter/')
    CONNECTOR = None

    @staticmethod
    def get_spam_status(message, read, folder_name, folder_flags):
        if not read or folder_name == 'INBOX':
            return None

        if 'junk' in folder_flags:
            return True

        return False

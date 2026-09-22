from typing import Dict, Any, List, Tuple, Protocol
from pydantic import BaseModel

from cat import plugin
from cat.db.cruds import plugins as crud_plugins
from cat.services.string_crypto import StringCrypto

#: settings encrypted at rest: every field whose key contains "_secret"
SECRET_SETTINGS = ("mistral_api_key")

# Plugin settings
class PluginSettings(BaseModel):
    mistral_api_key: str = ""
    save_text_to_rabbit_hole: bool = False


class Crypto(Protocol):
    def encrypt(self, plaintext: str) -> str: ...

    def decrypt(self, ciphertext: str) -> str: ...


def encrypt_secrets(settings: Dict[str, Any], crypto: Crypto) -> Dict[str, Any]:
    """Copy of ``settings`` with the non-empty secrets encrypted (empty means not configured)."""
    return {
        k: crypto.encrypt(v) if k in SECRET_SETTINGS and isinstance(v, str) and v else v
        for k, v in settings.items()
    }


def decrypt_secrets(settings: Dict[str, Any], crypto: Crypto) -> Tuple[Dict[str, Any], List[str]]:
    """Copy of ``settings`` with the secrets decrypted, and the keys that could not be decrypted.

    An undecryptable secret (e.g. ``CAT_CRYPTO_KEY`` changed) becomes empty: the feature it
    enables is off until the secret is saved again, and the rest of the settings keeps working.
    """
    decrypted = dict(settings)
    failed: List[str] = []
    for key in SECRET_SETTINGS:
        value = decrypted.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            decrypted[key] = crypto.decrypt(value)
        except Exception:  # noqa: BLE001 - Fernet raises InvalidToken, base64 raises ValueError
            decrypted[key] = ""
            failed.append(key)
    return decrypted, failed


# hook to give the cat settings
@plugin
def settings_schema():
    return PluginSettings.model_json_schema()


def _decrypted(stored: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    settings, failed = decrypt_secrets(stored, StringCrypto())
    for key in failed:
        log.error(
            f"[connectors] agent {agent_id}: cannot decrypt '{key}' (was CAT_CRYPTO_KEY changed?): "
            "it is ignored until saved again"
        )
    return settings


@plugin
async def load_settings(plugin_id: str, agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.get_setting(agent_id, plugin_id)
    if stored is None:
        return PluginSettings().model_dump()
    return _decrypted(stored, agent_id)


@plugin
async def save_settings(plugin_id: str, settings: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.update_setting(agent_id, plugin_id, encrypt_secrets(settings, StringCrypto()))
    return _decrypted(stored, agent_id)

import torch
from dataclasses import dataclass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Tokens:
    def __init__(self):
        """
        These are like html instructions but in the text
        """
        self.start_token = " <START> "
        self.end_token = " <END> "
        self.pad_token = " <PAD> "
        self.unknown_token = " <UNK> "
        self.line_token = " <LN> "
        self.tab_token = " <TAB> "
        self.sep_token = " <SEP> "
        # Code tokens - start token may include attributes, e.g. <CODE language="python">
        self.code_start_token = " <CODE "  # note trailing space before attributes
        self.code_end_token = " </CODE> "
        # Image tokens - start may include attributes, e.g. <IMG src="...">
        self.img_start_token = " <IMG "
        self.img_end_token = " </IMG> "
        # Link tokens - start may include attributes, e.g. <LINK href="...">
        self.link_start_token = " <LINK "
        self.link_end_token = " </LINK> "
        # Meta tokens for arbitrary metadata blocks
        self.meta_start_token = " <META> "
        self.meta_end_token = " </META> "
        # Convenience aliases
        self.bos_token = self.start_token
        self.eos_token = self.end_token

    def __repr__(self):
        return f"Tokens(start={self.start_token!r}, end={self.end_token!r})"

    def to_dict(self):
        """Return a shallow dict of all token attributes."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        """Create a Tokens instance from a dict produced by `to_dict`."""
        obj = cls()
        for k, v in d.items():
            setattr(obj, k, v)
        return obj

    def serialize(self):
        """Serialize tokens to a JSON string."""
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def deserialize(cls, s):
        """Deserialize tokens from a JSON string."""
        import json

        return cls.from_dict(json.loads(s))



@dataclass(slots=True)
class Neuromodulators:
    """
    Escalares globales que modulan comportamiento.
    """
    dopamina: float = 0.35
    serotonina: float = 0.35
    noradrenalina: float = 0.35
    acetilcolina: float = 0.35
    adrenalina: float = 0.05
    curiosidad: float = 0.50
    fatiga: float = 0.00
    confianza: float = 0.50

    _FIELDS = {"dopamina", "serotonina", "noradrenalina", "acetilcolina", "adrenalina", "curiosidad", "fatiga", "confianza"}

    def clamp(self) -> None:
        self.dopamina = max(0.0, min(1.0, float(self.dopamina)))
        self.serotonina = max(0.0, min(1.0, float(self.serotonina)))
        self.noradrenalina = max(0.0, min(1.0, float(self.noradrenalina)))
        self.acetilcolina = max(0.0, min(1.0, float(self.acetilcolina)))
        self.adrenalina = max(0.0, min(1.0, float(self.adrenalina)))
        self.curiosidad = max(0.0, min(1.0, float(self.curiosidad)))
        self.fatiga = max(0.0, min(1.0, float(self.fatiga)))
        self.confianza = max(0.0, min(1.0, float(self.confianza)))

    def as_vector(self, device: torch.device = DEVICE) -> torch.Tensor:
        return torch.tensor(
            [
                self.dopamina,
                self.serotonina,
                self.noradrenalina,
                self.acetilcolina,
                self.adrenalina,
                self.curiosidad,
                self.fatiga,
                self.confianza,
            ],
            dtype=DTYPE,
            device=device,
        )

    def boost(self, name: str, delta: float) -> None:
        if name not in self._FIELDS:
            raise KeyError(f"Neuromodulador desconocido: {name}")
        current_val = float(getattr(self, name))
        clamped_val = max(0.0, min(1.0, current_val + delta))
        setattr(self, name, clamped_val)
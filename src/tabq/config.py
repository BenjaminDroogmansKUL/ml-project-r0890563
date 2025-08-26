import json
from pathlib import Path


class Config:
    """
    Stores parameters and can save +load from JSON.
    """

    def __init__(self, **kwargs):
        # accept arbitrary key/value pairs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return self.__dict__

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

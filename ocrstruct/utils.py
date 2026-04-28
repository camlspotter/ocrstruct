from pathlib import Path
from typing import Self, TypeVar, Type
from pydantic import BaseModel, TypeAdapter


class BaseModelWithSave(BaseModel):
    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        tmp_path.write_text(self.model_dump_json(indent=2, exclude_defaults=True), encoding="utf-8")
        tmp_path.replace(target)

    @classmethod
    def load_json(cls, path: str | Path) -> Self:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


T = TypeVar('T')


def save_json(typ : Type[T], path : str|Path, data : T) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_text(
        TypeAdapter(typ).dump_json(data, indent=2, exclude_defaults= True).decode("utf-8"),
        encoding="utf-8",
    )
    tmp_path.replace(target)


def load_json(typ : Type[T], path : str | Path) -> T | None:
    target = Path(path)
    if target.exists():
        try:
            return TypeAdapter(typ).validate_json(target.read_text(encoding="utf-8"))
        except Exception as e:
            print(f'Failed to parse {target} as {typ.__name__}')
            raise e
    else:
        return None

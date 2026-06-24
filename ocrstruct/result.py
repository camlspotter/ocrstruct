from ocrstruct.middle import Middle
from ocrstruct.utils import BaseModelWithSave


class Parameters(BaseModelWithSave):
    source_checksum: str
    backend: str | None
    method: str | None
    lang: str | None
    seal_enable: bool
    formula_enable: bool
    with_image_understanding: bool
    image_screening_model: str | None
    image_understanding_model: str | None

    def without_image_understanding(self) -> 'Parameters':
        return Parameters(
            source_checksum= self.source_checksum,
            backend= self.backend,
            method= self.method,
            lang= self.lang,
            seal_enable= self.seal_enable,
            formula_enable= self.formula_enable,
            with_image_understanding= False,
            image_screening_model= None,
            image_understanding_model= None,
        )


dummy_parameters = Parameters(
    source_checksum= 'dummy',
    backend= None,
    method= None,
    lang= None,
    seal_enable= False,
    formula_enable= False,
    with_image_understanding= False,
    image_screening_model= None,
    image_understanding_model= None,
)


# middle.json has this type
class Result(BaseModelWithSave):
    middle: Middle
    source_path: str
    extracted_by: str
    parameters: Parameters

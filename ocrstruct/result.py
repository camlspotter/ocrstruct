from ocrstruct.middle import Middle
from utils import BaseModelWithSave


# middle.json has this type
class Result(BaseModelWithSave):
    middle: Middle
    source_path: str
    extracted_by: str

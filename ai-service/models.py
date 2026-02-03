from pydantic import BaseModel

class TitleModel(BaseModel):
    title: str

class DescriptionModel(TitleModel):
    description: str

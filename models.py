from typing import List, Dict, Tuple
from pydantic import BaseModel

class Cat(BaseModel):
    position: int
    name: str

class Mouse(BaseModel):
    position: int
    name: str
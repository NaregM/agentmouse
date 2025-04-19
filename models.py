from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel

class Cat(BaseModel):
    position: int
    name: Optional[str] = None

class Mouse(BaseModel):
    position: int
    name: Optional[str] = None
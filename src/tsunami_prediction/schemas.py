# src/tsunami_prediction/schemas.py

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# --------- Tectonic ---------
class TectonicItem(BaseModel):
    mag: float = Field(..., description="Earthquake magnitude")
    depth: float = Field(..., description="Hypocenter depth (km)")
    latitude: float
    longitude: float
    country: Optional[str] = None
    zone: Optional[str] = Field(None, description="Tectonic zone name (if used in training)")
    distance_to_coast_km: Optional[float] = 0.0
    is_subduction_zone: Optional[int] = 0  # 0/1

class TectonicRequest(BaseModel):
    datas: List[TectonicItem]


# --------- Volcanic ---------
class VolcanicItem(BaseModel):
    country: Optional[str] = None
    type: Optional[str] = Field(None, description="Volcano type (e.g., Stratovolcano/Caldera)")
    latitude: float
    longitude: float
    eq: Optional[float] = 0.0
    elevation: Optional[float] = 0.0
    vei: Optional[int] = 0
    distance_to_coast_km: Optional[float] = 0.0
    is_subduction_zone: Optional[int] = 0  # 0/1

class VolcanicRequest(BaseModel):
    datas: List[VolcanicItem]
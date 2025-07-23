from pydantic import BaseModel
from typing import List

# --- Tectonic Earthquake Input (pastikan konsisten dengan model dan encoder)
class TektonikSample(BaseModel):
    mag: float                     # Earthquake Magnitude
    depth: float                   # Focal Depth (km)
    latitude: float
    longitude: float
    country: str                   # Negara, HARUS ada jika OHE di training pakai country
    is_subduction_zone: int        # 1 jika zona subduksi, 0 bukan
    distance_to_coast_km: float    # Jarak ke pantai (coastline)
    # zone: str                    # Bisa diaktifkan jika pipeline butuh, tapi biasanya sudah digantikan dengan country+is_subduction_zone

class TektonikRequest(BaseModel):
    datas: List[TektonikSample]


# --- Volcanic Eruption Input (pastikan konsisten dengan model dan encoder)
class VulSample(BaseModel):
    eq: float                      # Earthquake Magnitude
    elevation: float               # Elevation in meters
    latitude: float
    longitude: float
    country: str                   # Negara
    type: str                      # Volcano type, e.g. 'Stratovolcano'
    vei: float                     # Volcanic Explosivity Index
    distance_to_coast_km: float    # Jarak ke pantai (coastline)
    is_subduction_zone: int        # 1 jika zona subduksi, 0 bukan

class VulRequest(BaseModel):
    datas: List[VulSample]

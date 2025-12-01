# src/tsunami_prediction/schemas.py
from __future__ import annotations

# cSpell:ignore vei

from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
#  COMMON: HASIL PREDIKSI (UNTUK RESPONSE API)
# ============================================================
class TsunamiPredictionBase(BaseModel):
    """
    Skema dasar hasil prediksi tsunami untuk satu kejadian.
    Dipakai baik untuk gempa tektonik maupun gunung api.

    Catatan:
    - predicted_label: 0 = non-tsunami, 1 = tsunami
    - Untuk endpoint khusus (tektonik / vulkanik), makna label_name
      bisa dijelaskan di dokumentasi (mis. 'Tectonic tsunami').
    """

    predicted_label: int = Field(
        ...,
        ge=0,
        le=1,
        description="Label biner hasil prediksi: 0 = non-tsunami, 1 = tsunami.",
    )
    predicted_label_name: str = Field(
        ...,
        description="Nama/keterangan label prediksi (mis. 'Non-tsunami' atau 'Tsunami').",
    )
    prob_non_tsunami: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilitas kelas 0 (non-tsunami).",
    )
    prob_tsunami: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilitas kelas 1 (tsunami).",
    )
    decision_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold keputusan yang dipakai model saat mengubah "
            "probabilitas menjadi label biner (mis. 0.50 atau 0.88)."
        ),
    )


# ============================================================
#  TECTONIC SCHEMAS (REQUEST + RESPONSE)
# ============================================================
class TectonicItem(BaseModel):
    """
    Satu kejadian gempa tektonik untuk inferensi API.

    Kolom diselaraskan dengan tabel data clean tektonik:
    id, year, month, day, hr, mn, sec, country, area, region, location,
    latitude, longitude, depth, mag, plus fitur rekayasa:
    distance_to_coast_km dan is_subduction_zone.

    Hanya subset numerik utama yang digunakan langsung oleh model;
    kolom lain tetap diterima untuk keperluan metadata/logging.
    """

    # --- informasi identitas & waktu (opsional) ---
    id: Optional[float] = Field(
        None,
        description="Event ID dari katalog (jika tersedia). Tidak digunakan langsung oleh model.",
    )
    year: Optional[float] = Field(
        None,
        description="Tahun kejadian gempa (Gregorian, sesuai katalog).",
    )
    month: Optional[float] = Field(
        None,
        ge=1,
        le=12,
        description="Bulan kejadian gempa (1–12).",
    )
    day: Optional[float] = Field(
        None,
        ge=1,
        le=31,
        description="Tanggal kejadian gempa (1–31).",
    )
    hr: Optional[float] = Field(
        None,
        ge=0,
        le=23,
        description="Jam origin time (0–23).",
    )
    mn: Optional[float] = Field(
        None,
        ge=0,
        le=59,
        description="Menit origin time (0–59).",
    )
    sec: Optional[float] = Field(
        None,
        ge=0,
        le=59,
        description="Detik origin time (0–59).",
    )

    # --- parameter fisis utama (WAJIB untuk model) ---
    mag: float = Field(
        ...,
        description="Magnitudo gempa (mis. Mw). Fitur utama untuk model tsunami tektonik.",
    )
    depth: float = Field(
        ...,
        ge=0,
        description="Kedalaman hiposenter (km, >=0).",
    )
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Lintang episenter (derajat).",
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Bujur episenter (derajat).",
    )

    # --- konteks geografis (kategorikal, opsional) ---
    country: Optional[str] = Field(
        None,
        description="Negara tempat kejadian (sesuai katalog).",
    )
    area: Optional[str] = Field(
        None,
        description="Area/basin tektonik (opsional, sesuai kolom 'area' pada data clean).",
    )
    region: Optional[str] = Field(
        None,
        alias="zone",
        description=(
            "Nama/kode region tektonik. "
            "Pada versi terdahulu disebut 'zone'; kedua nama field diterima oleh API."
        ),
    )
    location: Optional[str] = Field(
        None,
        description="Lokasi deskriptif (kota terdekat / lokasi katalog).",
    )

    # --- fitur rekayasa (opsional) ---
    distance_to_coast_km: Optional[float] = Field(
        0.0,
        ge=0,
        description=(
            "Jarak episenter ke garis pantai terdekat (km). "
            "Jika tidak diketahui, biarkan default dan pipeline dapat menghitung sendiri."
        ),
    )
    is_subduction_zone: Optional[int] = Field(
        0,
        ge=0,
        le=1,
        description=(
            "1 jika kejadian berada di zona subduksi utama, 0 jika tidak. "
            "Dapat diabaikan karena bisa diturunkan dari 'country' / region."
        ),
    )

    class Config:
        # agar bisa menerima JSON dengan key 'zone' maupun 'region'
        allow_population_by_field_name = True


class TectonicRequest(BaseModel):
    """
    Request batch untuk prediksi tsunami tektonik.

    Contoh bentuk JSON:

    {
      "datas": [
        {
          "year": 2024,
          "month": 1,
          "day": 1,
          "mag": 7.5,
          "depth": 10.0,
          "latitude": -3.5,
          "longitude": 135.2
        }
      ]
    }
    """

    datas: List[TectonicItem]


class TectonicPrediction(TsunamiPredictionBase):
    """
    Hasil prediksi untuk satu TectonicItem.
    Bisa ditambahkan field echo input_id kalau diperlukan.
    """

    index: int = Field(
        ...,
        ge=0,
        description="Indeks item dalam batch request (0-based).",
    )


class TectonicResponse(BaseModel):
    """
    Response batch untuk prediksi tsunami tektonik.
    """

    model_name: str = Field(
        ...,
        description="Nama model stacking yang digunakan (mis. 'events_smote_stacking_lr').",
    )
    model_version: Optional[str] = Field(
        None,
        description="Versi model / timestamp training (opsional).",
    )
    n_items: int = Field(
        ...,
        ge=0,
        description="Jumlah item yang diprediksi.",
    )
    predictions: List[TectonicPrediction]


# ============================================================
#  VOLCANIC SCHEMAS (REQUEST + RESPONSE)
# ============================================================
class VolcanicItem(BaseModel):
    """
    Satu kejadian gunung api untuk inferensi API.

    Kolom selaras dengan tabel data clean vulkanik:
    id, year, month, day, name, location, country, latitude, longitude,
    elevation, type, status, vei, eq, agent, plus fitur rekayasa
    distance_to_coast_km dan is_subduction_zone.

    Hanya subset numerik utama yang digunakan langsung oleh model;
    kolom lain tetap diterima untuk keperluan metadata/logging.
    """

    # --- identitas & waktu (opsional) ---
    id: Optional[float] = Field(
        None,
        description="ID gunung api / event (jika tersedia dalam katalog).",
    )
    year: Optional[float] = Field(
        None,
        description="Tahun letusan.",
    )
    month: Optional[float] = Field(
        None,
        ge=1,
        le=12,
        description="Bulan letusan (1–12).",
    )
    day: Optional[float] = Field(
        None,
        ge=1,
        le=31,
        description="Tanggal letusan (1–31).",
    )

    # --- informasi gunung api & lokasi (kategorikal + koordinat) ---
    name: Optional[str] = Field(
        None,
        description="Nama gunung api.",
    )
    location: Optional[str] = Field(
        None,
        description="Lokasi deskriptif / kota terdekat.",
    )
    country: Optional[str] = Field(
        None,
        description="Negara lokasi gunung api.",
    )

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Lintang puncak gunung api (derajat).",
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Bujur puncak gunung api (derajat).",
    )

    elevation: Optional[float] = Field(
        0.0,
        description="Elevasi puncak (meter di atas permukaan laut).",
    )
    type: Optional[str] = Field(
        None,
        description="Tipe gunung api (mis. Stratovolcano, Shield, dsb.).",
    )
    status: Optional[str] = Field(
        None,
        description="Status aktivitas (mis. Historical, Holocene, dsb.).",
    )

    # --- parameter erupsi ---
    vei: Optional[float] = Field(
        0.0,
        ge=0,
        le=8,
        description="Volcanic Explosivity Index (0–8).",
    )
    eq: Optional[float] = Field(
        0.0,
        ge=0,
        description="Indikator aktivitas seismik terkait (mis. jumlah gempa, indeks energi).",
    )
    agent: Optional[str] = Field(
        None,
        description="Agen pemicu / tipe erupsi (jika tersedia di katalog).",
    )

    # --- fitur rekayasa ---
    distance_to_coast_km: Optional[float] = Field(
        0.0,
        ge=0,
        description=(
            "Jarak puncak ke garis pantai terdekat (km). "
            "Jika tidak diisi, pipeline dapat menghitung sendiri dari shapefile pantai."
        ),
    )
    is_subduction_zone: Optional[int] = Field(
        0,
        ge=0,
        le=1,
        description=(
            "1 jika gunung api berada pada busur subduksi utama, 0 jika tidak. "
            "Dapat diturunkan dari 'country' / setting tektonik regional."
        ),
    )


class VolcanicRequest(BaseModel):
    """
    Request batch untuk prediksi tsunami vulkanik.

    Contoh bentuk JSON:

    {
      "datas": [
        {
          "year": 2020,
          "name": "Example Volcano",
          "latitude": -1.2,
          "longitude": 120.5,
          "vei": 4,
          "eq": 10
        }
      ]
    }
    """

    datas: List[VolcanicItem]


class VolcanicPrediction(TsunamiPredictionBase):
    """
    Hasil prediksi untuk satu VolcanicItem.
    """

    index: int = Field(
        ...,
        ge=0,
        description="Indeks item dalam batch request (0-based).",
    )


class VolcanicResponse(BaseModel):
    """
    Response batch untuk prediksi tsunami vulkanik.
    """

    model_name: str = Field(
        ...,
        description="Nama model stacking yang digunakan (mis. 'events_smote_stacking_lr').",
    )
    model_version: Optional[str] = Field(
        None,
        description="Versi model / timestamp training (opsional).",
    )
    n_items: int = Field(
        ...,
        ge=0,
        description="Jumlah item yang diprediksi.",
    )
    predictions: List[VolcanicPrediction]
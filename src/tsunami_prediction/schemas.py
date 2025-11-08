# src/tsunami_prediction/schemas.py
from __future__ import annotations

# cSpell:ignore vei

from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================
#   TECTONIC SCHEMAS
# ==========================
class TectonicItem(BaseModel):
    """
    Satu kejadian gempa tektonik untuk inferensi API.

    Kolom utamanya diselaraskan dengan tabel data clean tektonik:
    id, year, month, day, hr, mn, sec, country, area, region, location,
    latitude, longitude, depth, mag, plus dua fitur rekayasa utama:
    distance_to_coast_km dan is_subduction_zone.
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

    # --- parameter fisis utama (wajib) ---
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

    # --- konteks geografis (kategorikal) ---
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
            "Pada versi terdahulu disebut 'zone'; keduanya diterima oleh API."
        ),
    )
    location: Optional[str] = Field(
        None,
        description="Lokasi deskriptif (kota terdekat / lokasi katalog).",
    )

    # --- fitur rekayasa (opsional; akan dihitung ulang bila pipeline mendukung) ---
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
            "1 jika kejadian berada di negara/subduksi utama, 0 jika tidak. "
            "Dapat diabaikan karena bisa diturunkan dari 'country'."
        ),
    )

    class Config:
        # agar bisa menerima JSON dengan key 'zone' maupun 'region'
        allow_population_by_field_name = True


class TectonicRequest(BaseModel):
    """
    Request batch untuk prediksi tsunami tektonik.
    """
    datas: List[TectonicItem]


# ==========================
#   VOLCANIC SCHEMAS
# ==========================
class VolcanicItem(BaseModel):
    """
    Satu kejadian gunung api untuk inferensi API.

    Kolom selaras dengan tabel data clean vulkanik:
    id, year, month, day, name, location, country, latitude, longitude,
    elevation, type, status, vei, eq, agent, plus fitur rekayasa
    distance_to_coast_km dan is_subduction_zone.
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
            "Dapat diturunkan dari 'country'."
        ),
    )


class VolcanicRequest(BaseModel):
    """
    Request batch untuk prediksi tsunami vulkanik.
    """
    datas: List[VolcanicItem]
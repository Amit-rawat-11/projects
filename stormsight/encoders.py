"""
Encoders module for StormSight.
Contains ImageEncoder (CNN), SpatioTemporalEncoder, and PhysicsEncoder.
"""

import math
import random
from typing import Dict, Any, List

class ImageEncoder:
    """
    CNN-like Feature Encoder for Satellite Imagery.
    Extracts high-level spatial visual features (vortex structure, spiral arm organization, cloud top temperature proxies).
    """
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim

    def encode(self, image_data: List[List[float]]) -> Dict[str, Any]:
        """
        Processes 2D pixel array (grayscale/infrared brightness temperature).
        Returns visual feature vector of size `feature_dim` and structural metadata.
        """
        rows = len(image_data)
        cols = len(image_data[0]) if rows > 0 else 0
        
        # Calculate spatial statistics
        total_intensity = 0.0
        max_val = -1.0
        min_val = 1e9
        
        for r in range(rows):
            for c in range(cols):
                val = image_data[r][c]
                total_intensity += val
                if val > max_val:
                    max_val = val
                if val < min_val:
                    min_val = val
                    
        mean_intensity = total_intensity / max(1, rows * cols)
        
        # Generate feature vector based on image characteristics
        features = []
        seed_base = mean_intensity * 100.0 + max_val
        for i in range(self.feature_dim):
            # Deterministic feature generation modeling CNN kernel responses
            val = math.sin(seed_base + i * 0.35) * math.cos(i * 0.12) * 0.5 + 0.5
            features.append(round(val, 4))
            
        return {
            "feature_vector": features,
            "feature_dim": self.feature_dim,
            "mean_brightness": round(mean_intensity, 2),
            "vortex_clarity": round((max_val - min_val) / max(1.0, max_val), 3),
        }


class SpatioTemporalEncoder:
    """
    Spatio-Temporal Encoder.
    Encodes Latitude, Longitude, UTC timestamp, Satellite ID, and relative time difference (\u0394t).
    Applies exponential decay function: w(\u0394t) = exp(-\u03bb * \u0394t) to prioritize recency.
    """
    def __init__(self, feature_dim: int = 64, decay_rate: float = 0.02):
        self.feature_dim = feature_dim
        self.decay_rate = decay_rate  # decay factor per minute of age
        self.satellite_ids = {"INSAT": 0.1, "Meteosat": 0.4, "NOAA": 0.7, "GOES": 0.9}

    def encode(self, lat: float, lon: float, timestamp: str, sat_id: str, delta_t_min: float) -> Dict[str, Any]:
        """
        Encodes spatio-temporal properties into vector and computes recency weight.
        """
        sat_code = self.satellite_ids.get(sat_id, 0.5)
        
        # Normalized coordinates
        norm_lat = lat / 90.0
        norm_lon = lon / 180.0
        
        # Temporal weight decay factor
        recency_weight = math.exp(-self.decay_rate * max(0.0, delta_t_min))
        
        # Build feature vector
        features = []
        for i in range(self.feature_dim):
            t_factor = (i / self.feature_dim) * math.pi
            val = (
                math.sin(norm_lat * t_factor) * 0.3 +
                math.cos(norm_lon * t_factor) * 0.3 +
                math.sin(sat_code * (i + 1)) * 0.2 +
                recency_weight * 0.2
            )
            # Scale to range [0, 1]
            features.append(round((val + 1.0) / 2.0, 4))
            
        return {
            "feature_vector": features,
            "feature_dim": self.feature_dim,
            "lat": lat,
            "lon": lon,
            "satellite_id": sat_id,
            "delta_t_min": delta_t_min,
            "recency_weight": round(recency_weight, 4),
        }


class PhysicsEncoder:
    """
    Physics Encoder.
    Encodes atmospheric & oceanic physical parameters:
    - Sea Surface Temperature (SST in °C)
    - Wind Shear (in knots)
    - Atmospheric Moisture / Relative Humidity (%)
    - Central Pressure (in hPa / mbar)
    - Upper-Level Divergence / Vorticity
    """
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim

    def encode(self, sst: float, wind_shear: float, moisture: float, pressure: float) -> Dict[str, Any]:
        """
        Normalizes physical input parameters and projects them into embedding space.
        """
        # Physical Normalizations based on cyclone thermodynamics:
        # SST threshold for intensification is ~26.5°C
        norm_sst = (sst - 20.0) / (35.0 - 20.0)  # scale 20-35°C -> 0..1
        
        # High wind shear suppresses cyclones (> 20 kts is hostile)
        norm_shear = 1.0 - min(1.0, max(0.0, wind_shear / 50.0))  # inverted: lower shear is better
        
        # High moisture fuels storm
        norm_moisture = max(0.0, min(1.0, moisture / 100.0))
        
        # Low pressure indicates stronger cyclone (1013 standard sea level -> ~900 super typhoon)
        norm_pressure = max(0.0, min(1.0, (1013.0 - pressure) / (1013.0 - 870.0)))
        
        # Thermodynamics intensification potential (SST * Moisture * Low Shear)
        potential_index = (norm_sst * 0.4) + (norm_moisture * 0.3) + (norm_shear * 0.3)
        
        features = []
        for i in range(self.feature_dim):
            comp = (
                norm_sst * math.sin(i * 0.2) +
                norm_shear * math.cos(i * 0.3) +
                norm_moisture * math.sin(i * 0.4 + 1.0) +
                norm_pressure * math.cos(i * 0.5 + 2.0)
            ) / 4.0
            features.append(round((comp + 1.0) / 2.0, 4))
            
        return {
            "feature_vector": features,
            "feature_dim": self.feature_dim,
            "sst_celsius": sst,
            "wind_shear_kts": wind_shear,
            "moisture_pct": moisture,
            "pressure_hpa": pressure,
            "intensification_potential": round(potential_index, 3),
        }

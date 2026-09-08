"""
Feature Fusion module for StormSight.
Combines image features + location-time features + physical features.
"""

from typing import List, Dict, Any

class FeatureFusion:
    """
    Feature Fusion Layer.
    Fuses multi-modal embeddings across multiple asynchronous satellite observations
    and physical environmental parameters into a unified feature representation.
    """
    def __init__(self, fused_dim: int = 256):
        self.fused_dim = fused_dim

    def fuse(
        self,
        multi_sat_encodings: List[Dict[str, Any]],
        physics_encoding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuses a list of multi-satellite encodings (each having image + spatio-temporal features)
        along with the physical environment encoding.
        
        Applies weighted recency pooling across satellite sources.
        """
        if not multi_sat_encodings:
            raise ValueError("multi_sat_encodings list cannot be empty")
            
        # Calculate recency weights sum for normalization
        total_recency_weight = sum(
            sat["st_encoding"]["recency_weight"] for sat in multi_sat_encodings
        )
        if total_recency_weight <= 0:
            total_recency_weight = 1.0

        # Weighted aggregation of satellite image features & spatio-temporal features
        img_dim = len(multi_sat_encodings[0]["img_encoding"]["feature_vector"])
        st_dim = len(multi_sat_encodings[0]["st_encoding"]["feature_vector"])
        
        aggregated_img_feats = [0.0] * img_dim
        aggregated_st_feats = [0.0] * st_dim
        
        for sat in multi_sat_encodings:
            w = sat["st_encoding"]["recency_weight"] / total_recency_weight
            img_vec = sat["img_encoding"]["feature_vector"]
            st_vec = sat["st_encoding"]["feature_vector"]
            
            for i in range(img_dim):
                aggregated_img_feats[i] += img_vec[i] * w
            for i in range(st_dim):
                aggregated_st_feats[i] += st_vec[i] * w

        phys_feats = physics_encoding["feature_vector"]
        phys_dim = len(phys_feats)

        # Build combined concatenated feature vector
        raw_concat = aggregated_img_feats + aggregated_st_feats + phys_feats
        
        # Projection to fixed fused dimension size
        fused_vector = []
        raw_len = len(raw_concat)
        for i in range(self.fused_dim):
            # Non-linear feature mixing
            idx1 = i % raw_len
            idx2 = (i * 3 + 7) % raw_len
            val = (raw_concat[idx1] * 0.6) + (raw_concat[idx2] * 0.4)
            fused_vector.append(round(val, 4))

        return {
            "fused_vector": fused_vector,
            "fused_dim": self.fused_dim,
            "satellite_count": len(multi_sat_encodings),
            "modalities": {
                "image_dim": img_dim,
                "spatio_temporal_dim": st_dim,
                "physics_dim": phys_dim,
            },
            "total_recency_weight": round(total_recency_weight, 3),
        }

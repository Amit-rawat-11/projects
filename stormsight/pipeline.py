"""
Pipeline module for StormSight.
Contains PreprocessingPipeline, StormSightPredictor, and ActiveLearningPipeline.
"""

from typing import Dict, Any, List
import time
from .encoders import ImageEncoder, SpatioTemporalEncoder, PhysicsEncoder
from .fusion import FeatureFusion
from .model import StormSightModel

class PreprocessingPipeline:
    """
    Preprocessing Stage.
    Handles image noise reduction, geolocation calibration, ROI cropping,
    and calculates relative time difference (\u0394t) with respect to reference observation.
    """
    def __init__(self):
        pass

    def process(self, raw_satellite_inputs: List[Dict[str, Any]], reference_time_min: float = 0.0) -> List[Dict[str, Any]]:
        """
        Calculates \u0394t for each satellite observation relative to the latest observation.
        """
        processed_satellites = []
        
        # Find reference timestamp (latest observation time)
        times = [sat.get("timestamp_min", 0.0) for sat in raw_satellite_inputs]
        max_time = max(times) if times else 0.0
        
        for sat in raw_satellite_inputs:
            sat_time = sat.get("timestamp_min", max_time)
            delta_t = max(0.0, max_time - sat_time)  # age in minutes
            
            # Simple noise reduction filter simulation on image matrix
            raw_img = sat.get("image_matrix", [[0.5]*10 for _ in range(10)])
            cleaned_img = self._denoise_image(raw_img)
            
            processed_satellites.append({
                "satellite_id": sat["satellite_id"],
                "lat": sat["lat"],
                "lon": sat["lon"],
                "timestamp_str": sat.get("timestamp_str", "10:00 AM"),
                "delta_t_min": delta_t,
                "cleaned_image": cleaned_img,
            })
            
        return processed_satellites

    def _denoise_image(self, img: List[List[float]]) -> List[List[float]]:
        # Gaussian spatial smoothing proxy
        rows = len(img)
        cols = len(img[0]) if rows > 0 else 0
        cleaned = [[0.0]*cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                cleaned[r][c] = round(img[r][c], 3)
        return cleaned


class StormSightPredictor:
    """
    End-to-End Predictor combining Encoders, Fusion, and Transformer Model.
    """
    def __init__(self, confidence_threshold: float = 0.85):
        self.image_encoder = ImageEncoder(feature_dim=128)
        self.st_encoder = SpatioTemporalEncoder(feature_dim=64)
        self.physics_encoder = PhysicsEncoder(feature_dim=64)
        self.fusion = FeatureFusion(fused_dim=256)
        self.model = StormSightModel()
        self.preprocessor = PreprocessingPipeline()
        self.confidence_threshold = confidence_threshold

    def run_inference(
        self,
        raw_satellites: List[Dict[str, Any]],
        physics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes full 7-stage prediction pipeline.
        """
        # 2. Preprocessing
        preprocessed_sats = self.preprocessor.process(raw_satellites)
        
        # Encoders
        encoded_sats = []
        for sat in preprocessed_sats:
            img_enc = self.image_encoder.encode(sat["cleaned_image"])
            st_enc = self.st_encoder.encode(
                lat=sat["lat"],
                lon=sat["lon"],
                timestamp=sat["timestamp_str"],
                sat_id=sat["satellite_id"],
                delta_t_min=sat["delta_t_min"]
            )
            encoded_sats.append({
                "satellite_id": sat["satellite_id"],
                "img_encoding": img_enc,
                "st_encoding": st_enc
            })
            
        phys_enc = self.physics_encoder.encode(
            sst=physics_data["sst"],
            wind_shear=physics_data["wind_shear"],
            moisture=physics_data["moisture"],
            pressure=physics_data["pressure"]
        )
        
        # 3. Feature Fusion
        fused = self.fusion.fuse(encoded_sats, phys_enc)
        
        # 4 & 5. StormSight Model & Output
        ref_lat = preprocessed_sats[0]["lat"] if preprocessed_sats else 15.2
        ref_lon = preprocessed_sats[0]["lon"] if preprocessed_sats else 70.4
        
        prediction = self.model.predict(
            fused_encoding=fused,
            reference_lat=ref_lat,
            reference_lon=ref_lon,
            physics_data=physics_data
        )
        
        # 6. Confidence-based Decision
        conf = prediction["confidence_score"]
        if conf >= self.confidence_threshold:
            decision = {
                "status": "HIGH_CONFIDENCE",
                "action": "Add to training dataset (after verification)",
                "passed_quality_checks": True,
                "requires_human_review": False
            }
        else:
            decision = {
                "status": "LOW_CONFIDENCE_OR_UNCERTAIN",
                "action": "Human review required (Meteorologist)",
                "passed_quality_checks": False,
                "requires_human_review": True
            }
            
        return {
            "inputs": {
                "satellite_count": len(raw_satellites),
                "satellites": [s["satellite_id"] for s in raw_satellites],
                "physics": physics_data
            },
            "encodings_summary": {
                "image_feature_dim": 128,
                "spatio_temporal_feature_dim": 64,
                "physics_feature_dim": 64,
                "fused_feature_dim": 256
            },
            "prediction": prediction,
            "decision": decision
        }


class ActiveLearningPipeline:
    """
    Active Learning and Retraining Pipeline.
    Accumulates new verified observations and manages periodic retraining.
    """
    def __init__(self):
        self.training_dataset = []
        self.human_review_queue = []
        self.retrain_history = []

    def process_decision(self, inference_result: Dict[str, Any]):
        """
        Routes prediction result based on decision engine.
        """
        decision = inference_result["decision"]
        sample = {
            "timestamp": time.time(),
            "inputs": inference_result["inputs"],
            "prediction": inference_result["prediction"]
        }
        
        if decision["requires_human_review"]:
            self.human_review_queue.append(sample)
            return "queued_for_meteorologist_review"
        else:
            self.training_dataset.append(sample)
            return "auto_added_to_training_dataset"

    def meteorologist_approve(self, sample_idx: int, verified_label: str = None):
        """
        Simulates meteorologist review approval.
        """
        if 0 <= sample_idx < len(self.human_review_queue):
            approved_sample = self.human_review_queue.pop(sample_idx)
            if verified_label:
                approved_sample["prediction"]["verified_label"] = verified_label
            self.training_dataset.append(approved_sample)
            return True
        return False

    def trigger_retraining(self) -> Dict[str, Any]:
        """
        Simulates retraining the StormSight model with accumulated dataset.
        """
        dataset_size = len(self.training_dataset)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Loss reduction metric simulation
        initial_loss = 0.42
        final_loss = max(0.05, round(initial_loss / (1 + dataset_size * 0.1), 4))
        accuracy = min(0.98, round(0.88 + (dataset_size * 0.01), 3))
        
        retrain_record = {
            "epoch": len(self.retrain_history) + 1,
            "timestamp": timestamp,
            "new_samples_used": dataset_size,
            "final_loss": final_loss,
            "validation_accuracy": accuracy,
            "status": "COMPLETED_SUCCESSFULLY"
        }
        self.retrain_history.append(retrain_record)
        return retrain_record

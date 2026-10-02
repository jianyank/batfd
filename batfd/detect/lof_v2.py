"""版本隔离的 LOF 修正：仅训练分数改用标准 LOF 口径。"""
from .lof import LOFDetector


class LOFDetectorV2(LOFDetector):
    """保留旧尺度、留出评分和 per_pack 行为。"""

    def fit(self, x_train) -> "LOFDetectorV2":
        """复用旧拟合流程，训练分数取 -negative_outlier_factor_。"""
        super().fit(x_train)
        if self.mode == "train_novelty":
            self._train_score = -self.model_.negative_outlier_factor_
        return self

import numpy as np
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x
    
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a numpy array"
        return x
    
    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
# Test data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Testing MinMaxScaler
min_max_scaler = MinMaxScaler()
min_max_scaled_data = min_max_scaler.fit_transform(data)
print("MinMax Scaled Data:")
print(min_max_scaled_data)

# Testing StandardScaler
standard_scaler = StandardScaler()
standard_scaled_data = standard_scaler.fit_transform(data)
print("\nStandard Scaled Data:")
print(standard_scaled_data)
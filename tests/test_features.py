from aim5005.features import MinMaxScaler, StandardScaler
import numpy as np
import unittest
from unittest.case import TestCase
from sklearn.preprocessing import MinMaxScaler as MMS
### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line
    # Test data
    #def test_min_max_scaler(self):
        #data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Testing MinMaxScaler
        #min_max_scaler = MinMaxScaler()
        #min_max_scaled_data = min_max_scaler.fit_transform(data)
        #print("MinMax Scaled Data:")
        #print(min_max_scaled_data)
        #mms = MMS()
        #assert (min_max_scaled_data == mms.fit_transform(data)).all()

# # Testing StandardScaler
# standard_scaler = StandardScaler()
# standard_scaled_data = standard_scaler.fit_transform(data)
# print("\nStandard Scaled Data:")
# print(standard_scaled_data)

    def test_custom_vs_sklearn_min_max_scaler(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Custom MinMaxScaler
        custom_scaler = MinMaxScaler()
        custom_scaled_data = custom_scaler.fit_transform(data)
        
        # Sklearn MinMaxScaler
        sklearn_scaler = MMS()
        sklearn_scaled_data = sklearn_scaler.fit_transform(data)
        
        # Assert that all close
        self.assertTrue(np.allclose(custom_scaled_data, sklearn_scaled_data), "The custom MinMaxScaler results differ from sklearn's MinMaxScaler")





if __name__ == '__main__':
    unittest.main()

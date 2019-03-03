import unittest
import preproc
import numpy as np

class u_test(unittest.TestCase):

    def test_mean_center(self):
        test_img = np.array([[1,2,3],[1,2,3],[1,2,3]])
        result = preproc.mean_center(test_img)
        expected_result = np.array([[1,1,-1],[1,0,-1],[1,-1,-1]])
        self.assertEqual(result.all(), expected_result.all())
       
    def test_normalize_img(self):
        test_img = np.array([1,2,1],[1,2,4],[1,2,5])
        result = preproc.normalize_img(test_img)
        expected_result = np.array([0.25,0.50,0.25],[0.25,0.50,1],[0.25,0.5,1.25])
        self.assertEqual(result.all(), expected_result.all())

 
if __name__ == '__main__':
    unittest.main()

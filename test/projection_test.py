from amii_tf_nn.projection import l1_projection_to_simplex
import tensorflow as tf


class ProjectionTest(tf.test.TestCase):
    def test_l1_no_negative(self):
        with self.test_session():
            self.assertAllClose(
                l1_projection_to_simplex(tf.constant([2.0, 8.0, 0.0])).eval(),
                [0.2, 0.8, 0.0]
            )

    def test_l1_with_negative(self):
        with self.test_session():
            self.assertAllClose(
                l1_projection_to_simplex(tf.constant([2.0, 8.0, -5.0])).eval(),
                [0.2, 0.8, 0.0]
            )

    def test_l1_multiple_rows(self):
        patient = l1_projection_to_simplex(
            tf.transpose(
                tf.constant(
                    [
                        [2.0, 8.0, -5.0],
                        [9.5, 0.4, 0.1]
                    ]
                )
            )
        )
        with self.test_session():
            self.assertAllClose(
                tf.transpose(patient).eval(),
                [
                    [0.2, 0.8, 0.0],
                    [0.95, 0.04, 0.01]
                ]
            )

    def test_l1_multiple_rows_axis_1(self):
        patient = l1_projection_to_simplex(
            tf.constant(
                [
                    [2.0, 8.0, -5.0],
                    [9.5, 0.4, 0.1]
                ]
            ),
            row_normalize=True
        )
        with self.test_session():
            self.assertAllClose(
                patient.eval(),
                [
                    [0.2, 0.8, 0.0],
                    [0.95, 0.04, 0.01]
                ]
            )

if __name__ == '__main__':
    tf.test.main()

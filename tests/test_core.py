"""Behaviour tests for the quaternion rotation kernel.

Every expectation is written out as a plain result: a quaternion, a turned
vector, an axis with an angle, or the exception a caller must see.  Run them
from the project root:

    python3 -m unittest discover -s tests -v
"""

import math
import unittest

from quat.core import (
    QuatError,
    angle_between,
    as_quaternion,
    compose,
    conjugate,
    from_axis_angle,
    from_euler,
    identity,
    inverse,
    multiply,
    norm,
    normalize,
    rotate,
    slerp,
    to_axis_angle,
    to_euler,
)

#: cos and sin of a quarter turn.
HALF = math.sqrt(0.5)


class QuaternionKernelTest(unittest.TestCase):
    """Public behaviour of the quaternion kernels."""

    def assertQuatClose(self, got, want, delta=1e-9, msg=None):
        self.assertEqual(len(got), 4, msg)
        for index, (value, expect) in enumerate(zip(got, want)):
            self.assertAlmostEqual(
                value, expect, delta=delta, msg="%s entry %d" % (msg, index) if msg else "entry %d" % (index,)
            )

    def assertVectorClose(self, got, want, delta=1e-9, msg=None):
        self.assertEqual(len(got), 3, msg)
        for index, (value, expect) in enumerate(zip(got, want)):
            self.assertAlmostEqual(
                value, expect, delta=delta, msg="%s entry %d" % (msg, index) if msg else "entry %d" % (index,)
            )

    def test_multiply_matches_the_hamilton_product(self):
        i = (0.0, 1.0, 0.0, 0.0)
        j = (0.0, 0.0, 1.0, 0.0)
        k = (0.0, 0.0, 0.0, 1.0)
        self.assertQuatClose(multiply(i, j), k, msg="i j")
        self.assertQuatClose(multiply(j, k), i, msg="j k")
        self.assertQuatClose(multiply(k, i), j, msg="k i")
        self.assertQuatClose(multiply(j, i), (0.0, 0.0, 0.0, -1.0), msg="j i")
        self.assertQuatClose(multiply(i, i), (-1.0, 0.0, 0.0, 0.0), msg="i i")
        self.assertQuatClose(multiply((1.0, 2.0, 3.0, 4.0), identity()), (1.0, 2.0, 3.0, 4.0), msg="neutral")
        self.assertQuatClose(
            multiply((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
            (-60.0, 12.0, 30.0, 24.0),
            msg="hand worked product",
        )
        step = from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
        tilt = from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0)
        self.assertQuatClose(multiply(step, tilt), (0.5, 0.5, 0.5, 0.5), msg="yaw then pitch")
        self.assertQuatClose(multiply(tilt, step), (0.5, 0.5, -0.5, 0.5), msg="pitch then yaw")

    def test_conjugate_flips_the_vector_part(self):
        self.assertQuatClose(conjugate((1.0, 2.0, 3.0, 4.0)), (1.0, -2.0, -3.0, -4.0))
        self.assertQuatClose(conjugate(conjugate((1.0, 2.0, 3.0, 4.0))), (1.0, 2.0, 3.0, 4.0))
        quarter = from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
        self.assertQuatClose(conjugate(quarter), (quarter[0], -quarter[1], -quarter[2], -quarter[3]))
        self.assertQuatClose(multiply(quarter, conjugate(quarter)), identity())
        seventh = (0.5, 0.5, 0.5, 0.5)
        self.assertQuatClose(multiply(conjugate(seventh), seventh), identity())

    def test_inverse_undoes_a_rotation(self):
        turn = from_axis_angle((0.0, 0.0, 1.0), 0.9)
        self.assertQuatClose(multiply(turn, inverse(turn)), identity(), msg="unit round trip")
        self.assertQuatClose(multiply(inverse(turn), turn), identity(), msg="unit round trip back")
        self.assertQuatClose(inverse((2.0, 0.0, 0.0, 0.0)), (0.5, 0.0, 0.0, 0.0), msg="real scale")
        scaled = (1.0, 2.0, 2.0, 4.0)
        self.assertQuatClose(inverse(scaled), (0.04, -0.08, -0.08, -0.16), msg="scaled quaternion")
        self.assertQuatClose(multiply(scaled, inverse(scaled)), identity(), msg="scaled round trip")

    def test_normalize_gives_unit_quaternions(self):
        self.assertQuatClose(normalize((2.0, 0.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0))
        scaled = normalize((1.0, 2.0, 3.0, 4.0))
        self.assertAlmostEqual(norm(scaled), 1.0, places=9)
        length = math.sqrt(30.0)
        for index, want in enumerate((1.0, 2.0, 3.0, 4.0)):
            self.assertAlmostEqual(scaled[index] * length, want, places=9, msg="entry %d" % (index,))
        self.assertQuatClose(normalize((0.5, 0.5, 0.5, 0.5)), (0.5, 0.5, 0.5, 0.5))

    def test_rotation_turns_vectors_and_keeps_their_length(self):
        step = from_axis_angle((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0)
        self.assertAlmostEqual(norm(step), 1.0, places=9)
        self.assertVectorClose(rotate(step, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0), msg="cyclic turn")
        self.assertVectorClose(rotate(step, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0), msg="cyclic turn again")
        quarter = from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
        self.assertVectorClose(rotate(quarter, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0), msg="x to y")
        self.assertVectorClose(rotate(quarter, (0.0, 1.0, 0.0)), (-1.0, 0.0, 0.0), msg="y to -x")
        half = from_axis_angle((1.0, 0.0, 0.0), math.pi)
        self.assertVectorClose(rotate(half, (0.0, 1.0, 0.0)), (0.0, -1.0, 0.0), msg="half turn")
        source = (0.3, -0.5, 0.8)
        for axis, angle in (((1.0, 0.0, 0.0), 0.7), ((0.0, 1.0, 0.0), -1.1), ((1.0, 1.0, 1.0), 2.5)):
            turned = rotate(from_axis_angle(axis, angle), source)
            self.assertAlmostEqual(
                math.hypot(*turned), math.hypot(*source), places=9, msg="axis %r" % (axis,)
            )

    def test_compose_applies_the_second_rotation_first(self):
        step = from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
        tilt = from_axis_angle((1.0, 0.0, 0.0), math.pi / 2.0)
        source = (1.0, 0.0, 0.0)
        together = compose(tilt, step)
        self.assertVectorClose(rotate(together, source), (0.0, 0.0, 1.0), msg="roll after yaw")
        self.assertVectorClose(
            rotate(together, source), rotate(tilt, rotate(step, source)), msg="one step at a time"
        )
        reversed_pair = compose(step, tilt)
        self.assertVectorClose(rotate(reversed_pair, source), (0.0, 1.0, 0.0), msg="yaw after roll")
        self.assertVectorClose(rotate(reversed_pair, (0.0, 1.0, 0.0)), (0.0, 0.0, 1.0), msg="yaw after roll again")

    def test_axis_and_angle_read_back_from_a_rotation(self):
        cases = (
            ((1.0, 0.0, 0.0), 1.2),
            ((1.0, 1.0, 0.0), 2.0 * math.pi / 3.0),
            ((1.0, -2.0, 2.0), 0.4),
        )
        for axis, angle in cases:
            rotation = from_axis_angle(axis, angle)
            self.assertAlmostEqual(norm(rotation), 1.0, places=9, msg="unit for %r" % (axis,))
            unit_axis, read_angle = to_axis_angle(rotation)
            self.assertAlmostEqual(read_angle, angle, places=9, msg="angle for %r" % (axis,))
            self.assertAlmostEqual(math.hypot(*unit_axis), 1.0, places=9, msg="axis length for %r" % (axis,))
            length = math.hypot(*axis)
            for index in range(3):
                self.assertAlmostEqual(
                    unit_axis[index] * length, axis[index], places=9, msg="axis entry %d for %r" % (index, axis)
                )
            self.assertAlmostEqual(angle_between(identity(), rotation), angle, places=9, msg="span for %r" % (axis,))

    def test_slerp_takes_the_short_arc(self):
        axis = (0.0, 1.0, 0.0)
        start = identity()
        end = from_axis_angle(axis, 3.0 * math.pi / 2.0)
        self.assertQuatClose(slerp(start, end, 0.0), start, msg="start")
        self.assertQuatClose(slerp(start, end, 1.0), end, msg="end")
        self.assertQuatClose(
            slerp(start, end, 0.5), from_axis_angle(axis, -math.pi / 4.0), msg="half way round"
        )
        self.assertAlmostEqual(norm(slerp(start, end, 0.25)), 1.0, places=9)
        quarter = from_axis_angle(axis, math.pi / 2.0)
        self.assertQuatClose(slerp(start, quarter, 0.5), from_axis_angle(axis, math.pi / 4.0), msg="quarter way")
        self.assertQuatClose(slerp(quarter, quarter, 0.5), quarter, msg="same rotation")
        opposite = (-quarter[0], -quarter[1], -quarter[2], -quarter[3])
        self.assertQuatClose(slerp(quarter, opposite, 0.5), quarter, msg="same rotation the other sign")

    def test_euler_angles_round_trip_inside_the_range(self):
        cases = ((0.0, 0.0, 0.0), (0.3, -0.4, 0.5), (-1.0, 0.25, 2.0), (0.9, 1.4, -3.0))
        for roll, pitch, yaw in cases:
            rotation = from_euler(roll, pitch, yaw)
            self.assertAlmostEqual(norm(rotation), 1.0, places=9, msg="unit for %r" % ((roll, pitch, yaw),))
            read = to_euler(rotation)
            for got, want, name in zip(read, (roll, pitch, yaw), ("roll", "pitch", "yaw")):
                self.assertAlmostEqual(got, want, places=9, msg="%s for %r" % (name, (roll, pitch, yaw)))
        folded = to_euler(from_euler(0.0, 2.5, 0.0))
        self.assertLessEqual(abs(folded[1]), math.pi / 2.0)
        self.assertAlmostEqual(angle_between(from_euler(*folded), from_euler(0.0, 2.5, 0.0)), 0.0, places=9)
        self.assertQuatClose(from_euler(0.0, 0.0, math.pi / 2.0), (HALF, 0.0, 0.0, HALF), msg="yaw half turn")

    def test_rejects_malformed_values_and_zero_rotations(self):
        with self.assertRaises(QuatError):
            as_quaternion((1.0, 2.0, 3.0))
        with self.assertRaises(QuatError):
            as_quaternion((1.0, 2.0, 3.0, "4"))
        with self.assertRaises(QuatError):
            from_axis_angle((0.0, 0.0, 0.0), 1.0)
        with self.assertRaises(QuatError):
            normalize((0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(QuatError):
            inverse((0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(QuatError):
            to_euler((0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(QuatError):
            angle_between(identity(), (0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(QuatError):
            slerp(identity(), identity(), 1.5)
        with self.assertRaises(QuatError):
            rotate(identity(), (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()

"""Quaternion rotation kernels built on the standard library only.

A quaternion is the four-tuple (w, x, y, z) whose first entry is the scalar
part.  Unit quaternions stand for rotations: the Hamilton product of two of
them is the rotation that applies the right factor first, because

    (a b) v (a b)* = a (b v b*) a*,

the conjugate of a unit quaternion undoes it, and a vector is turned with
v' = q v q*.  On top of the product sit the axis and angle view, roll, pitch
and yaw angles and a spherical interpolation that always takes the short way
around.

Everything here is arithmetic on plain floats: no clock, no randomness, no I/O.
"""

import math

__all__ = [
    "EPS",
    "QuatError",
    "angle_between",
    "as_quaternion",
    "as_vector",
    "compose",
    "conjugate",
    "dot",
    "from_axis_angle",
    "from_euler",
    "identity",
    "inverse",
    "multiply",
    "norm",
    "normalize",
    "rotate",
    "slerp",
    "to_axis_angle",
    "to_euler",
]

#: A quaternion shorter than this counts as zero, and two rotations closer
#: than this count as the same way round.
EPS = 1e-12


class QuatError(ValueError):
    """Raised for malformed quaternions, vectors and parameters."""


def _number(value, what):
    """A float, or an error that names where the value came from."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuatError("%s must be a number, got %r" % (what, value))
    return float(value)


def _fraction(value, what):
    """A number inside the closed unit interval."""
    fraction = _number(value, what)
    if fraction < 0.0 or fraction > 1.0:
        raise QuatError("%s must lie between 0 and 1, got %r" % (what, value))
    return fraction


def as_quaternion(values):
    """Copy a length four sequence of numbers into a quaternion."""
    if not isinstance(values, (list, tuple)):
        raise QuatError("a quaternion must be a sequence, got %r" % (values,))
    if len(values) != 4:
        raise QuatError("a quaternion needs four components, got %r" % (values,))
    return tuple(_number(value, "a quaternion component") for value in values)


def as_vector(values, what="a vector"):
    """Copy a length three sequence of numbers into a vector."""
    if not isinstance(values, (list, tuple)):
        raise QuatError("%s must be a sequence, got %r" % (what, values))
    if len(values) != 3:
        raise QuatError("%s needs three components, got %r" % (what, values))
    return tuple(_number(value, "%s component" % what) for value in values)


def identity():
    """The quaternion of no rotation at all."""
    return (1.0, 0.0, 0.0, 0.0)


def dot(first, second):
    """The four dimensional inner product of two quaternions."""
    a = as_quaternion(first)
    b = as_quaternion(second)
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def norm(quaternion):
    """The length of a quaternion."""
    w, x, y, z = as_quaternion(quaternion)
    return math.hypot(w, x, y, z)


def normalize(quaternion, tol=EPS):
    """The unit quaternion that points the same way as the given one."""
    values = as_quaternion(quaternion)
    length = norm(values)
    if length <= tol:
        raise QuatError("a quaternion of length zero has no direction")
    scale = 1.0 / length
    return tuple(value * scale for value in values)


def conjugate(quaternion):
    """The quaternion with its vector part flipped: the inverse of a unit one."""
    w, x, y, z = as_quaternion(quaternion)
    return (w, -x, -y, -z)


def inverse(quaternion, tol=EPS):
    """The quaternion that multiplies back to the identity."""
    values = as_quaternion(quaternion)
    squared = dot(values, values)
    if squared <= tol:
        raise QuatError("a quaternion of length zero has no inverse")
    scale = 1.0 / squared
    return tuple(value * scale for value in conjugate(values))


def multiply(first, second):
    """The Hamilton product, which does not commute."""
    a = as_quaternion(first)
    b = as_quaternion(second)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate(quaternion, vector):
    """The vector turned by a unit quaternion, as q v q*."""
    values = as_quaternion(quaternion)
    x, y, z = as_vector(vector)
    pure = (0.0, x, y, z)
    turned = multiply(multiply(values, pure), conjugate(values))
    return (turned[1], turned[2], turned[3])


def compose(first, second):
    """The rotation that applies the second one and then the first."""
    return multiply(first, second)


def from_axis_angle(axis, angle):
    """The unit quaternion of a turn around an axis, in radians."""
    x, y, z = as_vector(axis, "an axis")
    radians = _number(angle, "an angle")
    length = math.sqrt(x * x + y * y + z * z)
    if length <= EPS:
        raise QuatError("an axis must not be the zero vector")
    scale = math.sin(radians / 2.0) / length
    return (math.cos(radians / 2.0), x * scale, y * scale, z * scale)


def to_axis_angle(quaternion, tol=EPS):
    """The unit axis and the angle between zero and pi of a rotation."""
    w, x, y, z = as_quaternion(quaternion)
    length = math.hypot(x, y, z)
    if length <= tol:
        return (1.0, 0.0, 0.0), 0.0
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
        length = math.hypot(x, y, z)
    angle = 2.0 * math.atan2(length, w)
    return (x / length, y / length, z / length), angle


def angle_between(first, second, tol=EPS):
    """The angle of the shortest turn from one orientation to the other."""
    a = as_quaternion(first)
    b = as_quaternion(second)
    lengths = norm(a) * norm(b)
    if lengths <= tol:
        raise QuatError("a quaternion of length zero has no orientation")
    cosine = dot(a, b) / lengths
    return 2.0 * math.acos(min(1.0, max(-1.0, abs(cosine))))


def from_euler(roll, pitch, yaw):
    """The quaternion of a roll about x, then a pitch about y, then a yaw
    about z, with all three angles in radians."""
    half_roll = _number(roll, "a roll angle") / 2.0
    half_pitch = _number(pitch, "a pitch angle") / 2.0
    half_yaw = _number(yaw, "a yaw angle") / 2.0
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def to_euler(quaternion, tol=EPS):
    """Roll, pitch and yaw of a rotation, with the pitch folded into the
    closed range from -pi/2 to pi/2 and the other two angles into the half
    open range from -pi to pi."""
    w, x, y, z = as_quaternion(quaternion)
    squared = dot((w, x, y, z), (w, x, y, z))
    if squared <= tol:
        raise QuatError("a quaternion of length zero has no angles")
    sine = 2.0 * (w * y - z * x) / squared
    pitch = math.asin(min(1.0, max(-1.0, sine)))
    roll = math.atan2(
        2.0 * (w * x + y * z) / squared,
        1.0 - 2.0 * (x * x + y * y) / squared,
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y) / squared,
        1.0 - 2.0 * (y * y + z * z) / squared,
    )
    return (roll, pitch, yaw)


def slerp(first, second, fraction):
    """The rotation a fraction of the short arc between two unit quaternions.

    The ends are exact, so a fraction of zero gives the first quaternion and a
    fraction of one gives the second one.
    """
    a = as_quaternion(first)
    b = as_quaternion(second)
    t = _fraction(fraction, "an interpolation parameter")
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    cosine = dot(a, b)
    if cosine < 0.0:
        b = tuple(-value for value in b)
        cosine = -cosine
    if cosine > 1.0 - EPS:
        blended = tuple(a[index] + (b[index] - a[index]) * t for index in range(4))
        return normalize(blended)
    theta = math.acos(min(1.0, cosine))
    sine = math.sin(theta)
    left = math.sin((1.0 - t) * theta) / sine
    right = math.sin(t * theta) / sine
    return tuple(a[index] * left + b[index] * right for index in range(4))

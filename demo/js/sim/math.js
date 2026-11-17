// Training conventions, ported from osl_deploy/math.py: scalar-first (wxyz)
// quaternions, world-frame SE(2) poses [x, y, yaw], and the scipy Rotation
// operations osl_deploy uses (quaternions normalized on construction,
// composition renormalized, no sign canonicalization). Every function writes
// into a caller-owned array and allocates nothing.

export const TWO_PI = 2 * Math.PI;

// numpy.rint: round half to even.
export function rint(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Python/numpy float modulo: the result takes the sign of the divisor.
export function pyMod(a, b) {
  let m = a % b;
  if (m !== 0 && (b < 0) !== (m < 0)) m += b;
  return m;
}

// scipy Rotation.from_quat(q, scalar_first=True) normalization, into out (wxyz).
export function quatNormalize(out, q, qo = 0) {
  const w = q[qo], x = q[qo + 1], y = q[qo + 2], z = q[qo + 3];
  const norm = Math.sqrt(x * x + y * y + z * z + w * w);
  out[0] = w / norm; out[1] = x / norm; out[2] = y / norm; out[3] = z / norm;
  return out;
}

const QA = new Float64Array(4), QB = new Float64Array(4), QR = new Float64Array(4);

// math.quat_mul: (rotation(a) * rotation(b)).as_quat(scalar_first=True).
// a at a[ao..ao+4], b at b[bo..bo+4]; out may alias neither input.
export function quatMul(out, a, b, ao = 0, bo = 0) {
  quatNormalize(QA, a, ao);
  quatNormalize(QB, b, bo);
  const pw = QA[0], px = QA[1], py = QA[2], pz = QA[3];
  const qw = QB[0], qx = QB[1], qy = QB[2], qz = QB[3];
  // scipy _compose_quat (xyzw storage): cross = p.xyz x q.xyz.
  const cx = py * qz - pz * qy;
  const cy = pz * qx - px * qz;
  const cz = px * qy - py * qx;
  QR[1] = pw * qx + qw * px + cx;
  QR[2] = pw * qy + qw * py + cy;
  QR[3] = pw * qz + qw * pz + cz;
  QR[0] = pw * qw - px * qx - py * qy - pz * qz;
  return quatNormalize(out, QR, 0);
}

// math.yaw_quat.
export function yawQuat(out, yaw) {
  out[0] = Math.cos(yaw / 2); out[1] = 0; out[2] = 0; out[3] = Math.sin(yaw / 2);
  return out;
}

// rotation(q).inv().apply(v): v rotated by the inverse of q (wxyz), into out.
export function rotateInverse(out, q, v, qo = 0, vo = 0, oo = 0) {
  quatNormalize(QA, q, qo);
  const w = QA[0], x = -QA[1], y = -QA[2], z = -QA[3];
  const x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
  const xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
  const m00 = x2 - y2 - z2 + w2, m01 = 2 * (xy - zw), m02 = 2 * (xz + yw);
  const m10 = 2 * (xy + zw), m11 = -x2 + y2 - z2 + w2, m12 = 2 * (yz - xw);
  const m20 = 2 * (xz - yw), m21 = 2 * (yz + xw), m22 = -x2 - y2 + z2 + w2;
  const v0 = v[vo], v1 = v[vo + 1], v2 = v[vo + 2];
  out[oo] = m00 * v0 + m01 * v1 + m02 * v2;
  out[oo + 1] = m10 * v0 + m11 * v1 + m12 * v2;
  out[oo + 2] = m20 * v0 + m21 * v1 + m22 * v2;
  return out;
}

// math.planar: [x, y, yaw] of a position and a wxyz quaternion.
export function planar(out, pos, quat, po = 0, qo = 0) {
  const w = quat[qo], x = quat[qo + 1], y = quat[qo + 2], z = quat[qo + 3];
  out[0] = pos[po];
  out[1] = pos[po + 1];
  out[2] = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return out;
}

// math.compose: SE(2) a then b. out may alias a or b.
export function compose(out, a, b, bo = 0) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]);
  const b0 = b[bo], b1 = b[bo + 1], b2 = b[bo + 2];
  const x = a[0] + c * b0 - s * b1;
  const y = a[1] + s * b0 + c * b1;
  out[0] = x; out[1] = y; out[2] = a[2] + b2;
  return out;
}

// math.inverse: SE(2) inverse. out may alias a.
export function inverse(out, a) {
  const c = Math.cos(a[2]), s = Math.sin(a[2]);
  const x = -c * a[0] - s * a[1];
  const y = s * a[0] - c * a[1];
  out[0] = x; out[1] = y; out[2] = -a[2];
  return out;
}

// Rotation.from_matrix(m).as_rotvec() for a row-major 3x3 rotation matrix
// m[mo..mo+9], written to out[oo..oo+3].
export function rotvecFromMatrix(out, m, mo = 0, oo = 0) {
  const m00 = m[mo], m01 = m[mo + 1], m02 = m[mo + 2];
  const m10 = m[mo + 3], m11 = m[mo + 4], m12 = m[mo + 5];
  const m20 = m[mo + 6], m21 = m[mo + 7], m22 = m[mo + 8];
  const trace = m00 + m11 + m22;
  // scipy from_matrix: the largest of the diagonal and the trace picks the
  // numerically safest formula (xyzw storage).
  let x, y, z, w, choice = 0, best = m00;
  if (m11 > best) { best = m11; choice = 1; }
  if (m22 > best) { best = m22; choice = 2; }
  if (trace > best) choice = 3;
  if (choice === 0) {
    x = 1 - trace + 2 * m00; y = m10 + m01; z = m20 + m02; w = m21 - m12;
  } else if (choice === 1) {
    y = 1 - trace + 2 * m11; z = m21 + m12; x = m01 + m10; w = m02 - m20;
  } else if (choice === 2) {
    z = 1 - trace + 2 * m22; x = m02 + m20; y = m12 + m21; w = m10 - m01;
  } else {
    x = m21 - m12; y = m02 - m20; z = m10 - m01; w = 1 + trace;
  }
  const norm = Math.sqrt(x * x + y * y + z * z + w * w);
  x /= norm; y /= norm; z /= norm; w /= norm;
  // as_rotvec: w >= 0, then angle-scaled vector part.
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const angle = 2 * Math.atan2(Math.sqrt(x * x + y * y + z * z), w);
  let scale;
  if (angle <= 1e-3) {
    const a2 = angle * angle;
    scale = 2 + a2 / 12 + (7 * a2 * a2) / 2880;
  } else {
    scale = angle / Math.sin(angle / 2);
  }
  out[oo] = scale * x; out[oo + 1] = scale * y; out[oo + 2] = scale * z;
  return out;
}

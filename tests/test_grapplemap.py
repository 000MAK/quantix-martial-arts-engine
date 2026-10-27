import unittest

import numpy as np

from martial_arts_engine.grapplemap import (
    BASE62, ENCODED, MIRROR_PERM, build_graph, decode_frame, parse, same_position,
)


def _encode(p: np.ndarray) -> str:
    q = p.copy()
    q[..., 0] += 2
    q[..., 2] += 2
    chars = []
    for v in np.round(q.reshape(-1) * 1000).astype(int):
        chars += [BASE62[v // 62], BASE62[v % 62]]
    s = "".join(chars)
    return "\n".join("    " + s[i:i + 69] for i in range(0, ENCODED, 69))


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.6, 0.6, (2, 23, 3))
    p[..., 1] = rng.uniform(0.0, 1.6, (2, 23))
    return p


def _rotate(p, theta, shift=(0.3, -0.2)):
    q = p.copy()
    c, s = np.cos(theta), np.sin(theta)
    x, z = q[..., 0].copy(), q[..., 2].copy()
    q[..., 0], q[..., 2] = c * x - s * z + shift[0], s * x + c * z + shift[1]
    return q


class GrappleMapTests(unittest.TestCase):
    def test_decode_round_trip(self):
        p = np.round(_frame(), 3)
        self.assertTrue(np.allclose(decode_frame(_encode(p)), p, atol=1e-9))

    def test_reorientations_are_the_same_position(self):
        p = _frame(1)
        self.assertTrue(same_position(p, _rotate(p, 1.1)))
        self.assertTrue(same_position(p, p[::-1]))                 # players swapped
        m = p[:, MIRROR_PERM].copy()
        m[..., 0] *= -1                                            # mirrored
        self.assertTrue(same_position(p, _rotate(m, -0.7)))

    def test_five_centimetres_is_a_different_position(self):
        p = _frame(2)
        q = p.copy()
        q[0, 14, 1] += 0.05                                        # one wrist moved 5 cm up
        self.assertFalse(same_position(p, q))

    def test_parse_and_graph(self):
        a, b = _frame(3), _rotate(_frame(3), 0.4)
        c = _frame(4)
        text = ("mount\ntags: mount top\n" + _encode(a) + "\n"
                "back take\nproperties: bidirectional\n" + _encode(b) + "\n" + _encode(c) + "\n")
        seqs = parse(text)
        self.assertEqual([len(s["frames"]) for s in seqs], [1, 2])
        self.assertEqual(seqs[0]["tags"], ["mount", "top"])
        nodes, edges = build_graph(seqs)
        self.assertEqual(len(nodes), 2)                            # end frame c becomes an unnamed node
        self.assertEqual((edges[0]["from"], edges[0]["to"]), (0, 1))
        self.assertTrue(edges[0]["bidirectional"])
        self.assertIsNone(nodes[1]["name"])


if __name__ == "__main__":
    unittest.main()

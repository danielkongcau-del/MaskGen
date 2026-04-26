from __future__ import annotations

import unittest

from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    build_token_vocabulary,
    decode_tokens_to_generator_target,
    encode_generator_target,
    tokens_to_ids,
)


class ParseGraphTokenizerTest(unittest.TestCase):
    def make_target(self) -> dict:
        return {
            "format": "maskgen_generator_target_v1",
            "target_type": "parse_graph",
            "size": [256, 256],
            "parse_graph": {
                "nodes": [
                    {
                        "id": "label_group_0",
                        "role": "label_group",
                        "label": 1,
                        "geometry_model": "none",
                        "children": ["face_0"],
                        "count": 1,
                    },
                    {
                        "id": "face_0",
                        "role": "semantic_face",
                        "label": 1,
                        "frame": {"origin": [128.0, 128.0], "scale": 64.0, "orientation": 0.0},
                        "geometry_model": "convex_atom_union",
                        "geometry": {"atom_ids": ["atom_0"]},
                        "atom_ids": ["atom_0"],
                    },
                    {
                        "id": "atom_0",
                        "role": "convex_atom",
                        "label": 1,
                        "parent_face": "face_0",
                        "frame": {"origin": [128.0, 128.0], "scale": 64.0, "orientation": 0.0},
                        "geometry_model": "convex_polygon",
                        "geometry": {
                            "outer_local": [[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]],
                            "type": "triangle",
                            "vertex_count": 3,
                            "area": 100.0,
                        },
                    },
                ],
                "relations": [
                    {"type": "label_group_contains", "parent": "label_group_0", "child": "face_0"},
                    {"type": "atom_part_of", "atom": "atom_0", "face": "face_0"},
                ],
                "residuals": [],
            },
            "metadata": {},
        }

    def test_encode_decode_structural_roundtrip(self) -> None:
        config = ParseGraphTokenizerConfig(coord_bins=32, area_bins=64, max_int=512)
        tokens = encode_generator_target(self.make_target(), config=config)
        self.assertEqual(tokens[0], "<BOS>")
        self.assertEqual(tokens[-1], "<EOS>")
        decoded = decode_tokens_to_generator_target(tokens, config=config)
        graph = decoded["parse_graph"]
        self.assertEqual(decoded["size"], [256, 256])
        self.assertEqual(sum(1 for node in graph["nodes"] if node["role"] == "semantic_face"), 1)
        self.assertEqual(sum(1 for node in graph["nodes"] if node["role"] == "convex_atom"), 1)
        self.assertTrue(any(relation["type"] == "atom_part_of" for relation in graph["relations"]))

    def test_vocab_covers_encoded_tokens(self) -> None:
        config = ParseGraphTokenizerConfig(coord_bins=32, area_bins=64, max_int=256)
        vocab = build_token_vocabulary(config)
        tokens = encode_generator_target(self.make_target(), config=config)
        ids = tokens_to_ids(tokens, vocab)
        self.assertEqual(len(ids), len(tokens))
        self.assertNotIn(vocab["<UNK>"], ids)


if __name__ == "__main__":
    unittest.main()

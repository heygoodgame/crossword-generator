# Custom ipuz Extensions

This project uses custom extension fields in the ipuz format, namespaced under `hgg.*` (reverse domain name notation per the ipuz spec).

These extensions are non-standard. Other ipuz readers will ignore them, but the data is preserved through ipuz read/write round-trips.

---

## `hgg.references` — Clue Cross-References

Links related clues in themed puzzles, connecting revealer clues to their theme entries (and vice versa). A consuming app can use this to highlight related clues when a solver focuses on a theme entry or the revealer.

### Schema

```json
{
  "hgg.references": [
    {
      "clue": [<number>, "<direction>"],
      "role": "revealer" | "theme_entry",
      "references": [[<number>, "<direction>"], ...]
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `clue` | `[int, string]` | The clue this entry describes, as `[number, "Across" or "Down"]` |
| `role` | `string` | Either `"revealer"` or `"theme_entry"` |
| `references` | `array` | List of related clues, each as `[number, direction]` |

### Example

For a "Things that are golden" puzzle where 11-Across (GOLDMINES) is the revealer and 6-Across (ORE), 13-Across (COIN), 19-Down (MEDAL), 22-Down (BAR) are theme entries:

```json
{
  "hgg.references": [
    {
      "clue": [11, "Across"],
      "role": "revealer",
      "references": [[6, "Across"], [13, "Across"], [19, "Down"], [22, "Down"]]
    },
    {
      "clue": [6, "Across"],
      "role": "theme_entry",
      "references": [[11, "Across"]]
    },
    {
      "clue": [13, "Across"],
      "role": "theme_entry",
      "references": [[11, "Across"]]
    },
    {
      "clue": [19, "Down"],
      "role": "theme_entry",
      "references": [[11, "Across"]]
    },
    {
      "clue": [22, "Down"],
      "role": "theme_entry",
      "references": [[11, "Across"]]
    }
  ]
}
```

### Auto-Population

The ipuz exporter builds this field automatically when the `PuzzleEnvelope` has a `ThemeConcept` with a revealer and seed entries. It matches the revealer and seeds against the clue list by answer word. Seeds that weren't placed in the grid are excluded.

The field is omitted entirely for puzzles without a theme (e.g., mini puzzles).

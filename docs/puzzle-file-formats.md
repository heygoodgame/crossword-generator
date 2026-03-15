# Crossword Puzzle File Format Specifications

## .PUZ Format (AcrossLite Binary Format)

The .puz format is a binary file format created by Literate Software Systems for their AcrossLite crossword software. It has been in use since 1996 and became the de facto standard for crossword puzzle interchange. The format was never officially documented by its creators — the specification below is the result of community reverse-engineering efforts.

**Key characteristics:** Binary format, little-endian byte order, ISO-8859-1 / Windows-1252 text encoding, CRC-16 checksums, optional solution scrambling.

### File Structure Overview

The file is laid out in four sequential sections:

1. A 52-byte fixed header
2. The puzzle solution and current player state grids
3. NUL-terminated variable-length strings (metadata and clues)
4. Optional extra sections (rebuses, circles, timer data)

### Header (52 bytes)

All multi-byte integers are little-endian ("short" = 2-byte LE integer).

| Field | Offset | Length | Type | Description |
|---|---|---|---|---|
| Overall Checksum | 0x00 | 2 | short | File-wide checksum |
| File Magic | 0x02 | 12 | string | NUL-terminated constant: `ACROSS&DOWN\0` (hex: `4143 524f 5353 2644 4f57 4e00`) |
| CIB Checksum | 0x0E | 2 | short | Checksum of 8 header bytes starting at 0x2C |
| Masked Low Checksums | 0x10 | 4 | bytes | Four checksums XOR-masked against magic string |
| Masked High Checksums | 0x14 | 4 | bytes | Four checksums XOR-masked against magic string |
| Version String | 0x18 | 4 | string | e.g. `"1.2\0"` |
| Reserved | 0x1C | 2 | — | Often uninitialized memory |
| Scrambled Checksum | 0x1E | 2 | short | Checksum of real solution if scrambled; otherwise 0x0000 |
| *(gap: 0x20–0x2B)* | — | — | — | Unknown / reserved bytes |
| Width | 0x2C | 1 | byte | Board width in cells |
| Height | 0x2D | 1 | byte | Board height in cells |
| Number of Clues | 0x2E | 2 | short | Total clue count |
| Unknown Bitmask | 0x30 | 2 | short | Purpose unknown |
| Scrambled Tag | 0x32 | 2 | short | 0 = unscrambled; nonzero (often 4) = scrambled |

### Puzzle Layout and State

Immediately after the header (offset 0x34):

**Solution grid:** `width × height` bytes of ASCII. Each byte is one cell, scanned left-to-right, top-to-bottom. Black cells are represented by `.` (period). Letters are uppercase ASCII.

**Player state grid:** `width × height` bytes. Same layout as solution. Empty (unfilled) cells are `-` (dash). Black cells remain `.`.

Example for a 3×3 grid:
```
Solution:  C A T . . A . . R  →  "CAT..A..R"
State:     - - - . . - . . -  →  "---..-..-"
```

### Strings Section

Immediately following the two grids. All strings are NUL-terminated and encoded in ISO-8859-1 (effectively Windows-1252 in practice). They appear in this fixed order:

| # | Field | Notes |
|---|---|---|
| 1 | Title | Puzzle title. Some NYT puzzles embed notes here after a space, prefixed with "NOTE:" |
| 2 | Author | Constructor name |
| 3 | Copyright | Copyright notice |
| 4–(3+n) | Clues | One string per clue, in numerical order. When two clues share a number, Across comes before Down. |
| Last | Notes | Optional puzzle notes |

Even empty strings still have their trailing NUL byte present in the file.

### Clue Assignment

Clue numbers are not stored in the file — they are derived from the grid shape using standard crossword numbering rules:

1. Scan cells left-to-right, top-to-bottom.
2. A cell gets an **Across** number if it has no non-black cell to its left AND there is room for at least a 2-cell word to the right.
3. A cell gets a **Down** number if it has no non-black cell above AND there is room for at least a 2-cell word below.
4. If a cell qualifies for either Across or Down (or both), it is assigned the next sequential number.
5. Clues in the strings section are ordered by number; Across before Down when they share a number.

### Checksums

The checksum algorithm is a CRC-16 variant:

```c
unsigned short cksum_region(unsigned char *base, int len, unsigned short cksum) {
    for (int i = 0; i < len; i++) {
        if (cksum & 0x0001)
            cksum = (cksum >> 1) + 0x8000;
        else
            cksum = cksum >> 1;
        cksum += *(base + i);
    }
    return cksum;
}
```

**CIB Checksum:** `cksum_region(data + 0x2C, 8, 0)`

**Overall Checksum (bytes 0x00–0x01):** Chains CIB checksum → solution → state → title (if non-empty, includes NUL) → author (includes NUL) → copyright (includes NUL) → each clue (without NUL) → notes (includes NUL).

**Masked Checksums (bytes 0x10–0x17):** Four independent checksums (CIB, solution, grid/state, partial-board) are XOR-masked against the ASCII string `"ICHEATED"`:

```c
file[0x10] = 0x49 ^ (c_cib  & 0xFF);    // 'I'
file[0x11] = 0x43 ^ (c_sol  & 0xFF);    // 'C'
file[0x12] = 0x48 ^ (c_grid & 0xFF);    // 'H'
file[0x13] = 0x45 ^ (c_part & 0xFF);    // 'E'
file[0x14] = 0x41 ^ ((c_cib  >> 8));    // 'A'
file[0x15] = 0x54 ^ ((c_sol  >> 8));    // 'T'
file[0x16] = 0x45 ^ ((c_grid >> 8));    // 'E'
file[0x17] = 0x44 ^ ((c_part >> 8));    // 'D'
```

### Optional Extra Sections

Extra sections appear after the strings and share a common envelope format:

| Field | Length | Description |
|---|---|---|
| Title | 4 bytes | ASCII section name (e.g. `GRBS`) |
| Length | 2 bytes (short) | Length of data section |
| Checksum | 2 bytes (short) | Checksum of data |
| Data | *Length* bytes | Section-specific payload |
| NUL | 1 byte | Trailing 0x00 |

Known sections (typically appear in this order):

**GRBS (Grid Rebus):** One byte per cell. `0` = not a rebus. `1+n` = rebus, where `n` is the key into the RTBL table.

**RTBL (Rebus Table):** ASCII string mapping keys to multi-character solutions. Format: `" n:SOLUTION;"` for each entry. The key `n` is always two characters wide (space-padded). Example: `" 0:HEART; 1:DIAMOND; 2:CLUB; 3:SPADE;"`

**GEXT (Grid Extras):** One byte per cell, bitmask for cell styling:
- `0x10` — cell was previously marked incorrect
- `0x20` — cell is currently marked incorrect
- `0x40` — cell was revealed by the player
- `0x80` — cell has a circle drawn around it

**LTIM (Timer):** ASCII string `"seconds,running"` where `seconds` is elapsed time and `running` is `0` (stopped) or `1` (running).

**RUSR (User Rebus):** Contains user-entered rebus values (for partially-solved puzzles). Currently undocumented.

### Solution Scrambling

Scrambled puzzles encrypt the solution grid using a 4-digit numeric key. The scrambled tag at offset 0x32 is nonzero, and the scrambled checksum at 0x1E allows verification after unscrambling. The scrambling algorithm was reverse-engineered by Brian Raiter. The clues and grid layout remain unencrypted — only the solution letters are obscured.

### Limitations

- ASCII-only cell values (one byte per cell in the base grid; multi-character entries require GRBS/RTBL extensions)
- No native support for barred grids, colored cells, or non-rectangular shapes
- No support for non-Latin character sets
- Gray squares are unsupported
- Limited metadata fields
- IP concerns: the format's owner has historically asserted IP rights

---

## .IPUZ Format (Open Puzzle Format)

The ipuz format is an openly-documented, JSON-based puzzle format created by Puzzazz, Inc. and released under a Creative Commons BY-ND 3.0 license. It was designed to address the limitations of .puz and support a wide variety of puzzle types.

**Key characteristics:** JSON-based (originally also supported JSONP, now deprecated for security), human-readable, extensible via PuzzleKinds and namespaced extensions, supports crosswords, sudoku, word search, acrostics, fill-in puzzles, block puzzles, and more.

**Spec versions:** v1 (original), v2 (current). The specification is maintained at http://ipuz.org.

### File Basics

An ipuz file is a `.ipuz` file containing a JSON object. The MIME type is not formally registered, but the file extension `.ipuz` is standard (apps may add a secondary suffix, e.g., `.ipuz.myapp`).

### Minimal Crossword Example

```json
{
    "version": "http://ipuz.org/v2",
    "kind": ["http://ipuz.org/crossword#1"],
    "dimensions": { "width": 3, "height": 3 },
    "puzzle": [
        [{ "cell": 1, "style": { "shapebg": "circle" } }, 2, "#"],
        [3, { "style": { "shapebg": "circle" } }, 4],
        [null, 5, { "style": { "shapebg": "circle" } }]
    ],
    "solution": [
        ["C", "A", "#"],
        ["B", "O", "T"],
        [null, "L", "O"]
    ],
    "clues": {
        "Across": [[1, "OR neighbor"], [3, "Droid"], [5, "Behold!"]],
        "Down": [[1, "Trucker's radio"], [2, "MSN competitor"], [4, "A preposition"]]
    }
}
```

### Common Fields (All Puzzle Types)

**Required fields:**

| Field | Type | Description |
|---|---|---|
| `version` | string | Must be `"http://ipuz.org/v1"` or `"http://ipuz.org/v2"` |
| `kind` | array of strings | PuzzleKind URIs (at least one required) |

**Optional metadata fields:**

| Field | Type | Description |
|---|---|---|
| `copyright` | string | Copyright information |
| `publisher` | HTML string | Publisher name/reference |
| `publication` | HTML string | Bibliographic reference |
| `url` | string | Permanent URL for the puzzle |
| `uniqueid` | string | Globally unique identifier |
| `title` | HTML string | Puzzle title |
| `intro` | HTML string | Text displayed above puzzle |
| `explanation` | HTML string | Text shown after successful solve |
| `annotation` | string | Non-displayed annotation |
| `author` | HTML string | Puzzle author |
| `editor` | HTML string | Puzzle editor |
| `date` | string | Date in `mm/dd/yyyy` format |
| `notes` | HTML string | Notes about the puzzle |
| `difficulty` | HTML string | Informational difficulty rating |
| `origin` | string | Program-specific info from authoring tool |
| `block` | string | Character representing a block (default: `"#"`) |
| `empty` | string/int | Value representing empty cell (default: `0`) |
| `styles` | object | Named StyleSpec definitions for the puzzle |

**Solution verification:**

| Field | Type | Description |
|---|---|---|
| `checksum` | array | `["salt", "SHA1_hash", ...]` — SHA1 hash of (correct solution + salt). Multiple hashes allowed for puzzles with multiple valid solutions. |

**Saved state:**

| Field | Type | Description |
|---|---|---|
| `saved` | varies | Partially-solved state; format depends on puzzle type |

### PuzzleKinds

Each PuzzleKind is identified by a URI. The URI must be a valid URL owned by its author. A version fragment identifies the schema version (e.g., `#1`). Use the lowest version compatible with the features in use.

**Built-in PuzzleKinds:**

| URI | Description |
|---|---|
| `http://ipuz.org/crossword` | Crossword puzzles (clued grid entries) |
| `http://ipuz.org/crossword/crypticcrossword` | Cryptic crosswords |
| `http://ipuz.org/crossword/arrowword` | Arrow/Pencil Pointer puzzles |
| `http://ipuz.org/crossword/logiccrossword` | Logic crosswords (unnumbered grid, alphabetical clues) |
| `http://ipuz.org/crossword/diagramless` | Diagramless crosswords |
| `http://ipuz.org/sudoku` | Standard sudoku |
| `http://ipuz.org/sudoku/wordoku` | Sudoku with letters |
| `http://ipuz.org/sudoku/latinsquare` | Latin squares (no boxes) |
| `http://ipuz.org/sudoku/diagonalsudoku` | Diagonal constraint sudoku |
| `http://ipuz.org/sudoku/hypersudoku` | Hyper sudoku (extra inner squares) |
| `http://ipuz.org/sudoku/jigsawsudoku` | Irregular-shaped regions |
| `http://ipuz.org/sudoku/calcudoku` | KenKen / Calcudoku |
| `http://ipuz.org/sudoku/killersudoku` | Killer sudoku |
| `http://ipuz.org/fill` | Fill-in puzzles |
| `http://ipuz.org/acrostic` | Acrostic puzzles |
| `http://ipuz.org/block` | Block/sliding puzzles |
| `http://ipuz.org/answer` | Question and answer puzzles |
| `http://ipuz.org/wordsearch` | Word search puzzles |

Custom PuzzleKinds can be created under domains you control.

### Data Types

**LabeledCell** — Represents a cell in the puzzle grid. Can be any of:
- An integer: the cell's clue number (0 = empty cell)
- `"#"` (or the value of `block`): a black/block cell
- `null`: an omitted cell (used for shaped puzzles)
- An object with detailed properties:
  ```json
  {
      "cell": 1,
      "value": "A",
      "style": { "shapebg": "circle" }
  }
  ```
  Where `cell` is the clue number, `value` is a pre-filled letter, and `style` is a StyleSpec.

**CrosswordValue** — A cell's solution or saved value:
- A string: the letter(s) in the cell
- `"#"` (or block value): black cell
- `null`: omitted cell
- An object: `{ "value": "AB", "style": {...} }` for styled solution cells

**StyleSpec** — Either a string (referencing a named style) or an object:
```json
{
    "shapebg": "circle",
    "highlight": true,
    "named": false,
    "border": 1,
    "barred": "T",
    "dotted": "R",
    "dashed": "B",
    "lessthan": "R",
    "greaterthan": "B",
    "equal": "L",
    "image": "url",
    "color": "#FF0000",
    "colortext": "#0000FF",
    "imagebg": "url"
}
```

Key style properties:
- `shapebg`: background shape (`"circle"` is the most common)
- `highlight`: boolean, highlights the cell
- `barred`: string of sides with thick bars (`"T"`, `"B"`, `"L"`, `"R"` or combinations like `"TL"`)
- `color` / `colortext`: hex color for cell background / text
- `border`: border thickness

**Clue** — Can be any of:
- `[number, "clue text"]` — standard clue
- `[number, "clue text", "enumeration"]` — clue with word-length pattern
- An object for more complex clues:
  ```json
  {
      "number": 1,
      "clue": "Clue text",
      "enumeration": "(4,3)",
      "answer": "WORD",
      "highlight": true,
      "location": [0, 0],
      "direction": "Across"
  }
  ```

**GroupSpec** — Defines an arbitrarily-shaped region:
```json
{
    "cells": [[0,0], [0,1], [0,2]],
    "style": { "highlight": true },
    "rect": [0, 0, 2, 2]
}
```
Can use `cells` (list of coordinates) or `rect` (`[col1, row1, col2, row2]`). Coordinates are 0-based.

**Dimension** — Grid dimensions object:
```json
{ "width": 15, "height": 15 }
```

**CalcSpec** — For Calcudoku/KenKen cages:
```json
{
    "value": 12,
    "operator": "+",
    "cells": [[0,0], [0,1], [1,0]]
}
```

### Crossword-Specific Fields

| Field | Type | Description |
|---|---|---|
| `dimensions` | Dimension | **Required.** Grid width and height |
| `puzzle` | 2D array of LabeledCell | **Required.** The puzzle grid |
| `solution` | 2D array of CrosswordValue | Correct solution |
| `saved` | 2D array of CrosswordValue | Current solve state |
| `clues` | object | Clue sets keyed by direction (e.g. `"Across"`, `"Down"`) |
| `zones` | array of GroupSpec | Arbitrarily-shaped overlay entries |
| `showenumerations` | boolean | Show word-length patterns with clues |
| `clueplacement` | string | `"before"`, `"after"`, `"blocks"`, or `null` (auto) |
| `answer` | string | Final meta-answer to the puzzle |
| `answers` | array of strings | Multiple final answers |
| `enumeration` | string | Word pattern for the final answer |
| `misses` | object | `{ "wrong_answer": "hint" }` — hints for incorrect submissions |

### Sudoku-Specific Fields

| Field | Type | Description |
|---|---|---|
| `charset` | string | Characters used (e.g. `"123456789"`) |
| `displaycharset` | boolean | Whether to show the character set |
| `boxes` | boolean | Whether to divide grid into boxes |
| `showoperators` | boolean | Show calculation operators (for Calcudoku) |
| `cageborder` | string | `"thick"` or `"dashed"` for cage borders |
| `puzzle` | 2D array of SudokuGiven | Given values |
| `solution` | 2D array of SudokuValue | Correct solution |
| `zones` | array of GroupSpec | Extra constrained regions |
| `cages` | array of CalcSpec | Mathematical constraint regions |

### Extensibility

Extensions use reverse domain name notation as a namespace prefix:

```json
{
    "version": "http://ipuz.org/v2",
    "kind": ["http://ipuz.org/crossword#1"],
    "com.example.myapp.theme": "holiday",
    "com.example.myapp.rating": 4,
    "volatile": { "com.example.myapp": ["com.example.myapp.rating"] }
}
```

Fields with a `.` in the name are extension fields. The `volatile` field declares which extensions should be removed when the puzzle is modified.

### Conventions

- Numbers are interchangeable with strings containing numbers (`1` and `"1"` are equivalent)
- `0` (or `"0"`) represents an empty cell by default; override with the `empty` field
- `"#"` represents a block by default; override with the `block` field
- `null` in a dictionary means the value is unspecified
- `null` as `true/false` means `false`
- HTML is allowed in many string fields but should be minimal
- Special characters `&`, `<`, `>` must be encoded as HTML entities
- The `"` character should be escaped as `\"` in JSON strings (not `&quot;`)

### Solution Checksum Format

For crosswords and sudoku, the solution string for checksumming is formed by concatenating all cell values in reading order (top-to-bottom, left-to-right), uppercased. For word search, concatenate all found entries in alphabetical order.

---

## Format Comparison

| Feature | .puz | .ipuz |
|---|---|---|
| **Encoding** | Binary | JSON (text) |
| **License** | Proprietary / reverse-engineered | CC BY-ND 3.0 |
| **Character encoding** | ISO-8859-1 / Windows-1252 | Unicode (UTF-8) |
| **Puzzle types** | Crosswords only | Crosswords, sudoku, word search, acrostics, etc. |
| **Rebus support** | Via GRBS/RTBL extensions | Native (multi-char cell values) |
| **Circled cells** | Via GEXT extension bitmask | Native (`"shapebg": "circle"`) |
| **Barred grids** | Not supported | Native (`"barred"` style) |
| **Cell coloring** | Not supported | Native (`"color"` style) |
| **Shaped (non-rectangular)** | Not supported | Native (`null` cells) |
| **Solution scrambling** | Built-in 4-digit key encryption | Not built-in; supports SHA1 checksum verification |
| **Timer data** | Via LTIM extension | Not in spec (use extensions) |
| **Extensibility** | Limited (fixed extra section types) | Namespaced extension fields |
| **Human-readable** | No | Yes |
| **File size** | Compact | Larger (JSON overhead) |
| **Ecosystem support** | Very broad (legacy) | Growing |
| **Max grid size** | 255×255 (byte width/height) | Unlimited |

---

## Reference Links

- .puz format spec (community): https://code.google.com/archive/p/puz/wikis/FileFormat.wiki
- .puz format gist (detailed): https://gist.github.com/sliminality/dab21fa834eae0a70193c7cd69c356d5
- .puz with extra sections: https://github.com/ajhyndman/puz/blob/main/PUZ%20File%20Format.md
- ipuz official spec: http://ipuz.org
- ipuz v1 archived spec: https://www.puzzazz.com/ipuz/v1
- ipuz Python library: https://pypi.org/project/ipuz/
- libipuz (C library + extensions): https://libipuz.org
- puzpy (Python .puz library): https://github.com/alexdej/puzpy
- [Custom ipuz extensions used by this project](ipuz-extensions.md)

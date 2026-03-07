package renderer

import (
	"encoding/json"
	"strings"

	"github.com/ahboujelben/go-crossword/modules/crossword"
)

// jsonOutput is the structured JSON output of a crossword.
type jsonOutput struct {
	Rows        int           `json:"rows"`
	Cols        int           `json:"cols"`
	Seed        int64         `json:"seed"`
	Grid        [][]string    `json:"grid"`
	WordsAcross []jsonWordRef `json:"words_across"`
	WordsDown   []jsonWordRef `json:"words_down"`
}

// jsonWordRef describes a word entry and its position.
type jsonWordRef struct {
	Word string `json:"word"`
	Row  int    `json:"row"`
	Col  int    `json:"col"`
}

// RenderJSON produces a JSON string from a solved crossword.
func RenderJSON(c *crossword.Crossword, seed int64) (string, error) {
	rows := c.Rows()
	cols := c.Columns()

	// Build the 2D grid
	grid := make([][]string, rows)
	for r := 0; r < rows; r++ {
		grid[r] = make([]string, cols)
	}

	for letter := crossword.CrosswordLetter(c); letter != nil; letter = letter.Next() {
		r := letter.Row()
		col := letter.Column()
		if letter.IsBlank() {
			grid[r][col] = "."
		} else {
			grid[r][col] = strings.ToUpper(string(letter.GetValue()))
		}
	}

	// Extract across words (row words)
	var wordsAcross []jsonWordRef
	for w := crossword.RowWord(c); w != nil; w = w.Next() {
		word := wordString(c, w.Row(), w.Column(), cols, true)
		if word != "" {
			wordsAcross = append(wordsAcross, jsonWordRef{
				Word: word,
				Row:  w.Row(),
				Col:  w.Column(),
			})
		}
	}

	// Extract down words (column words)
	var wordsDown []jsonWordRef
	for w := crossword.ColumnWord(c); w != nil; w = w.Next() {
		word := wordString(c, w.Row(), w.Column(), rows, false)
		if word != "" {
			wordsDown = append(wordsDown, jsonWordRef{
				Word: word,
				Row:  w.Row(),
				Col:  w.Column(),
			})
		}
	}

	output := jsonOutput{
		Rows:        rows,
		Cols:        cols,
		Seed:        seed,
		Grid:        grid,
		WordsAcross: wordsAcross,
		WordsDown:   wordsDown,
	}

	data, err := json.Marshal(output)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// wordString reads consecutive letters starting at (row, col) in the given direction.
func wordString(c *crossword.Crossword, row, col, limit int, across bool) string {
	var sb strings.Builder
	for i := 0; i < limit; i++ {
		var r, co int
		if across {
			r = row
			co = col + i
			if co >= c.Columns() {
				break
			}
		} else {
			r = row + i
			co = col
			if r >= c.Rows() {
				break
			}
		}
		letter := crossword.CrosswordLetterAt(c, r, co)
		if letter == nil || letter.IsBlank() {
			break
		}
		sb.WriteByte(letter.GetValue() + 'A' - 'a')
	}
	return sb.String()
}

package renderer

import (
	"github.com/ahboujelben/go-crossword/modules/crossword"
)

type CompactRenderer struct {
}

func NewCompactRenderer() CompactRenderer {
	return CompactRenderer{}
}

func (f CompactRenderer) RenderCrossword(c *crossword.Crossword, solved bool) string {
	var result string
	for letter := range getFormattedLetters(c, solved) {
		result += letter
	}

	return result
}

func getFormattedLetters(c *crossword.Crossword, solved bool) chan string {
	ch := make(chan string)
	go func() {
		for letter := crossword.CrosswordLetter(c); letter != nil; letter = letter.Next() {
			switch {
			case letter.IsBlank():
				ch <- "█ "
			case letter.IsEmpty() || !solved:
				ch <- ". "
			default:
				ch <- string(letter.GetValue()+'A'-'a') + " "
			}
			if letter.Column() == c.Columns()-1 && letter.Row() != c.Rows()-1 {
				ch <- "\n"
			}
		}
		close(ch)
	}()
	return ch
}

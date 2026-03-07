package crossword

import (
	"fmt"
)

type WordRef struct {
	pos       int
	length    int
	direction wordDirection

	crossword *Crossword
}

type wordDirection int

const (
	horizontal wordDirection = iota
	vertical
)

func Word(c *Crossword) *WordRef {
	return rowWord(0, c).WordRef
}

func (w *WordRef) Next() *WordRef {
	if w.direction == horizontal {
		rowWord := rowWord(w.pos, w.crossword)
		next := rowWord.Next()
		if next != nil {
			return next.WordRef
		}
		firstColumnWord := columnWord(0, w.crossword)
		if firstColumnWord != nil {
			return firstColumnWord.WordRef
		}
		return nil
	}
	columnWord := columnWord(w.pos, w.crossword)
	next := columnWord.Next()
	if next != nil {
		return next.WordRef
	}
	return nil
}

func (w *WordRef) GetValue() []byte {
	word := []byte{}
	for letter := WordLetter(w); letter != nil; letter = letter.Next() {
		word = append(word, letter.GetValue())
	}
	return word
}

func (w *WordRef) SetValue(value []byte) {
	if len(value) != w.length {
		panic(fmt.Sprintf("expected length %d but got %d", w.length, len(value)))
	}
	word := WordLetter(w)
	for index := range value {
		word.SetValue(value[index])
		word = word.Next()
	}
}

func (w *WordRef) IsFilled() bool {
	for letter := WordLetter(w); letter != nil; letter = letter.Next() {
		if letter.IsEmpty() {
			return false
		}
	}
	return true
}

func (w *WordRef) GetPos() int {
	return w.pos
}

func (w *WordRef) GetLength() int {
	return w.length
}

func (w *WordRef) GetDirection() wordDirection {
	return w.direction
}

func (w wordDirection) String() string {
	if w == horizontal {
		return "horizontal"
	}
	return "vertical"
}

type RowWordRef struct {
	*WordRef
}

func RowWord(c *Crossword) *RowWordRef {
	return rowWord(0, c)
}

func rowWord(pos int, c *Crossword) *RowWordRef {
	wordStart := -1
	wordLength := 0
	i := pos
	for i < len(c.data) {
		if c.data[i] != '.' {
			if wordStart == -1 {
				wordStart = i
			}
			wordLength++
			if (i+1)%c.columns == 0 || c.data[i+1] == '.' {
				if wordLength > 1 {
					return &RowWordRef{
						&WordRef{
							pos:       wordStart,
							length:    wordLength,
							direction: horizontal,
							crossword: c,
						},
					}
				}
				wordStart = -1
				wordLength = 0
			}
		}
		i++
	}
	return nil
}

func (w *RowWordRef) Row() int {
	return w.pos / w.crossword.columns
}

func (w *RowWordRef) Column() int {
	return w.pos % w.crossword.columns
}

func (w *RowWordRef) Next() *RowWordRef {
	return rowWord(w.pos+(w.length-1), w.crossword)
}

type ColumnWordRef struct {
	*WordRef
}

func ColumnWord(c *Crossword) *ColumnWordRef {
	return columnWord(0, c)
}

func columnWord(pos int, c *Crossword) *ColumnWordRef {
	wordStart := -1
	wordLength := 0
	i := pos
	for i < len(c.data) {
		if c.data[i] != '.' {
			if wordStart == -1 {
				wordStart = i
			}
			wordLength++
			if i+c.columns >= len(c.data) || c.data[i+c.columns] == '.' {
				if wordLength > 1 {
					return &ColumnWordRef{
						&WordRef{
							pos:       wordStart,
							length:    wordLength,
							direction: vertical,
							crossword: c,
						},
					}
				}
				wordStart = -1
				wordLength = 0
			}
		}
		if i == len(c.data)-1 {
			break
		} else if i+c.columns >= len(c.data) {
			i = i%c.columns + 1
		} else {
			i += c.columns
		}
	}
	return nil
}

func (w *ColumnWordRef) Column() int {
	return w.pos % w.crossword.columns
}

func (w *ColumnWordRef) Row() int {
	return w.pos / w.crossword.columns
}

func (w *ColumnWordRef) Next() *ColumnWordRef {
	return columnWord(w.pos+(w.length-1)*w.crossword.columns, w.crossword)
}

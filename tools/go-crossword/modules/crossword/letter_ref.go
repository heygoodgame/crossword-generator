package crossword

type LetterRef struct {
	pos       int
	crossword *Crossword
}

func (l *LetterRef) GetValue() byte {
	return l.crossword.data[l.pos]
}

func (l *LetterRef) SetValue(value byte) {
	l.crossword.data[l.pos] = value
}

func (l *LetterRef) IsEmpty() bool {
	return l.crossword.data[l.pos] == 0
}

func (l *LetterRef) IsBlank() bool {
	return l.crossword.data[l.pos] == Blank
}

func (l *LetterRef) Row() int {
	return l.pos / l.crossword.columns
}

func (l *LetterRef) Column() int {
	return l.pos % l.crossword.columns
}

type CrosswordLetterRef struct {
	LetterRef
}

func CrosswordLetter(c *Crossword) *CrosswordLetterRef {
	if len(c.data) == 0 {
		return nil
	}

	return &CrosswordLetterRef{
		LetterRef: LetterRef{
			crossword: c,
		},
	}
}

func CrosswordLetterAt(c *Crossword, row, column int) *CrosswordLetterRef {
	return &CrosswordLetterRef{
		LetterRef: LetterRef{
			pos:       row*c.columns + column,
			crossword: c,
		},
	}
}

func (l *CrosswordLetterRef) Next() *CrosswordLetterRef {
	if l.pos+1 < len(l.crossword.data) {
		return &CrosswordLetterRef{
			LetterRef: LetterRef{
				pos:       l.pos + 1,
				crossword: l.crossword,
			},
		}
	}
	return nil
}

type WordLetterRef struct {
	LetterRef
	word *WordRef
}

func WordLetter(word *WordRef) *WordLetterRef {
	return &WordLetterRef{
		LetterRef: LetterRef{
			pos:       word.pos,
			crossword: word.crossword,
		},
		word: word,
	}
}

func (l *WordLetterRef) Next() *WordLetterRef {
	if l.word.direction == horizontal {
		if l.pos+1 < l.word.pos+l.word.length {
			return &WordLetterRef{
				LetterRef: LetterRef{
					pos:       l.pos + 1,
					crossword: l.crossword,
				},
				word: l.word,
			}
		}
		return nil
	}

	if l.pos+l.crossword.columns < l.word.pos+l.word.length*l.crossword.columns {
		return &WordLetterRef{
			LetterRef: LetterRef{
				pos:       l.pos + l.crossword.columns,
				crossword: l.crossword,
			},
			word: l.word,
		}
	}
	return nil
}

package crossword_test

import (
	"fmt"
	"testing"

	"github.com/ahboujelben/go-crossword/modules/crossword"
	"github.com/ahboujelben/go-crossword/modules/dictionary"
	"github.com/stretchr/testify/assert"
)

func TestGenerateCrossword(t *testing.T) {
	wordDict := dictionary.NewWordDictionary()
	for rows := 3; rows <= 13; rows++ {
		for columns := 3; columns <= 13; columns++ {
			c := columns
			r := rows
			t.Run(fmt.Sprintf("Rows=%d_Columns=%d", rows, columns), func(t *testing.T) {
				t.Parallel()
				result := crossword.NewCrossword(crossword.CrosswordConfig{
					Rows:     r,
					Cols:     c,
					Threads:  100,
					WordDict: wordDict,
				})

				assert.True(t, result.Crossword.IsFilled())
				for word := crossword.ColumnWord(result.Crossword); word != nil; word = word.Next() {
					wordValue := string(word.GetValue())
					assert.True(t, wordDict.Contains(wordValue))
				}
			})
		}
	}
}

package crossword

import (
	"context"
	"fmt"
	"math/rand"
	"slices"
	"sort"

	"github.com/ahboujelben/go-crossword/modules/dictionary"
)

// starting with an empty crossword, try to fill the crossword word by word,
// starting with the longest ones. if stuck or we ended up creating
// non-existent words, backtrack and try again.
func generateCrossword(ctx context.Context, rows, columns int, seed int64, wordDict dictionary.WordDictionary, solvedCrossword chan CrosswordResult) {
	random := rand.New(rand.NewSource(seed))
	crossword := newEmptyCrossword(rows, columns, random)
	crawler := newCrosswordCrawler(crossword)

	for {
		// abort if the context is cancelled - a solution has already been found
		select {
		case <-ctx.Done():
			return
		default:
		}

		// if the whole crossword is filled then a solution has been found
		if crossword.IsFilled() {
			select {
			case solvedCrossword <- newCrosswordResult(crossword, seed):
				return
			default:
				return
			}
		}

		currentWord := crawler.currentWord()
		currentWordValue := currentWord.GetValue()

		if currentWord.IsFilled() {
			if !wordDict.Contains(string(currentWordValue)) {
				crawler.backtrack()
				continue
			}
			crawler.goToNextWord()
			continue
		}

		// find possible candidates for the current word based on the current
		// state of the crossword
		candidates := wordDict.Candidates(currentWordValue)
		// exclude words that are already in the crossword
		candidates = slices.DeleteFunc(candidates, func(e int) bool {
			_, exists := crawler.wordsSoFar[wordDict.AllWords[e]]
			return exists
		})

		if len(candidates) == 0 {
			crawler.backtrack()
			continue
		}

		candidate := wordDict.AllWords[candidates[random.Intn(len(candidates))]]
		crawler.pushToStack(currentWordValue)
		currentWord.SetValue([]byte(candidate))
		crawler.storeWord(candidate)
		crawler.goToNextWord()
	}
}

func newEmptyCrossword(rows, columns int, random *rand.Rand) *Crossword {
	if rows < 1 {
		panic(fmt.Sprintf("invalid rows: %d", rows))
	}
	if columns < 1 {
		panic(fmt.Sprintf("invalid columns: %d", columns))
	}

	data := make([]byte, columns*rows)

	// create blank squares based on specific conditions
	for i := range rows {
		for j := range columns {
			if i%2 == 1 && (i+j)%2 == 0 {
				data[i*columns+j] = Blank
			}
			if i == 0 && j%2 == 0 && random.Float64() < 0.75 {
				data[j+random.Intn(rows)*columns] = Blank
			}
		}

		if i%2 == 0 && columns > 7 {
			if random.Float64() < 0.75 {
				data[i*columns+random.Intn(columns)] = Blank
			}
		}
	}

	// replace any single letter words with empty space
	for y := range rows {
		for x := range columns {
			if data[y*columns+x] == Blank {
				continue
			}
			if (x == 0 || data[y*columns+x-1] == Blank) &&
				(x == columns-1 || data[y*columns+x+1] == Blank) &&
				(y == 0 || data[(y-1)*columns+x] == Blank) &&
				(y == rows-1 || data[(y+1)*columns+x] == Blank) {
				data[y*columns+x] = Blank
			}
		}
	}

	return &Crossword{
		rows:    rows,
		columns: columns,
		data:    data,
	}
}

type crosswordCrawler struct {
	words            []WordRef
	stack            []wordStack
	wordsSoFar       map[string]struct{}
	currentWordIndex int
	totalBacktracks  int
	backtrackSteps   int
}

type wordStack struct {
	index int
	word  []byte
}

func newCrosswordCrawler(c *Crossword) *crosswordCrawler {
	words := make([]WordRef, 0)
	for w := Word(c); w != nil; w = w.Next() {
		words = append(words, *w)
	}
	sort.Slice(words, func(i, j int) bool {
		return words[i].length > words[j].length
	})
	return &crosswordCrawler{
		words:            words,
		stack:            []wordStack{},
		wordsSoFar:       make(map[string]struct{}),
		currentWordIndex: 0,
		totalBacktracks:  0,
		backtrackSteps:   3,
	}
}

func (c *crosswordCrawler) pushToStack(value []byte) {
	c.stack = append(c.stack, wordStack{index: c.currentWordIndex, word: value})
}

func (c *crosswordCrawler) storeWord(value string) {
	c.wordsSoFar[value] = struct{}{}
}

func (c *crosswordCrawler) currentWord() *WordRef {
	return &c.words[c.currentWordIndex]
}

func (c *crosswordCrawler) goToNextWord() {
	c.currentWordIndex++
}

func (c *crosswordCrawler) backtrack() {
	c.totalBacktracks++
	if c.totalBacktracks%10 == 0 {
		c.backtrackSteps += 3
	}
	for range c.backtrackSteps {
		prevWord := c.stack[len(c.stack)-1]
		c.stack = c.stack[:len(c.stack)-1]
		wordToBeDeleted := string(c.words[prevWord.index].GetValue())
		delete(c.wordsSoFar, wordToBeDeleted)
		c.currentWordIndex = prevWord.index
		c.words[c.currentWordIndex].SetValue(prevWord.word)
		if len(c.stack) == 0 {
			c.backtrackSteps = 3
			c.totalBacktracks = 0
			break
		}
	}
}

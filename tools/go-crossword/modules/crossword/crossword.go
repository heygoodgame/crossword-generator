package crossword

import (
	"context"
	"math/rand"
	"sync"

	"github.com/ahboujelben/go-crossword/modules/dictionary"
)

const Blank = '.'

type Crossword struct {
	rows    int
	columns int
	data    []byte
}

type CrosswordConfig struct {
	Rows     int
	Cols     int
	Threads  int
	WordDict dictionary.WordDictionary
	Seed     int64
}

type CrosswordResult struct {
	Crossword *Crossword
	Seed      int64
}

func newCrosswordResult(crossword *Crossword, seed int64) CrosswordResult {
	return CrosswordResult{
		Crossword: crossword,
		Seed:      seed,
	}
}

func NewCrossword(config CrosswordConfig) CrosswordResult {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	solvedCrossword := make(chan CrosswordResult, 1)

	if config.Seed != 0 {
		generateCrossword(ctx, config.Rows, config.Cols, config.Seed, config.WordDict, solvedCrossword)
		return <-solvedCrossword
	}

	var wg sync.WaitGroup

	// Generating a random crossword can take an unpredictable amount of time,
	// depending on the initial crossword configuration and the words that are
	// tried. To speed up the process, we run multiple goroutines to generate
	// crosswords and return the first one that is solved. This typically takes
	// less than a second to generate a 13x13 crossword.
	for range config.Threads {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer cancel()
			generateCrossword(ctx, config.Rows, config.Cols, rand.Int63(), config.WordDict, solvedCrossword)
		}()
	}

	wg.Wait()

	return <-solvedCrossword
}

func (c *Crossword) Columns() int {
	return c.columns
}

func (c *Crossword) Rows() int {
	return c.rows
}

func (c *Crossword) IsFilled() bool {
	for letter := CrosswordLetter(c); letter != nil; letter = letter.Next() {
		if letter.IsEmpty() {
			return false
		}
	}
	return true
}

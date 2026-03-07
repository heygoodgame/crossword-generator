package main

import (
	"flag"
	"fmt"

	"github.com/ahboujelben/go-crossword/cli/renderer"
)

// parseResult holds the parsed command-line arguments
type parseResult struct {
	Rows           int
	Cols           int
	CrosswordSeed  int64
	Threads        int
	Renderer       renderer.Renderer
	Format         string
	DictionaryPath string
	MinScore       int
}

// parseArguments parses command-line arguments and returns a ParseResult
func parseArguments() (*parseResult, error) {
	rows := flag.Int("rows", 13, "number of rows in the crossword ([3, 15])")
	cols := flag.Int("cols", 13, "number of columns in the crossword ([3, 15])")
	crosswordSeed := flag.Int64("seed", 0, "seed for the crossword generation ([0, 2^63-1], 0 for a random seed)")
	threads := flag.Int("threads", 100, "number of goroutines to use (>= 1)")
	compact := flag.Bool("compact", false, "compact rendering")
	format := flag.String("format", "text", "output format: text or json")
	dictionaryPath := flag.String("dictionary", "", "path to external dictionary file (word;score format)")
	minScore := flag.Int("min-score", 0, "minimum word score for external dictionary (0-100)")

	flag.Parse()

	if !isSizeValid(*rows) || !isSizeValid(*cols) {
		return nil, fmt.Errorf("invalid dimensions")
	}

	if !isSeedValid(*crosswordSeed) {
		return nil, fmt.Errorf("invalid crossword seed")
	}

	if *threads < 1 {
		return nil, fmt.Errorf("invalid number of goroutines")
	}

	if *format != "text" && *format != "json" {
		return nil, fmt.Errorf("invalid format %q: must be text or json", *format)
	}

	var render renderer.Renderer = renderer.NewStandardRenderer()
	if *compact {
		render = renderer.NewCompactRenderer()
	}

	return &parseResult{
		Rows:           *rows,
		Cols:           *cols,
		CrosswordSeed:  *crosswordSeed,
		Threads:        *threads,
		Renderer:       render,
		Format:         *format,
		DictionaryPath: *dictionaryPath,
		MinScore:       *minScore,
	}, nil
}

// isSeedValid checks if a seed value is valid
func isSeedValid(seed int64) bool {
	return seed >= 0
}

// isSizeValid checks if a crossword size is valid
func isSizeValid(size int) bool {
	return size >= 3 && size <= 15
}

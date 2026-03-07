package main

import (
	"fmt"
	"os"

	"github.com/ahboujelben/go-crossword/cli/renderer"
	"github.com/ahboujelben/go-crossword/modules/crossword"
	"github.com/ahboujelben/go-crossword/modules/dictionary"
)

func generateCrossword(parseResult *parseResult) {
	if parseResult.Format == "json" {
		fmt.Fprintln(os.Stderr, "Generating crossword...")
	} else {
		fmt.Println("Generating crossword...")
	}

	crosswordResult := crossword.NewCrossword(crossword.CrosswordConfig{
		Rows:     parseResult.Rows,
		Cols:     parseResult.Cols,
		Seed:     parseResult.CrosswordSeed,
		Threads:  parseResult.Threads,
		WordDict: dictionary.NewWordDictionary(),
	})

	if parseResult.Format == "json" {
		jsonStr, err := renderer.RenderJSON(crosswordResult.Crossword, crosswordResult.Seed)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error rendering JSON: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(jsonStr)
	} else {
		fmt.Printf("\n%s\n\n", parseResult.Renderer.RenderCrossword(crosswordResult.Crossword, true))
		fmt.Println("Crossword generated successfully!")
		fmt.Printf("Seed: %d\n", crosswordResult.Seed)
	}
}

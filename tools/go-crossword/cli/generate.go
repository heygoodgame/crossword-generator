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

	var wordDict dictionary.WordDictionary
	if parseResult.DictionaryPath != "" {
		var err error
		wordDict, err = dictionary.NewWordDictionaryFromFile(parseResult.DictionaryPath, parseResult.MinScore)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading dictionary: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Loaded %d words from %s (min-score=%d)\n",
			len(wordDict.AllWords), parseResult.DictionaryPath, parseResult.MinScore)
	} else {
		wordDict = dictionary.NewWordDictionary()
	}

	crosswordResult := crossword.NewCrossword(crossword.CrosswordConfig{
		Rows:     parseResult.Rows,
		Cols:     parseResult.Cols,
		Seed:     parseResult.CrosswordSeed,
		Threads:  parseResult.Threads,
		WordDict: wordDict,
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

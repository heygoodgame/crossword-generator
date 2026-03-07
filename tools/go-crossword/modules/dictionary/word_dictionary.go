package dictionary

import (
	"bufio"
	_ "embed"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
)

//go:embed words.txt
var words string

type WordDictionary struct {
	AllWords  []string
	wordSet   map[string]struct{}
	lengthMap map[int][]int
	letterMap map[wordDictionaryKey]map[int]struct{}
}

type wordDictionaryKey struct {
	letter byte
	pos    int
}

// NewWordDictionary creates a WordDictionary from the embedded word list.
func NewWordDictionary() WordDictionary {
	return buildWordDictionary(strings.Fields(words))
}

// NewWordDictionaryFromFile creates a WordDictionary from an external file.
// Supports two formats:
//   - "word;score" per line (Jeff Chen format): filters to words with score >= minScore
//   - plain word per line: all words are included regardless of minScore
func NewWordDictionaryFromFile(filePath string, minScore int) (WordDictionary, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return WordDictionary{}, fmt.Errorf("open dictionary: %w", err)
	}
	defer f.Close()

	var wordList []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, ";", 2)
		word := strings.ToLower(parts[0])
		if word == "" {
			continue
		}

		if len(parts) == 2 {
			score, err := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err != nil {
				continue // skip malformed lines
			}
			if score < minScore {
				continue
			}
		}

		wordList = append(wordList, word)
	}
	if err := scanner.Err(); err != nil {
		return WordDictionary{}, fmt.Errorf("read dictionary: %w", err)
	}

	if len(wordList) == 0 {
		return WordDictionary{}, fmt.Errorf("no words in dictionary after filtering (minScore=%d)", minScore)
	}

	return buildWordDictionary(wordList), nil
}

// buildWordDictionary constructs a WordDictionary from a slice of words.
func buildWordDictionary(wordList []string) WordDictionary {
	dict := WordDictionary{
		AllWords:  make([]string, 0, len(wordList)),
		wordSet:   make(map[string]struct{}, len(wordList)),
		lengthMap: map[int][]int{},
		letterMap: map[wordDictionaryKey]map[int]struct{}{},
	}

	for wordIndex, word := range wordList {
		dict.AllWords = append(dict.AllWords, word)
		dict.wordSet[word] = struct{}{}
		dict.lengthMap[len(word)] = append(dict.lengthMap[len(word)], wordIndex)
		for i := range len(word) {
			key := wordDictionaryKey{letter: word[i], pos: i}
			if _, exists := dict.letterMap[key]; !exists {
				dict.letterMap[key] = map[int]struct{}{}
			}
			dict.letterMap[key][wordIndex] = struct{}{}
		}
	}

	return dict
}

func (wd WordDictionary) Contains(word string) bool {
	_, exists := wd.wordSet[word]
	return exists
}

func (wd WordDictionary) Candidates(word []byte) []int {
	candidates := make([]int, len(wd.lengthMap[len(word)]))
	copy(candidates, wd.lengthMap[len(word)])

	for i, letter := range word {
		if letter != 0 {
			key := wordDictionaryKey{letter: letter, pos: i}
			currentSet := wd.letterMap[key]
			candidates = slices.DeleteFunc(candidates, func(e int) bool {
				_, exists := currentSet[e]
				return !exists
			})
		}
	}
	return candidates
}

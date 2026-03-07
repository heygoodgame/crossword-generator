package dictionary

import (
	_ "embed"
	"slices"
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

func NewWordDictionary() WordDictionary {
	dict := WordDictionary{
		AllWords:  []string{},
		wordSet:   map[string]struct{}{},
		lengthMap: map[int][]int{},
		letterMap: map[wordDictionaryKey]map[int]struct{}{},
	}

	for wordIndex, word := range strings.Fields(words) {
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

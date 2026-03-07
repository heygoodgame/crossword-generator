package dictionary

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTempDict(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "dict.txt")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write temp dict: %v", err)
	}
	return path
}

func TestNewWordDictionaryFromFile_ScoredFormat(t *testing.T) {
	path := writeTempDict(t, "ocean;60\nparse;75\nobscure;30\nangel;90\n")
	dict, err := NewWordDictionaryFromFile(path, 50)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(dict.AllWords) != 3 {
		t.Errorf("expected 3 words, got %d", len(dict.AllWords))
	}
	if !dict.Contains("ocean") {
		t.Error("expected dictionary to contain 'ocean'")
	}
	if !dict.Contains("parse") {
		t.Error("expected dictionary to contain 'parse'")
	}
	if !dict.Contains("angel") {
		t.Error("expected dictionary to contain 'angel'")
	}
	if dict.Contains("obscure") {
		t.Error("expected dictionary to NOT contain 'obscure' (score 30 < 50)")
	}
}

func TestNewWordDictionaryFromFile_PlainFormat(t *testing.T) {
	path := writeTempDict(t, "hello\nworld\ntest\n")
	dict, err := NewWordDictionaryFromFile(path, 50)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(dict.AllWords) != 3 {
		t.Errorf("expected 3 words, got %d", len(dict.AllWords))
	}
	if !dict.Contains("hello") {
		t.Error("expected dictionary to contain 'hello'")
	}
}

func TestNewWordDictionaryFromFile_MixedFormat(t *testing.T) {
	path := writeTempDict(t, "ocean;60\nplainword\nlow;10\n")
	dict, err := NewWordDictionaryFromFile(path, 50)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// ocean (60 >= 50) + plainword (no score, included) = 2
	if len(dict.AllWords) != 2 {
		t.Errorf("expected 2 words, got %d: %v", len(dict.AllWords), dict.AllWords)
	}
	if !dict.Contains("ocean") {
		t.Error("expected 'ocean'")
	}
	if !dict.Contains("plainword") {
		t.Error("expected 'plainword'")
	}
	if dict.Contains("low") {
		t.Error("should not contain 'low' (score 10 < 50)")
	}
}

func TestNewWordDictionaryFromFile_UppercaseNormalized(t *testing.T) {
	path := writeTempDict(t, "OCEAN;60\nPARSE;75\n")
	dict, err := NewWordDictionaryFromFile(path, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !dict.Contains("ocean") {
		t.Error("expected lowercased 'ocean'")
	}
	if dict.Contains("OCEAN") {
		t.Error("should not contain uppercase 'OCEAN'")
	}
}

func TestNewWordDictionaryFromFile_MissingFile(t *testing.T) {
	_, err := NewWordDictionaryFromFile("/nonexistent/path/dict.txt", 0)
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNewWordDictionaryFromFile_EmptyAfterFilter(t *testing.T) {
	path := writeTempDict(t, "low;10\nreally_low;5\n")
	_, err := NewWordDictionaryFromFile(path, 50)
	if err == nil {
		t.Error("expected error when no words pass threshold")
	}
}

func TestNewWordDictionaryFromFile_SkipsEmptyLines(t *testing.T) {
	path := writeTempDict(t, "\nocean;60\n\nparse;75\n\n")
	dict, err := NewWordDictionaryFromFile(path, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(dict.AllWords) != 2 {
		t.Errorf("expected 2 words, got %d", len(dict.AllWords))
	}
}

func TestNewWordDictionaryFromFile_CandidatesWork(t *testing.T) {
	path := writeTempDict(t, "ocean;60\noaken;60\noasis;60\napple;60\n")
	dict, err := NewWordDictionaryFromFile(path, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Search for 5-letter words starting with 'o'
	pattern := []byte{'o', 0, 0, 0, 0}
	candidates := dict.Candidates(pattern)
	if len(candidates) != 3 {
		t.Errorf("expected 3 candidates starting with 'o', got %d", len(candidates))
	}
}

func TestNewWordDictionary_EmbeddedStillWorks(t *testing.T) {
	dict := NewWordDictionary()
	if len(dict.AllWords) == 0 {
		t.Error("embedded dictionary should have words")
	}
}

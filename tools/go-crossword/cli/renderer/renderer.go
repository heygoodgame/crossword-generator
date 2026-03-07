package renderer

import (
	"github.com/ahboujelben/go-crossword/modules/crossword"
)

type Renderer interface {
	RenderCrossword(c *crossword.Crossword, solved bool) string
}

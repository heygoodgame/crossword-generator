package renderer

import (
	"fmt"

	"github.com/ahboujelben/go-crossword/modules/crossword"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
)

var blackColor = lipgloss.Color("#000")
var whiteColor = lipgloss.Color("#fff")

func getBorderTable() *table.Table {
	borderColor := blackColor
	if lipgloss.HasDarkBackground() {
		borderColor = whiteColor
	}

	return table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(borderColor))
}

type StandardRenderer struct {
}

func NewStandardRenderer() StandardRenderer {
	return StandardRenderer{}
}

func (f StandardRenderer) RenderCrossword(c *crossword.Crossword, solved bool) string {
	crosswordGrid := getBorderTable().
		BorderRow(true).
		StyleFunc(func(row, col int) lipgloss.Style {
			s := lipgloss.NewStyle()
			if row == 0 || col == 0 {
				s = s.Foreground(lipgloss.Color("#00ff00"))
			}
			return s
		}).
		Data(newCrosswordCharmWrapper(c, solved))

	return crosswordGrid.Render()
}

type crosswordCharmWrapper struct {
	*crossword.Crossword
	solved bool
}

func newCrosswordCharmWrapper(c *crossword.Crossword, solved bool) *crosswordCharmWrapper {
	return &crosswordCharmWrapper{
		Crossword: c,
		solved:    solved,
	}
}

func (w *crosswordCharmWrapper) Columns() int {
	return w.Crossword.Columns() + 1
}

func (w *crosswordCharmWrapper) Rows() int {
	return w.Crossword.Rows() + 1
}

func (w *crosswordCharmWrapper) At(row, column int) string {
	if row == 0 && column == 0 {
		return "   "
	}
	if row == 0 {
		return fmt.Sprintf(" %-2d", column)
	}
	if column == 0 {
		return fmt.Sprintf(" %-2d", row)
	}
	letter := crossword.CrosswordLetterAt(w.Crossword, row-1, column-1)
	switch {
	case letter.IsBlank():
		return "▐█▌"
	case letter.IsEmpty() || !w.solved:
		return "   "
	default:
		return fmt.Sprintf(" %c ", letter.GetValue()+'A'-'a')
	}
}

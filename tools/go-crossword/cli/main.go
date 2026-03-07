package main

import (
	"fmt"
)

func main() {
	parseResult, err := parseArguments()
	if err != nil {
		fmt.Println(fmt.Errorf("Something is not right: %w", err))
		return
	}

	generateCrossword(parseResult)
}

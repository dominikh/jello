// Copyright 2023 the Vello Authors
// Copyright 2024 Dominik Honnef and contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	var (
		in      string
		out     string
		verbose bool
	)
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [-v] -in <dir> -out <dir>\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.StringVar(&in, "in", "", "Path to `directory` to process")
	flag.StringVar(&out, "out", "./out", "Path to output `directory`")
	flag.BoolVar(&verbose, "v", false, "Be verbose")
	flag.Parse()

	if len(flag.Args()) != 0 {
		flag.Usage()
		os.Exit(2)
	}

	write := func(src []byte, name string) error {
		if !strings.HasSuffix(name, ".wgsl") {
			name += ".wgsl"
		}
		return os.WriteFile(filepath.Join(out, name), src, 0666)
	}

	dief := func(f string, v ...any) {
		fmt.Fprintf(os.Stderr, f, v...)
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}

	var permutations map[string][]Permutation
	permSource, err := os.ReadFile(filepath.Join(in, "permutations"))
	switch err {
	case nil:
		permutations = parsePermutations(permSource)
	case io.EOF:
		if verbose {
			fmt.Fprintln(os.Stderr, "didn't find permutations")
		}
	default:
		dief("Couldn't read permutations: %s", err)
	}

	defaultDefines := map[string]struct{}{"full": {}}

	p := Preprocessor{
		ImportDir: filepath.Join(in, "shared"),
		Verbose:   verbose,
	}

	matches, err := filepath.Glob(filepath.Join(in, "*.wgsl"))
	if err != nil {
		panic(err)
	}

	if err := os.MkdirAll(out, 0777); err != nil {
		dief("Couldn't create output directory: %s", err)
	}

	for i, m := range matches {
		if verbose {
			if i != 0 {
				fmt.Fprintln(os.Stderr)
			}
			fmt.Fprintf(os.Stderr, "compiling %s\n", filepath.Base(m))
		}
		src, err := os.ReadFile(m)
		if err != nil {
			dief("Couldn't read %q: %s", m, err)
		}

		shaderName := strings.TrimSuffix(filepath.Base(m), ".wgsl")
		if perms, ok := permutations[shaderName]; ok {
			for _, perm := range perms {
				defines := maps.Clone(defaultDefines)
				for _, d := range perm.Defines {
					defines[d] = struct{}{}
				}
				if verbose {
					fmt.Fprintf(os.Stderr, "preprocessing permutation %q with defines %v\n",
						perm.Name, perm.Defines)
				}
				p.Defines = defines
				src, err := p.Preprocess(src, perm.Name)
				if err != nil {
					dief("Couldn't preprocess source: %s", err)
				}
				src = postprocess(src)
				write(src, perm.Name)
			}
		} else {
			p.Defines = defaultDefines
			src, err = p.Preprocess(src, m)
			if err != nil {
				dief("Couldn't preprocess source: %s", err)
			}
			src = postprocess(src)
			write(src, filepath.Base(m))
		}
	}
}

type Preprocessor struct {
	ImportDir string
	Verbose   bool
	Defines   map[string]struct{}

	imports map[string][]byte
}

func (p *Preprocessor) debugf(f string, v ...any) {
	if !p.Verbose {
		return
	}
	fmt.Fprintf(os.Stderr, f, v...)
	fmt.Fprintln(os.Stderr)
}

func (p *Preprocessor) getImport(name string) ([]byte, error) {
	p.debugf("substituting import %q", name)
	if src, ok := p.imports[name]; ok {
		return src, nil
	}
	p.debugf("loading import %q from disk", name)
	src, err := os.ReadFile(filepath.Join(p.ImportDir, name+".wgsl"))
	if err != nil {
		return nil, err
	}
	if p.imports == nil {
		p.imports = make(map[string][]byte)
	}
	p.imports[name] = src
	return src, nil
}

func (p *Preprocessor) Preprocess(source []byte, name string) ([]byte, error) {
	var out []byte
	nl := []byte("\n")
	space := []byte(" ")
	dirMarker := []byte("#")
	commentMarker := []byte("//")
	let := []byte("let ")
	type stackItem struct {
		active     bool
		elsePassed bool
	}
	var stack []stackItem
	lineNo := 0
	location := func() string {
		return fmt.Sprintf("%s:%d", name, lineNo)
	}
	errorf := func(f string, v ...any) error {
		v = append(v[:len(v):len(v)], location())
		return fmt.Errorf(f+" (at %s)", v...)
	}
	error := func(f string) error {
		return errorf("%s", f)
	}
allLines:
	for len(source) > 0 {
		lineNo++
		var line []byte
		line, source, _ = bytes.Cut(source, nl)

		for len(line) > 0 {
			hashIdx := bytes.IndexByte(line, '#')
			commentIdx := bytes.Index(line, commentMarker)

			if hashIdx == -1 || (commentIdx != -1 && commentIdx < hashIdx) {
				// No directives that aren't commented
				break
			}

			end := bytes.IndexByte(line[hashIdx+1:], ' ')
			if end == -1 {
				end = len(line)
			} else {
				end += hashIdx + 1
			}

			directive := string(line[hashIdx+1 : end])
			atStart := bytes.HasPrefix(bytes.TrimSpace(line), dirMarker)
			arg := bytes.TrimSpace(line[end:])

			p.debugf("processing directive %q", directive)

			switch directive {
			case "ifdef", "ifndef", "else", "endif", "enable":
				if !atStart {
					return nil, errorf(
						"%q directives must be the first non-whitespace item on their line",
						directive)
				}
			}

			switch directive {
			case "ifdef", "ifndef":
				_, exists := p.Defines[string(arg)]
				active := (directive == "ifdef") == exists
				stack = append(stack, stackItem{
					active:     active,
					elsePassed: false,
				})
				if active {
					if directive == "ifdef" {
						p.debugf("current branch is active (%s is defined)", string(arg))
					} else {
						p.debugf("current branch is active (%s is not defined)", string(arg))
					}
				} else {
					if directive == "ifdef" {
						p.debugf("current branch is not active (%s is not defined)", string(arg))
					} else {
						p.debugf("current branch is not active (%s is defined)", string(arg))
					}
				}
				continue allLines

			case "else":
				// XXX shouldn't we complain about an else without a matching stack entry?
				if len(stack) > 0 {
					item := &stack[len(stack)-1]
					if item.elsePassed {
						return nil, error("second else for same ifdef/ifndef")
					} else {
						item.elsePassed = true
						item.active = !item.active
					}
					if item.active {
						p.debugf("current branch is active")
					} else {
						p.debugf("current branch is not active")
					}
				}
				if len(arg) != 0 {
					return nil, error("#else directive doesn't accept arguments")
				}
				continue allLines

			case "endif":
				if len(stack) == 0 {
					return nil, error("mismatched endif")
				}
				stack = stack[:len(stack)-1]
				// XXX if endif allows a trailing comment, then shouldn't all directives?
				if len(arg) != 0 && !bytes.HasPrefix(arg, commentMarker) {
					return nil, error("#endif directive doesn't accept arguments")
				}
				continue allLines

			case "import":
				out = append(out, line[:hashIdx]...)
				if len(arg) == 0 {
					return nil, error("#import needs an argument")
				}
				var importName []byte
				importName, line, _ = bytes.Cut(arg, space)
				importSrc, err := p.getImport(string(importName))
				if err != nil {
					return nil, errorf("couldn't import %q: %w", importName, err)
				}
				all := true
				for _, item := range stack {
					if !item.active {
						all = false
						break
					}
				}
				if all {
					imported, err := p.Preprocess(importSrc, "#include "+string(importName))
					if err != nil {
						return nil, err
					}
					out = append(out, imported...)
				}

			case "enable":
				all := true
				for _, item := range stack {
					if !item.active {
						all = false
						break
					}
				}
				if all {
					out = append(out, "//__"...)
					out = append(out, line...)
					out = append(out, '\n')
				}
				continue allLines

			default:
				return nil, errorf("unknown preprocessor directive %q", directive)
			}
		}

		all := true
		for _, item := range stack {
			if !item.active {
				all = false
				break
			}
		}

		if all {
			if bytes.HasPrefix(line, let) {
				out = append(out, "const"...)
				out = append(out, line[3:]...)
			} else {
				out = append(out, line...)
			}
			out = append(out, '\n')
		}
	}

	return out, nil
}

type Permutation struct {
	Name    string
	Defines []string
}

func parsePermutations(source []byte) map[string][]Permutation {
	nl := []byte("\n")
	colon := []byte(":")
	out := make(map[string][]Permutation)
	var currentSource []byte
	for len(source) > 0 {
		var line []byte
		line, source, _ = bytes.Cut(source, nl)
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		if line[0] == '+' {
			line = line[1:]
			if len(currentSource) != 0 {
				parts := bytes.SplitN(line, colon, 2)
				if len(parts) == 0 {
					continue
				}
				name := string(bytes.TrimSpace(parts[0]))
				var defines []string
				if len(parts) == 2 {
					defines = strings.Fields(string(parts[1]))
				}
				out[string(currentSource)] = append(out[string(currentSource)], Permutation{name, defines})
			}
		} else {
			currentSource = line
		}
	}
	return out
}

func postprocess(src []byte) []byte {
	return bytes.ReplaceAll(src, []byte("//__#enable"), nil)
}

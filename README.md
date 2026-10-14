*This project has been created as part of the 42 curriculum by paapahid.*

# Call Me Maybe

Function calling with a small language model and constrained decoding.

## Description

Given a request in natural language, the program does **not** answer it.
It translates it into a function call: the name of the function and its
arguments, with the right types.

```
"What is the sum of 40 and 2?"  ->  fn_add_numbers {"a": 40.0, "b": 2.0}
```

It uses `Qwen/Qwen3-0.6B`, a model that is too small to write valid JSON
reliably by itself. The output is always valid because the generation is
**constrained**: at every step the logits of the tokens that would break
the structure or the schema are set to minus infinity, and only the
remaining tokens can be chosen.

The function is chosen by the model. There are no keywords or rules
based on the text of the prompt.

Supported types: `number`, `integer`, `string` and `boolean`. Functions
without parameters are supported too.

## Instructions

Requirements: Python 3.10 or later and [uv](https://docs.astral.sh/uv/).

```bash
make install     # uv sync: creates the environment and installs everything
make run         # runs the program with the default files
make debug       # runs the program with pdb
make lint        # flake8 and mypy
make lint-strict # flake8 and mypy --strict
make clean       # removes caches and the virtual environment
```

Direct usage:

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Argument | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

Each output item has exactly three keys: `prompt`, `name` and
`parameters`. If a file is missing or invalid, the program prints a
single line starting with `Error:` and exits with code 1.

## How it works

1. **Prompt.** The text sent to the model follows the chat format that
   Qwen was trained with: the functions go inside `<tools>`, then the
   user request, and the answer of the assistant starts with
   `<tool_call>{"name": "`.
2. **Function name.** The names of the functions are stored in a trie.
   Only tokens that follow a path of the trie are allowed. The name ends
   when the model writes the closing quote, so a name that is the
   beginning of another one (`fn_sum`, `fn_sum_squares`) can be chosen
   correctly.
3. **Arguments.** A state machine (`JsonState`) controls the JSON:
   which characters are valid after each character. The keys come from
   a trie of the parameters that are still pending, so no key is
   repeated and the object closes when all of them are written. The
   value depends on the type of the parameter: strings, numbers with
   sign and decimals, integers, and `true` or `false`.
4. **Masking.** Every forbidden token gets `-inf`, and the token with
   the best score among the rest is chosen.
5. **Glued tokens.** The model often wants a token that joins a
   structural character to other ones, such as `"}` or `",`. When the
   character that closes a string, a number or a name is allowed, it
   receives the best score of all the tokens that start with it.
6. **Result.** Values of type `number` are written as floats (`2.0`),
   and values of type `integer` as integers.
7. **Safety.** The generation of the arguments stops with an error after
   500 tokens, so the program can never hang.

## Project structure

```
src/
├── __main__.py          arguments and error handling
├── decoder/
│   ├── generator.py     generate_json: constrained generation of the arguments
│   ├── func_select.py   choice of the function with the trie
│   ├── selector.py      masking of the logits and choice of the token
│   ├── generic.py       JSON states, valid characters and transitions
│   ├── bool_node.py     true and false, letter by letter
│   └── trie.py          prefix tree
├── errors/errors.py     exceptions of the project
├── models/schema.py     pydantic models
├── parse/parser.py      reading, validation and writing of files
└── pipeline/process.py  from a prompt to a function call
```

## Testing

The folder `tests/` has JSON files to try special cases (decimals,
negative numbers, booleans, integers, functions without parameters,
names and keys that start the same). For example:

```bash
uv run python -m src \
  --functions_definition tests/all_functions.json \
  --input tests/all_prompts.json \
  --output tests/all_results.json
```

## Limitations

- Supported types: `number`, `integer`, `string` and `boolean`. There
  are no arrays, objects or `null`.
- Inside a string value the characters `"`, `\` and line breaks cannot
  be written.

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [JSON specification](https://www.json.org/)
- Willard and Louf, *Efficient Guided Generation for Large Language
  Models*, about constrained generation.
- Documentation of [uv](https://docs.astral.sh/uv/) and
  [pydantic](https://docs.pydantic.dev/).

### Use of AI

Claude was used as a tutor: to explain concepts
(tokenization, logits, constrained decoding, state machines), to review
the code and find bugs by reading the output of the model.
The code was tested step by step, and every part was run and checked.
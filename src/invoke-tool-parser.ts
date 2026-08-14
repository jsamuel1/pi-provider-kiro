// ABOUTME: Extracts XML-dialect tool calls from content text as a fallback.
// ABOUTME: Parses <invoke name="..."><parameter name="...">value</parameter></invoke> blocks.

export interface InvokeToolCall {
  toolUseId: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface InvokeParseResult {
  toolCalls: InvokeToolCall[];
  cleanedText: string;
}

const INVOKE_OPEN_ANCHORED = /^<invoke name="([\w-]+)">/;
const INVOKE_OPEN_SEARCH = "<invoke name=";
const INVOKE_OPEN_ANYWHERE = /<invoke name="[\w-]+">/;
const PARAM_OPEN_ANCHORED = /^<parameter name="([^"]*)">/;
const INVOKE_CLOSE = "</invoke>";
const PARAM_CLOSE = "</parameter>";
const FENCE = "```";

interface ParsedBlock {
  name: string;
  arguments: Record<string, unknown>;
  /** Index just past the block's `</invoke>`. */
  end: number;
}

/**
 * Byte ranges covered by an *opened* fenced code block. Fences are counted
 * pairwise; a trailing unclosed fence extends to end of text. Used to reject
 * `<invoke>` blocks that appear inside documentation rather than as a real
 * (misrouted) tool call — model output that discusses this very bug quotes the
 * dialect verbatim, and executing a command out of a code sample would be
 * strictly worse than not recovering it.
 */
function fencedRanges(text: string): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = [];
  let searchFrom = 0;
  while (true) {
    const open = text.indexOf(FENCE, searchFrom);
    if (open < 0) break;
    const close = text.indexOf(FENCE, open + FENCE.length);
    if (close < 0) {
      ranges.push({ start: open, end: text.length });
      break;
    }
    ranges.push({ start: open, end: close + FENCE.length });
    searchFrom = close + FENCE.length;
  }
  return ranges;
}

/** Thrown when a block's markup cannot be attributed unambiguously. */
class AmbiguousDialectError extends Error {}

/**
 * Parses one `<invoke>` block starting at `start`, or returns null if the
 * markup there is not a complete, well-formed block.
 *
 * The block end is found structurally — by walking parameters — never by
 * searching for `</invoke>`. A naive search would stop at a `</invoke>` that
 * happens to sit inside a parameter value, truncating the value and then
 * accepting the truncation as if it were the whole argument.
 *
 * Strict by design: the body must consist of nothing but whitespace and
 * well-formed `<parameter name="...">...</parameter>` elements. Any other
 * content (prose, an ellipsis placeholder, a stray tag) rejects the block,
 * because a partial parse would either fabricate arguments or half-strip the
 * text.
 */
function parseBlockAt(text: string, start: number): ParsedBlock | null {
  const openMatch = INVOKE_OPEN_ANCHORED.exec(text.substring(start));
  if (!openMatch) return null;

  const args: Record<string, unknown> = {};
  let cursor = start + openMatch[0].length;

  while (true) {
    const remainder = text.substring(cursor);
    const leadingWhitespace = remainder.length - remainder.trimStart().length;
    if (leadingWhitespace > 0) {
      cursor += leadingWhitespace;
      continue;
    }
    if (text.startsWith(INVOKE_CLOSE, cursor)) {
      return { name: openMatch[1], arguments: args, end: cursor + INVOKE_CLOSE.length };
    }
    const paramMatch = PARAM_OPEN_ANCHORED.exec(text.substring(cursor));
    if (!paramMatch) return null;
    const valueStart = cursor + paramMatch[0].length;
    const valueEnd = text.indexOf(PARAM_CLOSE, valueStart);
    if (valueEnd < 0) return null;
    const value = text.substring(valueStart, valueEnd);
    // A parameter value that itself contains an `<invoke>` open tag makes the
    // block unattributable: the value's real extent may have been cut short by
    // a `</parameter>` belonging to the nested markup, and the nested tag would
    // otherwise be harvested as an independent call. Neither outcome is
    // acceptable — one corrupts an argument, the other invents a call out of
    // data — so the whole text is abandoned rather than guessed at.
    if (INVOKE_OPEN_ANYWHERE.test(value)) throw new AmbiguousDialectError();
    args[paramMatch[1]] = coerceValue(value);
    cursor = valueEnd + PARAM_CLOSE.length;
  }
}

/**
 * Parameter values are raw text — multi-line, and freely containing quotes,
 * newlines, `>` and braces — so they are preserved byte-for-byte. The single
 * exception is a value whose first non-whitespace character is `[` or `{` and
 * which parses as JSON: those are structured arguments (array/object-typed
 * parameters) that the dialect encodes as JSON, and are decoded so the
 * recovered call matches the tool's schema. Scalars are never coerced, so a
 * literal `true` or `42` stays the string the model wrote.
 */
function coerceValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("[") && !trimmed.startsWith("{")) return raw;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return raw;
  }
}

/**
 * Recovers tool calls that a model emitted as XML text instead of as structured
 * tool-use frames. Mirrors {@link parseBracketToolCalls}: returns the recovered
 * calls plus the text with each consumed `<invoke>` span spliced out. A block
 * that cannot be parsed in full is left in the text untouched.
 */
export function parseInvokeToolCalls(text: string): InvokeParseResult {
  const toolCalls: InvokeToolCall[] = [];
  const removals: Array<{ start: number; end: number }> = [];
  const fences = fencedRanges(text);
  const untouched: InvokeParseResult = { toolCalls: [], cleanedText: text };

  let cursor = 0;
  while (cursor < text.length) {
    const openStart = text.indexOf(INVOKE_OPEN_SEARCH, cursor);
    if (openStart < 0) break;
    let block: ParsedBlock | null;
    try {
      block = parseBlockAt(text, openStart);
    } catch (e) {
      if (e instanceof AmbiguousDialectError) return untouched;
      throw e;
    }
    if (block === null) {
      cursor = openStart + INVOKE_OPEN_SEARCH.length;
      continue;
    }
    const insideFence = fences.some((r) => openStart >= r.start && openStart < r.end);
    if (!insideFence) {
      toolCalls.push({ toolUseId: crypto.randomUUID(), name: block.name, arguments: block.arguments });
      removals.push({ start: openStart, end: block.end });
    }
    // Resume past the whole block: its parameter values are data, not markup.
    // With the nesting bail above this is belt-and-braces — any value holding a
    // full open tag has already abandoned the parse — so no test can currently
    // distinguish it from resuming at the search hit. Kept because it is the
    // structurally correct resume point, and the bail is the only thing making
    // the weaker one safe.
    cursor = block.end;
  }

  // Splice out consumed spans in reverse order so earlier indices stay valid.
  let cleanedText = text;
  for (let i = removals.length - 1; i >= 0; i--) {
    const { start, end } = removals[i];
    cleanedText = cleanedText.substring(0, start) + cleanedText.substring(end);
  }

  return { toolCalls, cleanedText };
}

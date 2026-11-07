//! Conservative output diagnostics. A warning never repairs or reclassifies content.
const std = @import("std");

pub const ToolWarning = enum { orphan_dsml_closers };

/// Recognize only a whole answer made of known closing tags and ASCII whitespace.
/// Prose, quotes, JSON, code fences, openers and partial/unknown tags remain ordinary text.
/// The caller limits this check to answers with active tools outside the JSON channel.
pub fn toolWarning(answer: []const u8) ?ToolWarning {
    const blanks = " \t\r\n";
    const closers = [_][]const u8{
        "</\u{ff5c}DSML\u{ff5c} parameter>",
        "</\u{ff5c}DSML\u{ff5c} invoke>",
        "</\u{ff5c}DSML\u{ff5c} calls>",
        "</\u{ff5c}DSML\u{ff5c}parameter>",
        "</\u{ff5c}DSML\u{ff5c}invoke>",
        "</\u{ff5c}DSML\u{ff5c}tool_calls>",
    };
    var rest = std.mem.trim(u8, answer, blanks);
    if (rest.len == 0) return null;
    while (rest.len > 0) {
        const n = for (closers) |tag| {
            if (std.mem.startsWith(u8, rest, tag)) break tag.len;
        } else return null;
        rest = std.mem.trimStart(u8, rest[n..], blanks);
    }
    return .orphan_dsml_closers;
}

test "output diagnostic is limited to complete orphan closing tags" {
    const close = "</\u{ff5c}DSML\u{ff5c} calls>";
    try std.testing.expectEqual(ToolWarning.orphan_dsml_closers, toolWarning(" \n" ++ close ++ "\t" ++ close ++ "\r").?);
    try std.testing.expectEqual(ToolWarning.orphan_dsml_closers, toolWarning("</\u{ff5c}DSML\u{ff5c}parameter></\u{ff5c}DSML\u{ff5c}invoke></\u{ff5c}DSML\u{ff5c}tool_calls>").?);
    for ([_][]const u8{ "", " \t", "\"" ++ close ++ "\"", "```\n" ++ close ++ "\n```", "Example: " ++ close, close ++ " suffix", "{\"example\":\"" ++ close ++ "\"}", "<\u{ff5c}DSML\u{ff5c} calls>" ++ close, "</\u{ff5c}DSML\u{ff5c}  calls>", "</\u{ff5c}DSML\u{ff5c} calls" }) |text| try std.testing.expect(toolWarning(text) == null);
}

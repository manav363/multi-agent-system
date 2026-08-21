use crate::core::agent::AgentRole;
use crate::core::text::truncate_chars;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// Reasoning lines kept on screen while an agent is still thinking.
const LIVE_THOUGHT_LINES: usize = 4;

const DIM: Color = Color::Rgb(120, 122, 138);
const BODY: Color = Color::Rgb(216, 218, 228);
const RULE: Color = Color::Rgb(58, 62, 76);
const CODE_BG: Color = Color::Rgb(28, 30, 40);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Success,
    Warning,
    Error,
}

impl NoticeLevel {
    fn decoration(&self) -> (&'static str, Color) {
        match self {
            NoticeLevel::Success => ("✓", Color::LightGreen),
            NoticeLevel::Warning => ("▲", Color::Yellow),
            NoticeLevel::Error => ("✗", Color::Red),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TranscriptItem {
    UserGoal {
        text: String,
        timestamp: String,
    },
    AgentOutput {
        agent_id: String,
        agent_name: String,
        role: AgentRole,
        text: String,
        thoughts: Option<String>,
        is_streaming: bool,
    },
    ToolExecution {
        agent_name: String,
        tool_name: String,
        args: String,
        output: String,
        is_error: bool,
        duration_ms: u64,
        is_running: bool,
    },
    Milestone {
        step_title: String,
        step_index: usize,
        total_steps: usize,
        duration_ms: Option<u64>,
    },
    Notice {
        level: NoticeLevel,
        text: String,
    },
}

/// What the last render actually measured, so key handling can clamp scrolling
/// to real content instead of guessing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViewportInfo {
    pub content_height: u16,
    pub view_height: u16,
}

impl ViewportInfo {
    pub fn max_scroll(&self) -> u16 {
        self.content_height.saturating_sub(self.view_height)
    }

    /// Apply a scroll delta, returning the new offset and whether the view is
    /// parked at the bottom (which re-arms follow mode).
    pub fn apply_scroll(&self, current: u16, following: bool, delta: i32) -> (u16, bool) {
        let max = self.max_scroll();
        let from = if following { max } else { current };
        let next = (from as i32 + delta).clamp(0, max as i32) as u16;
        (next, next >= max)
    }
}

/// Rows a line occupies once `Wrap` folds it at `width`.
///
/// `Paragraph::scroll` counts *wrapped* rows, so measuring unwrapped lines
/// under-reports the height and the tail of a long output becomes unreachable.
fn wrapped_height(line: &Line, width: u16) -> u16 {
    if width == 0 {
        return 1;
    }
    let width = width as usize;
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    if text.trim().is_empty() {
        return 1;
    }

    let mut rows: u16 = 1;
    let mut col = 0usize;
    for word in text.split_inclusive(' ') {
        let w = UnicodeWidthStr::width(word);
        if w > width {
            // A single unbroken token longer than the viewport spills over
            // however many rows it needs.
            rows = rows.saturating_add((w / width) as u16);
            col = w % width;
        } else if col + w > width {
            rows = rows.saturating_add(1);
            col = w;
        } else {
            col += w;
        }
    }
    rows.max(1)
}

fn empty_state() -> Vec<Line<'static>> {
    vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "  ⚡ Agent Orchestra is ready.",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled(
                "  Type a goal below and press ",
                Style::default().fg(Color::Gray),
            ),
            Span::styled(
                "[Enter]",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to start the pipeline.", Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Navigate  ", Style::default().fg(DIM)),
            Span::styled("j/k", Style::default().fg(Color::White)),
            Span::styled(" or wheel · ", Style::default().fg(DIM)),
            Span::styled("PgUp/PgDn", Style::default().fg(Color::White)),
            Span::styled(" · ", Style::default().fg(DIM)),
            Span::styled("g/G", Style::default().fg(Color::White)),
            Span::styled(" top/bottom", Style::default().fg(DIM)),
        ]),
        Line::from(vec![
            Span::styled("  Control   ", Style::default().fg(DIM)),
            Span::styled("t", Style::default().fg(Color::Magenta)),
            Span::styled(" topology · ", Style::default().fg(DIM)),
            Span::styled("m", Style::default().fg(Color::Cyan)),
            Span::styled(" model · ", Style::default().fg(DIM)),
            Span::styled("Esc", Style::default().fg(Color::Red)),
            Span::styled(" cancel run · ", Style::default().fg(DIM)),
            Span::styled("?", Style::default().fg(Color::Green)),
            Span::styled(" help", Style::default().fg(DIM)),
        ]),
    ]
}

fn render_user_goal(lines: &mut Vec<Line<'static>>, text: &str, timestamp: &str) {
    lines.push(Line::from(vec![
        Span::styled(
            " GOAL ",
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  {}", timestamp), Style::default().fg(DIM)),
    ]));
    for l in text.lines() {
        lines.push(Line::from(vec![Span::styled(
            format!("  {}", l),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )]));
    }
}

fn render_thoughts(lines: &mut Vec<Line<'static>>, thoughts: &str, is_streaming: bool) {
    let trimmed = thoughts.trim();
    if trimmed.is_empty() {
        return;
    }
    let all: Vec<&str> = trimmed.lines().filter(|l| !l.trim().is_empty()).collect();
    if all.is_empty() {
        return;
    }

    // Chain-of-thought is long and mostly noise once the answer exists. While
    // the agent is still thinking a live tail is useful; afterwards a single
    // summary line keeps the transcript readable.
    if !is_streaming {
        lines.push(Line::from(vec![Span::styled(
            format!("  💭 reasoning · {} lines", all.len()),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )]));
        return;
    }

    let start = all.len().saturating_sub(LIVE_THOUGHT_LINES);
    if start > 0 {
        lines.push(Line::from(vec![Span::styled(
            format!("  💭 reasoning · {} earlier lines", start),
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )]));
    }
    for l in &all[start..] {
        lines.push(Line::from(vec![Span::styled(
            format!("   │ {}", truncate_chars(l.trim(), 300)),
            Style::default()
                .fg(Color::Rgb(108, 110, 132))
                .add_modifier(Modifier::ITALIC),
        )]));
    }
}

fn render_agent_output(
    lines: &mut Vec<Line<'static>>,
    agent_name: &str,
    role: &AgentRole,
    text: &str,
    thoughts: &Option<String>,
    is_streaming: bool,
    spinner_idx: usize,
) {
    let color = role.default_color();
    let mut header = vec![
        Span::styled(
            format!(" {} {} ", role.icon(), agent_name.to_uppercase()),
            Style::default()
                .bg(color)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}", role.name()), Style::default().fg(color)),
    ];
    if is_streaming {
        header.push(Span::styled(
            format!(
                "  {} streaming",
                SPINNER_FRAMES[spinner_idx % SPINNER_FRAMES.len()]
            ),
            Style::default().fg(Color::LightGreen),
        ));
    }
    lines.push(Line::from(header));

    if let Some(th) = thoughts {
        render_thoughts(lines, th, is_streaming);
    }

    let mut in_code_block = false;
    for out_line in text.lines() {
        if out_line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            let lang = out_line.trim().trim_start_matches('`');
            let label = if in_code_block && !lang.is_empty() {
                format!("  ┌─ {} ", lang)
            } else if in_code_block {
                "  ┌─ code ".to_string()
            } else {
                "  └────────".to_string()
            };
            lines.push(Line::from(vec![Span::styled(
                label,
                Style::default().fg(Color::Rgb(140, 120, 60)),
            )]));
            continue;
        }

        if in_code_block {
            lines.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::Rgb(140, 120, 60))),
                Span::styled(
                    out_line.to_string(),
                    Style::default().fg(Color::Rgb(226, 214, 168)).bg(CODE_BG),
                ),
            ]));
        } else if let Some(heading) = out_line.strip_prefix('#') {
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", heading.trim_start_matches('#').trim()),
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            )]));
        } else if out_line.trim_start().starts_with("- ") || out_line.trim_start().starts_with("* ")
        {
            let body = out_line.trim_start()[2..].to_string();
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(color)),
                Span::styled(body, Style::default().fg(BODY)),
            ]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                format!("  {}", out_line),
                Style::default().fg(BODY),
            )]));
        }
    }
}

/// Fields of a `ToolExecution` entry, grouped so rendering takes one argument
/// instead of nine positional ones.
struct ToolRender<'a> {
    agent_name: &'a str,
    tool_name: &'a str,
    args: &'a str,
    output: &'a str,
    is_error: bool,
    duration_ms: u64,
    is_running: bool,
}

fn render_tool_execution(lines: &mut Vec<Line<'static>>, t: &ToolRender<'_>, spinner_idx: usize) {
    let ToolRender {
        agent_name,
        tool_name,
        args,
        output,
        is_error,
        duration_ms,
        is_running,
    } = *t;
    let (status_text, status_col) = if is_running {
        (
            format!(
                "{} running",
                SPINNER_FRAMES[spinner_idx % SPINNER_FRAMES.len()]
            ),
            Color::Yellow,
        )
    } else if is_error {
        ("failed".to_string(), Color::Red)
    } else {
        (format!("ok · {}ms", duration_ms), Color::LightGreen)
    };

    lines.push(Line::from(vec![
        Span::styled(
            " TOOL ",
            Style::default()
                .bg(Color::Magenta)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {} → ", agent_name),
            Style::default().fg(Color::Gray),
        ),
        Span::styled(
            tool_name.to_string(),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {}", status_text),
            Style::default().fg(status_col),
        ),
    ]));

    // Byte slicing here panicked whenever a tool argument held a path or a
    // string with any non-ASCII character in the first 100 bytes.
    lines.push(Line::from(vec![Span::styled(
        format!("   ├ args: {}", truncate_chars(args, 160)),
        Style::default().fg(DIM),
    )]));

    for out_line in output.lines().take(10) {
        lines.push(Line::from(vec![Span::styled(
            format!("   │ {}", truncate_chars(out_line, 400)),
            Style::default().fg(if is_error {
                Color::Rgb(210, 140, 140)
            } else {
                Color::Rgb(158, 160, 178)
            }),
        )]));
    }
    let extra = output.lines().count().saturating_sub(10);
    if extra > 0 {
        lines.push(Line::from(vec![Span::styled(
            format!("   └ … {} more lines", extra),
            Style::default().fg(DIM),
        )]));
    }
}

fn render_milestone(
    lines: &mut Vec<Line<'static>>,
    step_title: &str,
    step_index: usize,
    total_steps: usize,
    duration_ms: Option<u64>,
) {
    let status = match duration_ms {
        Some(d) => format!(" · {:.1}s", d as f64 / 1000.0),
        None => " · running".to_string(),
    };
    lines.push(Line::from(vec![
        Span::styled(
            format!(" ◆ STEP {}/{} ", step_index, total_steps),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {}", step_title),
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(status, Style::default().fg(DIM)),
        Span::styled(" ────────────────", Style::default().fg(RULE)),
    ]));
}

pub fn render_transcript(
    f: &mut Frame,
    area: Rect,
    items: &[TranscriptItem],
    scroll_offset: u16,
    auto_scroll: bool,
    spinner_idx: usize,
) -> ViewportInfo {
    let inner_width = area.width.saturating_sub(2);
    let view_height = area.height.saturating_sub(2);

    let mut lines: Vec<Line> = if items.is_empty() {
        empty_state()
    } else {
        Vec::new()
    };

    for item in items {
        lines.push(Line::from(""));
        match item {
            TranscriptItem::UserGoal { text, timestamp } => {
                render_user_goal(&mut lines, text, timestamp)
            }
            TranscriptItem::AgentOutput {
                agent_name,
                role,
                text,
                thoughts,
                is_streaming,
                ..
            } => render_agent_output(
                &mut lines,
                agent_name,
                role,
                text,
                thoughts,
                *is_streaming,
                spinner_idx,
            ),
            TranscriptItem::ToolExecution {
                agent_name,
                tool_name,
                args,
                output,
                is_error,
                duration_ms,
                is_running,
            } => render_tool_execution(
                &mut lines,
                &ToolRender {
                    agent_name,
                    tool_name,
                    args,
                    output,
                    is_error: *is_error,
                    duration_ms: *duration_ms,
                    is_running: *is_running,
                },
                spinner_idx,
            ),
            TranscriptItem::Milestone {
                step_title,
                step_index,
                total_steps,
                duration_ms,
            } => render_milestone(
                &mut lines,
                step_title,
                *step_index,
                *total_steps,
                *duration_ms,
            ),
            TranscriptItem::Notice { level, text } => {
                let (icon, color) = level.decoration();
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {} ", icon),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(text.clone(), Style::default().fg(color)),
                ]));
            }
        }
    }

    let content_height: u16 = lines
        .iter()
        .map(|l| wrapped_height(l, inner_width))
        .fold(0u16, |acc, h| acc.saturating_add(h));

    let viewport = ViewportInfo {
        content_height,
        view_height,
    };
    let max_scroll = viewport.max_scroll();
    let effective_scroll = if auto_scroll {
        max_scroll
    } else {
        scroll_offset.min(max_scroll)
    };

    let position_hint = if max_scroll == 0 {
        String::new()
    } else if auto_scroll {
        " following ".to_string()
    } else {
        format!(
            " {}% ",
            (effective_scroll as f32 / max_scroll as f32 * 100.0).round() as u16
        )
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(RULE))
        .title(Span::styled(
            " Live Multi-Agent Workspace ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(position_hint, Style::default().fg(DIM)));

    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((effective_scroll, 0)),
        area,
    );

    if max_scroll > 0 {
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_style(Style::default().fg(RULE))
                .thumb_style(Style::default().fg(Color::Rgb(110, 115, 135))),
            area,
            &mut ScrollbarState::new(max_scroll as usize).position(effective_scroll as usize),
        );
    }

    viewport
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(text: &str) -> Line<'static> {
        Line::from(vec![Span::raw(text.to_string())])
    }

    #[test]
    fn short_line_occupies_one_row() {
        assert_eq!(wrapped_height(&plain("hello"), 40), 1);
        assert_eq!(wrapped_height(&plain(""), 40), 1);
    }

    #[test]
    fn long_line_wraps_into_multiple_rows() {
        let text = "word ".repeat(40); // 200 columns
        let rows = wrapped_height(&plain(&text), 40);
        assert!(
            rows >= 5,
            "expected >=5 rows for 200 cols at width 40, got {}",
            rows
        );
    }

    #[test]
    fn unbroken_token_longer_than_viewport_still_wraps() {
        let rows = wrapped_height(&plain(&"x".repeat(200)), 40);
        assert!(rows >= 5, "got {}", rows);
    }

    #[test]
    fn wide_characters_count_by_display_width() {
        // Each CJK glyph is two columns, so 30 of them exceed a 40-column line.
        let rows = wrapped_height(&plain(&"日".repeat(30)), 40);
        assert!(rows >= 2, "got {}", rows);
    }

    #[test]
    fn scrolling_up_from_follow_mode_detaches_and_back_down_reattaches() {
        let v = ViewportInfo {
            content_height: 100,
            view_height: 20,
        }; // max 80

        // Following, then scroll up 3: leaves follow mode at 77.
        let (offset, following) = v.apply_scroll(0, true, -3);
        assert_eq!((offset, following), (77, false));

        // Back down 3: returns to the bottom and follow mode re-arms.
        let (offset, following) = v.apply_scroll(offset, false, 3);
        assert_eq!((offset, following), (80, true));

        // Cannot scroll above the top or past the bottom.
        assert_eq!(v.apply_scroll(2, false, -50), (0, false));
        assert_eq!(v.apply_scroll(10, false, 5000), (80, true));
    }

    #[test]
    fn content_shorter_than_the_viewport_always_follows() {
        let v = ViewportInfo {
            content_height: 5,
            view_height: 40,
        };
        assert_eq!(v.apply_scroll(0, true, -10), (0, true));
    }

    #[test]
    fn max_scroll_never_goes_negative() {
        let v = ViewportInfo {
            content_height: 3,
            view_height: 20,
        };
        assert_eq!(v.max_scroll(), 0);
        let v = ViewportInfo {
            content_height: 50,
            view_height: 20,
        };
        assert_eq!(v.max_scroll(), 30);
    }
}
